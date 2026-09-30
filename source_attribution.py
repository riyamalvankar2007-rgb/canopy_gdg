"""
source_attribution.py
-----------------------
Classifies a confirmed hotspot into a source type:
    industrial | crop_burning | smog | construction_dust | traffic

Real per-source labels essentially don't exist at hyperlocal resolution
(the CEEW figures in the project report note only ~30% of non-attainment
cities have ever completed a source-apportionment study), so this module
follows the report's recommended fix: train on a SEMI-SYNTHETIC dataset
built from known chemical/physical signatures of each source type, with
noise added, rather than claiming access to real labels we don't have.

Features (all derived from data already flowing through the pipeline):
    pm25_pm10_ratio, no2_so2_ratio, co_pm25_ratio,
    fire_distance_km, hour_of_day, is_burning_season, wind_from_industrial_zone

The classifier itself (Random Forest) is an off-the-shelf model — what's
hand-built here is the semi-synthetic label generator, which encodes the
actual domain knowledge (which ratios/signatures distinguish each source).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

SOURCE_CLASSES = ["industrial", "crop_burning", "smog", "construction_dust", "traffic"]

FEATURE_NAMES = [
    "pm25_pm10_ratio", "no2_so2_ratio", "co_pm25_ratio",
    "fire_distance_km", "hour_of_day", "is_burning_season",
    "wind_from_industrial_zone",
]


def _signature(source: str, rng: np.random.Generator) -> Dict[str, float]:
    """
    Hand-encoded characteristic feature signature per source type, drawn
    from the pollutant-ratio and fire-proximity reasoning laid out in the
    project report. Each call draws one noisy sample from that source's
    signature distribution — this is the "semi-synthetic ground truth"
    generator, standing in for a real emissions inventory / source survey.
    """
    hour = rng.uniform(0, 24)
    if source == "industrial":
        return dict(
            pm25_pm10_ratio=rng.normal(0.55, 0.08),   # fine-particle dominated
            no2_so2_ratio=rng.normal(1.2, 0.3),        # stack emissions: both elevated, SO2 relatively higher
            co_pm25_ratio=rng.normal(0.15, 0.05),
            fire_distance_km=rng.uniform(5, 30),        # far from any fire
            hour_of_day=hour,
            is_burning_season=0,
            wind_from_industrial_zone=1,
        )
    if source == "crop_burning":
        return dict(
            pm25_pm10_ratio=rng.normal(0.75, 0.07),    # very fine smoke particles
            no2_so2_ratio=rng.normal(2.5, 0.5),         # combustion NOx dominant, low SO2
            co_pm25_ratio=rng.normal(0.35, 0.08),       # high CO from biomass combustion
            fire_distance_km=rng.uniform(0, 3),         # near an active FIRMS fire pixel
            hour_of_day=hour,
            is_burning_season=1,
            wind_from_industrial_zone=0,
        )
    if source == "smog":
        return dict(
            pm25_pm10_ratio=rng.normal(0.65, 0.1),
            no2_so2_ratio=rng.normal(1.8, 0.4),
            co_pm25_ratio=rng.normal(0.2, 0.06),
            fire_distance_km=rng.uniform(3, 25),
            hour_of_day=hour,
            is_burning_season=rng.choice([0, 1], p=[0.4, 0.6]),
            wind_from_industrial_zone=rng.choice([0, 1], p=[0.6, 0.4]),
        )
    if source == "construction_dust":
        return dict(
            pm25_pm10_ratio=rng.normal(0.25, 0.06),    # coarse-particle dominated
            no2_so2_ratio=rng.normal(0.8, 0.2),
            co_pm25_ratio=rng.normal(0.05, 0.02),
            fire_distance_km=rng.uniform(5, 30),
            hour_of_day=rng.uniform(7, 19),             # daytime construction hours
            is_burning_season=0,
            wind_from_industrial_zone=0,
        )
    if source == "traffic":
        return dict(
            pm25_pm10_ratio=rng.normal(0.5, 0.08),
            no2_so2_ratio=rng.normal(3.5, 0.6),          # NOx-dominant, very little SO2
            co_pm25_ratio=rng.normal(0.4, 0.1),          # high CO from vehicle exhaust
            fire_distance_km=rng.uniform(5, 30),
            hour_of_day=rng.choice(                     # rush-hour bimodal
                np.concatenate([rng.normal(8, 1, 1), rng.normal(19, 1, 1)])
            ),
            is_burning_season=0,
            wind_from_industrial_zone=0,
        )
    raise ValueError(source)


def generate_synthetic_training_data(n_per_class: int = 400, seed: int = 42) -> pd.DataFrame:
    """Builds the semi-synthetic labelled dataset used to train the classifier."""
    rng = np.random.default_rng(seed)
    rows = []
    for source in SOURCE_CLASSES:
        for _ in range(n_per_class):
            feats = _signature(source, rng)
            feats["label"] = source
            rows.append(feats)
    df = pd.DataFrame(rows)
    # clip ratios to physically sane, non-negative ranges
    for col in ["pm25_pm10_ratio", "no2_so2_ratio", "co_pm25_ratio", "fire_distance_km", "hour_of_day"]:
        df[col] = df[col].clip(lower=0)
    df["hour_of_day"] = df["hour_of_day"].clip(upper=23.99)
    return df


@dataclass
class AttributionResult:
    label: str
    confidence: float
    probabilities: Dict[str, float]


class SourceAttributionClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=0,
        )
        self.fitted = False

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        X = df[FEATURE_NAMES].to_numpy()
        y = df["label"].to_numpy()

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        cv_scores = cross_val_score(self.model, X, y, cv=skf, scoring="f1_macro")
        self.model.fit(X, y)
        self.fitted = True

        preds = self.model.predict(X)
        report = classification_report(y, preds, output_dict=True)
        cm = confusion_matrix(y, preds, labels=SOURCE_CLASSES)

        return dict(
            cv_macro_f1_mean=round(float(np.mean(cv_scores)), 3),
            cv_macro_f1_std=round(float(np.std(cv_scores)), 3),
            train_macro_f1=round(report["macro avg"]["f1-score"], 3),
            confusion_matrix=cm.tolist(),
            classes=SOURCE_CLASSES,
        )

    def predict(self, features: Dict[str, float]) -> AttributionResult:
        if not self.fitted:
            raise RuntimeError("Call .train() before .predict()")
        x = np.array([[features[f] for f in FEATURE_NAMES]])
        proba = self.model.predict_proba(x)[0]
        classes = self.model.classes_
        probs = {c: round(float(p), 3) for c, p in zip(classes, proba)}
        best_idx = int(np.argmax(proba))
        return AttributionResult(label=classes[best_idx],
                                 confidence=round(float(proba[best_idx]), 3),
                                 probabilities=probs)

    def feature_importances(self) -> Dict[str, float]:
        if not self.fitted:
            raise RuntimeError("Call .train() before .feature_importances()")
        return {f: round(float(imp), 3) for f, imp in
                zip(FEATURE_NAMES, self.model.feature_importances_)}