"""
hotspot_detection.py
----------------------
Hidden-hotspot detector using Gaussian Process regression (kriging) over
the sparse sensor network, extended with a residual-based hotspot test
and explicit uncertainty — this is the module that directly targets the
"hidden hotspot" finding from Bhardwaj et al. (arXiv 2410.04309), whose
kriging + threshold-filter approach is the baseline this module is
designed to be compared against.

Pipeline:
    1. Fit a GP over the (sparse) sensor observations for this timestep.
    2. Predict the field at every grid-cell centre, with a predictive
       std (this is what a plain fixed-station network cannot give you:
       a confidence-aware estimate *between* stations).
    3. Flag "hidden hotspots": cells whose GP-predicted value exceeds a
       threshold (persistent + significant relative to predictive
       uncertainty) AND that are not co-located with an existing
       reference station — i.e. hotspots the official network would
       have missed.
    4. Track a short persistence window so single-timestep noise spikes
       don't count as hotspots (matches the "persistence window"
       parameter in the project's parameter table).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C


@dataclass
class HotspotConfig:
    length_scale_km: float = 1.5          # GP kernel length-scale
    hotspot_threshold: float = 90.0        # µg/m3, matches CPCB "poor" breakpoint region
    z_threshold: float = 1.5               # how many predictive-std above local mean to count
    persistence_window: int = 1            # consecutive timesteps required to confirm
    station_exclusion_radius_km: float = 0.5   # "not already known" radius around ref stations


class GPHotspotDetector:
    def __init__(self, config: HotspotConfig = HotspotConfig()):
        self.cfg = config
        self._persistence: Dict[Tuple[int, int], int] = {}

    def fit_predict(self, obs_xy: np.ndarray, obs_vals: np.ndarray,
                     query_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        obs_xy: (n_obs, 2) sensor coordinates in km
        obs_vals: (n_obs,) PM2.5 values at those sensors this timestep
        query_xy: (n_query, 2) grid-cell centre coordinates in km
        Returns (pred_mean, pred_std), each shape (n_query,).
        """
        if len(obs_xy) < 3:
            # too few points for a meaningful GP fit this timestep
            mean_val = float(np.mean(obs_vals)) if len(obs_vals) else 40.0
            return (np.full(len(query_xy), mean_val),
                    np.full(len(query_xy), 25.0))

        kernel = C(50.0, (1.0, 500.0)) * RBF(
            length_scale=self.cfg.length_scale_km, length_scale_bounds=(0.2, 10.0)
        ) + WhiteKernel(noise_level=9.0, noise_level_bounds=(1.0, 100.0))

        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                       n_restarts_optimizer=1, alpha=1e-6)
        gp.fit(obs_xy, obs_vals)
        mean, std = gp.predict(query_xy, return_std=True)
        return mean, std

    def detect(self, obs_xy: np.ndarray, obs_vals: np.ndarray,
               query_xy: np.ndarray, query_ij: List[Tuple[int, int]],
               station_xy: np.ndarray) -> Dict[Tuple[int, int], dict]:
        """
        Runs one timestep of hidden-hotspot detection.
        Returns {(i,j): {"value", "std", "confidence", "hidden"} } for
        every query cell that currently qualifies as a confirmed hotspot.
        """
        mean, std = self.fit_predict(obs_xy, obs_vals, query_xy)

        # local expectation = median of the predicted field, used to
        # test whether a cell is a significant local excess, not just
        # generally polluted along with everywhere else
        local_expectation = float(np.median(mean))

        results = {}
        for idx, (i, j) in enumerate(query_ij):
            val, sd = float(mean[idx]), float(std[idx])
            z = (val - local_expectation) / max(sd, 1e-6)

            is_candidate = (val >= self.cfg.hotspot_threshold) and (z >= self.cfg.z_threshold)

            if is_candidate:
                self._persistence[(i, j)] = self._persistence.get((i, j), 0) + 1
            else:
                self._persistence[(i, j)] = 0

            confirmed = self._persistence.get((i, j), 0) >= self.cfg.persistence_window

            if confirmed:
                # "hidden" = no reference station sits within the exclusion
                # radius of this cell, i.e. the official network would not
                # have this reading on its own
                if len(station_xy) > 0:
                    d = np.min(np.linalg.norm(station_xy - np.array(query_xy[idx]), axis=1))
                    hidden = d > self.cfg.station_exclusion_radius_km
                else:
                    hidden = True

                confidence = float(1.0 - np.exp(-max(z, 0) / 2.0))  # 0..1, saturates with z
                results[(i, j)] = dict(value=val, std=sd, z=z,
                                        confidence=round(confidence, 3), hidden=hidden)
        return results


def evaluate_hotspot_detection(detected: Dict[Tuple[int, int], dict],
                                true_hotspot_cells: List[Tuple[int, int]],
                                all_cells: List[Tuple[int, int]]) -> Dict[str, float]:
    """
    Precision / recall against injected semi-synthetic ground truth —
    the evaluation the project report calls for to compare against the
    95% precision / 88% recall reported by the Delhi hidden-hotspot study.
    """
    detected_set = set(detected.keys())
    true_set = set(true_hotspot_cells)
    tp = len(detected_set & true_set)
    fp = len(detected_set - true_set)
    fn = len(true_set - detected_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(precision=round(precision, 3), recall=round(recall, 3), f1=round(f1, 3))
