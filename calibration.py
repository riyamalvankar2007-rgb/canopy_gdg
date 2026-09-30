"""
calibration.py
---------------
Corrects raw low-cost optical-sensor readings before they enter fusion.

Two effects modelled, both cited in the project report as required
corrections for PMS5003-class sensors:

    1. Humidity-driven hygroscopic growth: optical counters over-read
       PM2.5 in humid air because particles swell and scatter more light.
    2. Sensor drift: cheap optical counters read progressively high (or
       low) the longer they go without recalibration against a reference.

Both corrections are written from scratch (a small hand-fit growth-factor
curve plus a linear drift model), not pulled from a calibration library.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class CalibrationParams:
    kappa: float = 0.35          # hygroscopic growth shape parameter
    drift_rate_per_day: float = 0.0015   # fractional positive drift per day


def humidity_growth_factor(rh_percent: float, kappa: float = 0.35) -> float:
    """
    kappa-Kohler-style growth factor GF(RH): how much a particle's
    effective optical diameter (and hence measured mass) swells at a
    given relative humidity. RH is clipped below 95% to avoid the
    singularity as RH -> 100%.
    """
    rh = np.clip(rh_percent, 0.0, 95.0) / 100.0
    return float((1.0 - rh) ** (-kappa))


def drift_factor(days_since_calibration: float, rate_per_day: float = 0.0015) -> float:
    """Simple linear drift model: reading grows `rate_per_day` per day uncalibrated."""
    return float(1.0 + rate_per_day * max(0.0, days_since_calibration))


def calibrate_reading(raw_pm25: float, rh_percent: float,
                       days_since_calibration: float = 0.0,
                       params: CalibrationParams = CalibrationParams()) -> float:
    """
    Corrects a single raw PM2.5 reading for humidity swelling and drift.
    Applied as raw / (growth * drift) since both effects inflate the
    optical reading above the true dry mass concentration.
    """
    gf = humidity_growth_factor(rh_percent, params.kappa)
    df = drift_factor(days_since_calibration, params.drift_rate_per_day)
    corrected = raw_pm25 / (gf * df)
    return max(0.0, corrected)


def fit_calibration_params(raw_readings: np.ndarray, rh: np.ndarray,
                            reference_readings: np.ndarray) -> CalibrationParams:
    """
    Learns `kappa` from a batch of low-cost readings paired with a
    co-located reference-station reading (this is the "reference-station
    pairing" parameter from the project's model-parameter table).
    A simple 1-D grid search — enough for a hackathon-scale calibration
    fit without pulling in a full optimisation library.
    """
    best_kappa, best_err = 0.35, np.inf
    for kappa in np.linspace(0.05, 0.9, 40):
        gf = np.array([humidity_growth_factor(r, kappa) for r in rh])
        pred = raw_readings / gf
        err = np.mean((pred - reference_readings) ** 2)
        if err < best_err:
            best_err, best_kappa = err, kappa
    return CalibrationParams(kappa=best_kappa)
