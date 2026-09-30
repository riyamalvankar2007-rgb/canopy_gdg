"""
plume_localization.py
-----------------------
Given a hotspot cell plus its nearby fused concentration readings and
the current wind field, estimates WHERE the emitting source actually
is (not just which grid cell is elevated) and how strong it is.

Implements the standard Gaussian plume dispersion equation from
scratch, then localises the source by a hand-written inverse search
(coordinate-descent-style grid refinement, no black-box optimiser) that
finds the (source_x, source_y, emission_rate) minimising squared error
against the observed concentration field.

    C(x,y) = Q / (2*pi*u*sigma_y*sigma_z)
             * exp(-y_c^2 / (2*sigma_y^2))
             * exp(-H^2 / (2*sigma_z^2))     [ground-level receptor, H = 0 => exp term = 1]

where x_c, y_c are receptor coordinates in the plume's own reference
frame (x_c downwind of the source, y_c crosswind), and sigma_y, sigma_z
are Pasquill-Gifford dispersion coefficients that grow with downwind
distance and depend on atmospheric stability class.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


# Pasquill-Gifford sigma_y, sigma_z power-law coefficients (rural, common
# tabulated approximation): sigma = a * x_km^b  (x in km, sigma in m)
_PG_COEFFS = {
    "A": dict(ay=0.22, by=0.894, az=0.20, bz=0.900),
    "B": dict(ay=0.16, by=0.894, az=0.12, bz=0.900),
    "C": dict(ay=0.11, by=0.894, az=0.08, bz=0.900),
    "D": dict(ay=0.08, by=0.894, az=0.06, bz=0.750),
    "E": dict(ay=0.06, by=0.894, az=0.03, bz=0.650),
    "F": dict(ay=0.04, by=0.894, az=0.016, bz=0.600),
}


def _dispersion_sigmas(downwind_km: float, stability: str) -> Tuple[float, float]:
    downwind_km = max(downwind_km, 0.01)
    c = _PG_COEFFS[stability]
    sigma_y = c["ay"] * (downwind_km ** c["by"]) * 1000.0   # -> metres
    sigma_z = c["az"] * (downwind_km ** c["bz"]) * 1000.0
    return sigma_y, sigma_z


def gaussian_plume_concentration(receptor_xy_km: Tuple[float, float],
                                  source_xy_km: Tuple[float, float],
                                  Q_g_per_s: float, wind_speed_mps: float,
                                  wind_dir_from_deg: float, stability: str) -> float:
    """
    Ground-level, ground-release Gaussian plume concentration (µg/m3) at
    one receptor point, given a candidate source location and strength.
    """
    dx = (receptor_xy_km[0] - source_xy_km[0]) * 1000.0   # metres, east
    dy = (receptor_xy_km[1] - source_xy_km[1]) * 1000.0   # metres, north

    # wind blows FROM wind_dir_from_deg, i.e. TOWARD (wind_dir_from_deg+180)
    theta = np.deg2rad((wind_dir_from_deg + 180.0) % 360.0)
    # rotate receptor offset into the plume frame: x_c downwind, y_c crosswind
    x_c = dx * np.sin(theta) + dy * np.cos(theta)
    y_c = dx * np.cos(theta) - dy * np.sin(theta)

    if x_c <= 0:
        return 0.0   # receptor is upwind of the candidate source: no contribution

    sigma_y, sigma_z = _dispersion_sigmas(x_c / 1000.0, stability)
    u = max(wind_speed_mps, 0.3)

    conc_g_m3 = (Q_g_per_s / (2 * np.pi * u * sigma_y * sigma_z) *
                 np.exp(-(y_c ** 2) / (2 * sigma_y ** 2)))
    return float(conc_g_m3 * 1e6)   # g/m3 -> µg/m3


@dataclass
class LocalizationResult:
    source_xy_km: Tuple[float, float]
    emission_rate_g_s: float
    residual: float
    localisation_error_km: Optional[float] = None


def localize_source(observed_points: List[Tuple[float, float, float]],
                     wind_speed_mps: float, wind_dir_from_deg: float,
                     stability: str, search_bounds: Tuple[float, float, float, float],
                     search_resolution_km: float = 0.25,
                     true_source_xy_km: Optional[Tuple[float, float]] = None
                     ) -> LocalizationResult:
    """
    observed_points: list of (x_km, y_km, concentration_ugm3) — the fused
                      readings around a confirmed hotspot.
    search_bounds: (xmin, xmax, ymin, ymax) in km to search for the source.

    Two-stage hand-written search (no scipy.optimize black box):
      Stage 1 — coarse grid over candidate (x, y); at each candidate,
                Q is solved in closed form by least-squares (concentration
                is LINEAR in Q for a fixed location), so only (x, y) need
                a grid search.
      Stage 2 — local refinement: a finer grid centred on the best coarse
                candidate, halving resolution twice.
    """
    xmin, xmax, ymin, ymax = search_bounds
    obs = np.array(observed_points)
    receptors = obs[:, :2]
    conc_obs = obs[:, 2]

    def best_Q_and_residual(sx, sy):
        # concentration = Q * unit_response(receptor | source, wind) for each receptor
        unit_resp = np.array([
            gaussian_plume_concentration((rx, ry), (sx, sy), 1.0,
                                          wind_speed_mps, wind_dir_from_deg, stability)
            for rx, ry in receptors
        ])
        denom = np.dot(unit_resp, unit_resp)
        if denom < 1e-12:
            return 0.0, np.inf
        Q = float(np.dot(unit_resp, conc_obs) / denom)   # closed-form least-squares
        Q = max(Q, 0.0)
        pred = Q * unit_resp
        residual = float(np.sum((pred - conc_obs) ** 2))
        return Q, residual

    def grid_search(x0, x1, y0, y1, res):
        xs = np.arange(x0, x1 + 1e-9, res)
        ys = np.arange(y0, y1 + 1e-9, res)
        best = (None, None, 0.0, np.inf)
        for sx in xs:
            for sy in ys:
                Q, resid = best_Q_and_residual(sx, sy)
                if resid < best[3]:
                    best = (sx, sy, Q, resid)
        return best

    # stage 1: coarse
    sx, sy, Q, resid = grid_search(xmin, xmax, ymin, ymax, search_resolution_km)

    # stage 2: two rounds of local refinement around the current best
    res = search_resolution_km
    for _ in range(2):
        res /= 4.0
        sx, sy, Q, resid = grid_search(sx - 2 * res, sx + 2 * res,
                                        sy - 2 * res, sy + 2 * res, res)

    loc_error = None
    if true_source_xy_km is not None:
        loc_error = float(np.hypot(sx - true_source_xy_km[0], sy - true_source_xy_km[1]))

    return LocalizationResult(source_xy_km=(round(sx, 3), round(sy, 3)),
                               emission_rate_g_s=round(Q, 3),
                               residual=round(resid, 4),
                               localisation_error_km=(round(loc_error, 3)
                                                       if loc_error is not None else None))
