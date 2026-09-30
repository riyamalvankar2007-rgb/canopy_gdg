"""
fusion.py
---------
Hand-built sensor fusion: combines noisy, sparse, calibrated sensor
readings into one PM2.5 estimate (with uncertainty) per grid cell.

This is a per-cell Kalman filter bank where the PREDICT step is
physics-informed: instead of assuming each cell's value simply persists,
the previous field is advected along the wind vector before new
observations are folded in. This is what distinguishes it from a plain
statistical smoother — wind, not just recent history, drives the prior.

    predict:  x_t|t-1  = advect(x_{t-1}, wind)              (physics)
              P_t|t-1  = P_{t-1} + Q                        (process noise)

    update:   K        = P_t|t-1 / (P_t|t-1 + R)             (Kalman gain)
              x_t      = x_t|t-1 + K * (z_t - x_t|t-1)
              P_t      = (1 - K) * P_t|t-1

Multiple sensor observations landing in the same cell in one timestep are
first combined by inverse-variance weighting (weighted by each sensor's
trust score from the CRDT layer) before the Kalman update.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from .data_ingestion import Reading


@dataclass
class GridConfig:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    cell_size_km: float = 0.5     # matches the report's 500 m resolution parameter

    @property
    def nx(self) -> int:
        return max(1, int(np.ceil((self.xmax - self.xmin) / self.cell_size_km)))

    @property
    def ny(self) -> int:
        return max(1, int(np.ceil((self.ymax - self.ymin) / self.cell_size_km)))

    def cell_index(self, x_km: float, y_km: float) -> Optional[tuple]:
        i = int((x_km - self.xmin) / self.cell_size_km)
        j = int((y_km - self.ymin) / self.cell_size_km)
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return (i, j)
        return None

    def cell_id(self, i: int, j: int) -> str:
        return f"c{i:03d}_{j:03d}"

    def cell_center(self, i: int, j: int) -> tuple:
        return (self.xmin + (i + 0.5) * self.cell_size_km,
                self.ymin + (j + 0.5) * self.cell_size_km)


class KalmanGridFusion:
    """
    Maintains one scalar Kalman filter (mean + variance) per grid cell for
    PM2.5, with a wind-advection predict step shared across all cells.
    """

    def __init__(self, grid: GridConfig, process_noise: float = 4.0,
                 obs_noise: float = 9.0, init_value: float = 40.0,
                 init_var: float = 400.0):
        self.grid = grid
        self.Q = process_noise      # how much we trust the physics prediction to drift
        self.R = obs_noise          # base sensor observation noise (variance)
        self.mean = np.full((grid.nx, grid.ny), init_value, dtype=float)
        self.var = np.full((grid.nx, grid.ny), init_var, dtype=float)
        self.initialized = False

    # -- physics-informed predict step -----------------------------------

    def _advect(self, wind_speed: float, wind_dir_deg: float, dt_hours: float) -> np.ndarray:
        """
        Shifts the current field along the wind direction by the distance
        travelled in dt_hours, using bilinear resampling. wind_dir_deg is
        the meteorological "from" direction (e.g. 315 = wind FROM the NW,
        blowing toward the SE), converted to a displacement vector.
        """
        dist_km = wind_speed * dt_hours * 3.6   # m/s -> km/h -> km over dt_hours
        # meteorological convention: direction wind blows FROM; the plume
        # moves TOWARD (dir + 180)
        theta = np.deg2rad((wind_dir_deg + 180.0) % 360.0)
        dx_cells = (dist_km * np.sin(theta)) / self.grid.cell_size_km
        dy_cells = (dist_km * np.cos(theta)) / self.grid.cell_size_km

        advected = np.empty_like(self.mean)
        xs = np.arange(self.grid.nx)
        ys = np.arange(self.grid.ny)
        src_x = xs[:, None] - dx_cells
        src_y = ys[None, :] - dy_cells
        src_x = np.broadcast_to(src_x, (self.grid.nx, self.grid.ny))
        src_y = np.broadcast_to(src_y, (self.grid.nx, self.grid.ny))

        x0 = np.clip(np.floor(src_x).astype(int), 0, self.grid.nx - 1)
        x1 = np.clip(x0 + 1, 0, self.grid.nx - 1)
        y0 = np.clip(np.floor(src_y).astype(int), 0, self.grid.ny - 1)
        y1 = np.clip(y0 + 1, 0, self.grid.ny - 1)
        fx = np.clip(src_x - x0, 0, 1)
        fy = np.clip(src_y - y0, 0, 1)

        v00 = self.mean[x0, y0]; v10 = self.mean[x1, y0]
        v01 = self.mean[x0, y1]; v11 = self.mean[x1, y1]
        advected = (v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) +
                    v01 * (1 - fx) * fy + v11 * fx * fy)
        return advected

    def predict(self, wind_speed: float, wind_dir_deg: float, dt_hours: float = 1.0):
        self.mean = self._advect(wind_speed, wind_dir_deg, dt_hours)
        self.var = self.var + self.Q * dt_hours

    # -- observation update step ------------------------------------------

    def update(self, readings: List[Reading], grid_origin_km: tuple):
        """
        Folds calibrated sensor observations (already projected into local
        km coordinates by the caller) into the grid via a Kalman update.
        Multiple readings in one cell are first inverse-variance combined,
        weighted additionally by each sensor's CRDT trust score.
        """
        per_cell_obs = {}   # (i,j) -> list of (value, variance)
        ox, oy = grid_origin_km
        for r in readings:
            x_km, y_km = r.lat - ox, r.lon - oy   # caller passes lat/lon pre-projected to km
            idx = self.grid.cell_index(x_km, y_km)
            if idx is None:
                continue
            obs_var = self.R / max(r.trust, 0.05)   # low-trust sensors get inflated noise
            per_cell_obs.setdefault(idx, []).append((r.pollutants["pm25"], obs_var))

        for (i, j), obs_list in per_cell_obs.items():
            values = np.array([v for v, _ in obs_list])
            variances = np.array([var for _, var in obs_list])
            weights = 1.0 / variances
            z = float(np.sum(values * weights) / np.sum(weights))
            r_eff = float(1.0 / np.sum(weights))   # combined observation variance

            p_pred = self.var[i, j]
            k_gain = p_pred / (p_pred + r_eff)
            self.mean[i, j] = self.mean[i, j] + k_gain * (z - self.mean[i, j])
            self.var[i, j] = (1 - k_gain) * p_pred

        self.initialized = True

    def snapshot(self):
        """Returns (mean_grid, std_grid) for the current timestep."""
        return self.mean.copy(), np.sqrt(self.var).copy()
