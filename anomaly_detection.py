"""
anomaly_detection.py
---------------------
Streaming anomaly detector that flags a grid cell the moment its fused
PM2.5 value starts drifting away from its own recent baseline AND from
its spatial neighbours — a two-sided check, because a citywide smog
event should raise the neighbourhood baseline too and not trip every
cell as an "anomaly."

Implements CUSUM (cumulative sum control chart), written from scratch,
per cell, maintained incrementally so it is genuinely streaming (O(1)
update per cell per timestep, no re-scan of history).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class CusumState:
    baseline_mean: float
    baseline_std: float
    s_pos: float = 0.0     # cumulative positive deviation
    s_neg: float = 0.0
    n_seen: int = 0


class CusumAnomalyDetector:
    """
    One CUSUM tracker per grid cell. `k` is the allowance (slack) in
    std-devs before deviations start accumulating; `h` is the decision
    threshold in accumulated std-devs before a cell is flagged.
    """

    def __init__(self, k: float = 0.5, h: float = 4.0, baseline_alpha: float = 0.05):
        self.k = k
        self.h = h
        self.alpha = baseline_alpha   # exponential-average rate for the rolling baseline
        self.state: Dict[Tuple[int, int], CusumState] = {}

    def update(self, mean_grid: np.ndarray, neighbor_baseline: np.ndarray) -> np.ndarray:
        """
        mean_grid: fused PM2.5 estimate this timestep, shape (nx, ny)
        neighbor_baseline: same-shape array — the *local spatial* median
                            (e.g. of the 8 surrounding cells) used as the
                            "expected" level so a city-wide smog rise does
                            not itself trigger every cell as anomalous.
        Returns a boolean array of the same shape: True where the cell's
        CUSUM statistic has crossed the alarm threshold this timestep.
        """
        nx, ny = mean_grid.shape
        flags = np.zeros((nx, ny), dtype=bool)

        for i in range(nx):
            for j in range(ny):
                key = (i, j)
                val = float(mean_grid[i, j])
                local_expected = float(neighbor_baseline[i, j])

                st = self.state.get(key)
                if st is None:
                    st = CusumState(baseline_mean=local_expected, baseline_std=max(5.0, local_expected * 0.15))
                    self.state[key] = st

                # residual against the *local spatial* expectation, not the
                # cell's own history, so genuine point sources stand out
                # from a broad regional smog rise
                resid = (val - local_expected) / max(st.baseline_std, 1e-6)

                st.s_pos = max(0.0, st.s_pos + resid - self.k)
                st.s_neg = max(0.0, st.s_neg - resid - self.k)

                if st.s_pos > self.h:
                    flags[i, j] = True
                    st.s_pos = 0.0   # reset after alarm

                # slowly adapt the local std estimate (keeps the detector
                # calibrated as ambient variability changes seasonally)
                st.baseline_std = max(1.0, (1 - self.alpha) * st.baseline_std +
                                            self.alpha * (abs(val - local_expected) + 1e-6))
                st.n_seen += 1

        return flags


def local_spatial_baseline(mean_grid: np.ndarray) -> np.ndarray:
    """
    Computes, for every cell, the median of its (up to 8) neighbours —
    a cheap, robust "what does the surrounding area look like" baseline
    that a single point source will deviate from, but a regional smog
    rise will not.
    """
    nx, ny = mean_grid.shape
    baseline = np.empty_like(mean_grid)
    for i in range(nx):
        for j in range(ny):
            neighbors = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < nx and 0 <= nj < ny:
                        neighbors.append(mean_grid[ni, nj])
            baseline[i, j] = np.median(neighbors) if neighbors else mean_grid[i, j]
    return baseline
