"""
pipeline.py
------------
Wires every module in this package into the single stream Person 2 is
responsible for delivering to the rest of the Canopy team:

    {cell_id, estimate, confidence, hotspot, source_type, source_location}

One call to `Pipeline.run_timestep(...)` = one full pass:

    readings -> calibrate -> fuse (Kalman + wind advection)
             -> anomaly/hotspot flag (CUSUM local-baseline + GP hidden-hotspot)
             -> for each confirmed hotspot: attribute source, localise it
             -> emit events matching the shared schema
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

from .data_ingestion import Reading, stability_class_from_met
from .calibration import calibrate_reading, CalibrationParams
from .fusion import GridConfig, KalmanGridFusion
from .anomaly_detection import CusumAnomalyDetector, local_spatial_baseline
from .hotspot_detection import GPHotspotDetector, HotspotConfig
from .source_attribution import SourceAttributionClassifier, FEATURE_NAMES
from .plume_localization import localize_source


@dataclass
class HotspotEvent:
    cell_id: str
    ts: float
    estimate: float
    confidence: float
    hotspot: bool
    hidden: bool
    source_type: Optional[str]
    source_confidence: Optional[float]
    source_location: Optional[Tuple[float, float]]
    emission_rate_g_s: Optional[float]

    def to_dict(self):
        return asdict(self)


class Pipeline:
    def __init__(self, grid: GridConfig, grid_origin_km: Tuple[float, float],
                 fire_feed=None, calibration_params: CalibrationParams = CalibrationParams()):
        self.grid = grid
        self.origin = grid_origin_km
        self.fusion = KalmanGridFusion(grid)
        self.cusum = CusumAnomalyDetector()
        self.hotspot_detector = GPHotspotDetector(HotspotConfig())
        self.attributor = SourceAttributionClassifier()
        self.fire_feed = fire_feed
        self.calib_params = calibration_params
        self._attributor_trained = False

    def train_attributor(self, training_df):
        metrics = self.attributor.train(training_df)
        self._attributor_trained = True
        return metrics

    # -- helpers -----------------------------------------------------------

    def _calibrate(self, readings: List[Reading]) -> List[Reading]:
        out = []
        for r in readings:
            rh = r.met.get("rh", 50.0)
            corrected_pm25 = calibrate_reading(r.pollutants["pm25"], rh,
                                                days_since_calibration=0.0,
                                                params=self.calib_params)
            pollutants = dict(r.pollutants)
            pollutants["pm25"] = corrected_pm25
            out.append(Reading(node_id=r.node_id, ts=r.ts, lat=r.lat, lon=r.lon,
                                pollutants=pollutants, met=r.met,
                                source_type=r.source_type, trust=r.trust))
        return out

    def _nearest_fire_distance_km(self, x_km, y_km, active_fires) -> float:
        if not active_fires:
            return 999.0
        dists = [np.hypot(x_km - (f["lat"] - self.origin[0]),
                           y_km - (f["lon"] - self.origin[1])) for f in active_fires]
        return float(min(dists))

    # -- main entry point ----------------------------------------------------

    def run_timestep(self, ts: float, readings: List[Reading],
                      station_ids: set, wind_speed: float, wind_dir_deg: float,
                      true_hotspot_cells: Optional[List[Tuple[int, int]]] = None,
                      true_source_xy_km: Optional[Tuple[float, float]] = None
                      ) -> List[HotspotEvent]:

        calibrated = self._calibrate(readings)

        # --- physics-informed fusion ---
        self.fusion.predict(wind_speed, wind_dir_deg, dt_hours=1.0)
        self.fusion.update(calibrated, self.origin)
        mean_grid, std_grid = self.fusion.snapshot()

        # --- CUSUM streaming anomaly check (cheap, runs every timestep) ---
        baseline = local_spatial_baseline(mean_grid)
        anomaly_flags = self.cusum.update(mean_grid, baseline)

        # --- GP hidden-hotspot detection (spatial kriging + persistence) ---
        obs_xy, obs_vals, station_xy = [], [], []
        for r in calibrated:
            x_km, y_km = r.lat - self.origin[0], r.lon - self.origin[1]
            obs_xy.append([x_km, y_km])
            obs_vals.append(r.pollutants["pm25"])
            if r.node_id in station_ids:
                station_xy.append([x_km, y_km])
        obs_xy = np.array(obs_xy)
        obs_vals = np.array(obs_vals)
        station_xy = np.array(station_xy) if station_xy else np.empty((0, 2))

        query_ij = [(i, j) for i in range(self.grid.nx) for j in range(self.grid.ny)]
        query_xy = np.array([self.grid.cell_center(i, j) for i, j in query_ij])

        gp_hotspots = self.hotspot_detector.detect(obs_xy, obs_vals, query_xy,
                                                     query_ij, station_xy)

        active_fires = self.fire_feed.fetch(ts) if self.fire_feed else []
        hour_of_day = (ts / 3600.0) % 24
        is_burning_season = 1 if _is_burning_season(ts) else 0

        events: List[HotspotEvent] = []
        for i in range(self.grid.nx):
            for j in range(self.grid.ny):
                cell_id = self.grid.cell_id(i, j)
                val = float(mean_grid[i, j])
                sd = float(std_grid[i, j])
                confidence = float(np.clip(1.0 - sd / (val + 1e-6), 0.0, 1.0))
                is_hotspot = anomaly_flags[i, j] or (i, j) in gp_hotspots

                source_type = source_conf = source_loc = emission_rate = None

                if is_hotspot and self._attributor_trained:
                    x_km, y_km = self.grid.cell_center(i, j)
                    fire_dist = self._nearest_fire_distance_km(x_km, y_km, active_fires)
                    pm10 = val / 0.6   # placeholder ratio if PM10 wasn't separately fused
                    features = dict(
                        pm25_pm10_ratio=val / max(pm10, 1e-6),
                        no2_so2_ratio=1.5,     # would come from the fused multi-pollutant grid
                        co_pm25_ratio=0.2,
                        fire_distance_km=fire_dist,
                        hour_of_day=hour_of_day,
                        is_burning_season=is_burning_season,
                        wind_from_industrial_zone=0,
                    )
                    result = self.attributor.predict(features)
                    source_type, source_conf = result.label, result.confidence

                    # localisation runs for any confirmed hotspot attributed to a
                    # point-source-like origin, using whichever nearby readings
                    # are available — it does not require the GP detector
                    # specifically to have flagged the cell, since the GP's
                    # smoothing length-scale can under-resolve a narrow plume
                    # that a sparse sensor network only barely samples (the
                    # CUSUM local-baseline check catches those cases instead)
                    if source_type in ("industrial", "crop_burning"):
                        neighborhood_obs = [
                            (obs_xy[k][0], obs_xy[k][1], obs_vals[k])
                            for k in range(len(obs_vals))
                            if np.hypot(obs_xy[k][0] - x_km, obs_xy[k][1] - y_km) < 4.0
                        ]
                        if len(neighborhood_obs) >= 3:
                            stability = stability_class_from_met(wind_speed, hour_of_day)
                            loc_result = localize_source(
                                neighborhood_obs, wind_speed, wind_dir_deg, stability,
                                search_bounds=(x_km - 3, x_km + 3, y_km - 3, y_km + 3),
                                true_source_xy_km=true_source_xy_km,
                            )
                            source_loc = loc_result.source_xy_km
                            emission_rate = loc_result.emission_rate_g_s

                events.append(HotspotEvent(
                    cell_id=cell_id, ts=ts, estimate=round(val, 2),
                    confidence=round(confidence, 3), hotspot=bool(is_hotspot),
                    hidden=bool(gp_hotspots.get((i, j), {}).get("hidden", False)),
                    source_type=source_type, source_confidence=source_conf,
                    source_location=source_loc, emission_rate_g_s=emission_rate,
                ))

        return events


def _is_burning_season(ts: float) -> bool:
    """Rough Oct-Nov burning-season flag from a unix timestamp (day-of-year based)."""
    day_of_year = (ts / 86400.0) % 365
    return 274 <= day_of_year <= 335   # approx Oct 1 - Dec 1
