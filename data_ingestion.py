"""
data_ingestion.py
------------------
Pulls the raw inputs Person 2's pipeline needs:

    - Government/reference stations      (CPCB-style)
    - Low-cost citizen/mesh sensors       (OpenAQ-style + Canopy mesh)
    - Fire/thermal-anomaly detections     (NASA FIRMS-style)
    - Meteorology                         (ERA5/IMD-style)

This sandbox has no network access to CPCB, OpenAQ, NASA FIRMS or GEE
(only pypi/npm/github-class domains are reachable), so every feed below
is a SIMULATOR that reproduces the real feed's schema and statistical
shape. Each class has a single method — fetch() — so swapping in a real
HTTP/API client later does not require touching any downstream module.

Real-API swap points are marked with  # >>> REAL API HERE
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ---------------------------------------------------------------------
# Shared reading schema (matches the project's fixed interface)
# ---------------------------------------------------------------------

@dataclass
class Reading:
    node_id: str
    ts: float                      # unix seconds
    lat: float
    lon: float
    pollutants: Dict[str, float]   # pm25, pm10, no2, so2, co, o3
    met: Dict[str, float]          # wind_speed, wind_dir_deg, rh, temp
    source_type: Optional[str] = None   # filled later by attribution
    trust: float = 1.0             # filled later by the CRDT trust layer


# ---------------------------------------------------------------------
# Reference-grade stations (CPCB-style): sparse, accurate, hourly
# ---------------------------------------------------------------------

class CPCBFeed:
    """Simulates a handful of accurate, sparsely-placed government stations."""

    def __init__(self, station_locations: List[tuple], rng: np.random.Generator):
        self.stations = station_locations   # [(lat, lon, node_id), ...]
        self.rng = rng

    def fetch(self, ts: float, true_field_fn) -> List[Reading]:
        # >>> REAL API HERE: replace this body with a call to
        # https://api.cpcb.gov.in (or the state SAMEER portal) and
        # parse the returned station JSON into Reading objects.
        readings = []
        for lat, lon, node_id in self.stations:
            true_vals = true_field_fn(lat, lon, ts)
            noisy = {k: max(0.0, v + self.rng.normal(0, 1.5)) for k, v in true_vals.items()}
            readings.append(Reading(
                node_id=node_id, ts=ts, lat=lat, lon=lon,
                pollutants=noisy,
                met=_sample_met(lat, lon, ts, self.rng),
                trust=0.95,   # reference stations start with high trust
            ))
        return readings


# ---------------------------------------------------------------------
# Low-cost citizen / mesh sensors (OpenAQ-style + Canopy's own mesh)
# ---------------------------------------------------------------------

class CitizenMeshFeed:
    """Simulates dense, noisy, occasionally faulty or malicious low-cost sensors."""

    def __init__(self, n_sensors: int, bounds: tuple, rng: np.random.Generator,
                 faulty_frac: float = 0.1, malicious_frac: float = 0.05):
        self.rng = rng
        (xmin, xmax, ymin, ymax) = bounds
        self.sensors = [
            dict(node_id=f"mesh-{i:03d}",
                 lat=rng.uniform(xmin, xmax), lon=rng.uniform(ymin, ymax),
                 faulty=rng.random() < faulty_frac,
                 malicious=rng.random() < malicious_frac,
                 drift_days=rng.uniform(0, 120),           # since last calibration
                 device="PMS5003")
            for i in range(n_sensors)
        ]

    def fetch(self, ts: float, true_field_fn) -> List[Reading]:
        # >>> REAL API HERE: replace with OpenAQ v3 /locations + /measurements
        # calls, or with reads from the Canopy mesh's own CRDT reading log.
        readings = []
        for s in self.sensors:
            true_vals = true_field_fn(s["lat"], s["lon"], ts)
            pm25 = true_vals["pm25"]

            if s["malicious"]:
                # a poisoning node: reports a wildly wrong value
                pm25 = max(0.0, self.rng.normal(20, 5))
            elif s["faulty"]:
                # stuck / flat-lined sensor
                pm25 = 15.0
            else:
                # normal low-cost sensor noise + humidity-driven positive bias
                rh = _sample_met(s["lat"], s["lon"], ts, self.rng)["rh"]
                humidity_bias = 1.0 + 0.3 * (rh / (100 - rh + 1e-6))
                drift_bias = 1.0 + 0.0015 * s["drift_days"]
                pm25 = max(0.0, pm25 * humidity_bias * drift_bias + self.rng.normal(0, 4))

            pollutants = dict(true_vals)
            pollutants["pm25"] = pm25
            readings.append(Reading(
                node_id=s["node_id"], ts=ts, lat=s["lat"], lon=s["lon"],
                pollutants=pollutants,
                met=_sample_met(s["lat"], s["lon"], ts, self.rng),
                trust=0.5,   # starts neutral; the CRDT trust layer updates this
            ))
        return readings


# ---------------------------------------------------------------------
# Fire / thermal anomaly feed (NASA FIRMS-style)
# ---------------------------------------------------------------------

class FIRMSFeed:
    """Simulates active-fire detections (crop burning, landfill fires)."""

    def __init__(self, fire_sites: List[tuple], rng: np.random.Generator):
        self.fire_sites = fire_sites   # [(lat, lon, active_from_ts, active_to_ts, frp), ...]
        self.rng = rng

    def fetch(self, ts: float) -> List[Dict]:
        # >>> REAL API HERE: replace with NASA FIRMS area/CSV API
        # (https://firms.modaps.eosdis.nasa.gov/api/) filtered to the study bbox.
        active = []
        for lat, lon, t0, t1, frp in self.fire_sites:
            if t0 <= ts <= t1:
                active.append(dict(lat=lat, lon=lon,
                                    frp=frp + self.rng.normal(0, 2),
                                    detected_ts=ts))
        return active


# ---------------------------------------------------------------------
# Meteorology (ERA5 / IMD-style)
# ---------------------------------------------------------------------

def _sample_met(lat: float, lon: float, ts: float, rng: np.random.Generator) -> Dict[str, float]:
    """
    Shared synthetic meteorology model used by every feed above, so all
    sensors see a spatially/temporally consistent (but noisy) wind field.

    >>> REAL API HERE: replace with ERA5 (cdsapi) or IMD gridded fields,
    interpolated to (lat, lon, ts).
    """
    hour = (ts / 3600.0) % 24
    # diurnal wind pattern: calmer at night, stronger midday, from the NW
    base_speed = 1.5 + 2.5 * max(0.0, np.sin((hour - 6) / 24 * 2 * np.pi))
    wind_speed = max(0.2, base_speed + rng.normal(0, 0.4))
    wind_dir_deg = (315 + rng.normal(0, 15)) % 360   # predominant NW wind
    rh = float(np.clip(55 + 25 * np.sin((hour - 4) / 24 * 2 * np.pi) + rng.normal(0, 5), 15, 98))
    temp = float(np.clip(22 + 8 * np.sin((hour - 9) / 24 * 2 * np.pi) + rng.normal(0, 1), 5, 45))
    blh = max(100.0, 800 + 600 * np.sin((hour - 13) / 24 * 2 * np.pi))  # boundary-layer height, m
    return dict(wind_speed=wind_speed, wind_dir_deg=wind_dir_deg, rh=rh, temp=temp, blh=blh)


def stability_class_from_met(wind_speed: float, hour: float) -> str:
    """
    Rough Pasquill-Gifford stability-class lookup from wind speed and
    time of day (day/night), used by the plume localisation module.
    Classes: A (very unstable) .. F (very stable).
    """
    is_day = 6 <= hour <= 18
    if wind_speed < 2:
        return "A" if is_day else "F"
    elif wind_speed < 3:
        return "B" if is_day else "E"
    elif wind_speed < 5:
        return "C" if is_day else "D"
    else:
        return "D"
