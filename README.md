# Canopy — Person 2: Fusion, Hotspot Detection & Source Attribution

This is the full, runnable implementation of Person 2's role in the Canopy
project: the sensing-to-source pipeline that turns raw, noisy, sparse
sensor readings into the event stream every other Canopy module consumes.

```
raw readings -> calibration -> physics-informed fusion (Kalman + wind advection)
             -> anomaly / hidden-hotspot detection (CUSUM + Gaussian Process)
             -> source attribution (classifier)
             -> source localisation (inverse Gaussian plume)
             -> {cell_id, estimate, confidence, hotspot, source_type, source_location}
```

## Files

| File | What it does |
|---|---|
| `data_ingestion.py` | CPCB / citizen-mesh / FIRMS / meteorology feeds. **Simulated** — see "Real data" below. |
| `calibration.py` | Humidity-growth-factor and drift correction for low-cost PM2.5 sensors. |
| `fusion.py` | Hand-built per-cell Kalman filter bank with a wind-advection predict step. |
| `anomaly_detection.py` | Streaming CUSUM detector against a local spatial baseline. |
| `hotspot_detection.py` | Gaussian-process (kriging) hidden-hotspot detector with uncertainty and persistence. |
| `source_attribution.py` | Random-forest classifier + semi-synthetic training-data generator (industrial / crop_burning / smog / construction_dust / traffic). |
| `plume_localization.py` | From-scratch Gaussian-plume dispersion model + inverse source-location search. |
| `pipeline.py` | Orchestrates all of the above into the shared event schema. |
| `simulate_and_run.py` | End-to-end demo: builds a semi-synthetic city, injects a known hidden source, runs 6 timesteps, evaluates against ground truth, saves a JSON event stream and a PNG map. |

## Run it

```bash
pip install -r requirements.txt
python -m canopy_person2.simulate_and_run
```

This prints the hotspot events detected each hour, trains the source
classifier, evaluates hotspot detection against injected ground truth,
and writes:

- `canopy_person2_event_stream.json` — the full output object stream
- `canopy_person2_hotspot_map.png` — true field vs. detected/localised hotspots

## What's real data vs. simulated

This sandbox has no network access to CPCB, OpenAQ, NASA FIRMS or Google
Earth Engine (only pypi/npm/github-class domains are reachable here), so
`data_ingestion.py` **simulates** all four feeds with the correct schema
and realistic statistical behaviour (sensor noise, humidity bias, drift,
faulty and malicious nodes, a diurnal wind pattern). Every simulated feed
has a single `fetch()` method with a `# >>> REAL API HERE` comment marking
exactly where to drop in the real HTTP/API call — no downstream module
needs to change when real data is wired in.

**Before the actual hackathon, replace:**
- `CPCBFeed.fetch()` → CPCB / state SAMEER portal API
- `CitizenMeshFeed.fetch()` → OpenAQ v3 API + the Canopy mesh's own CRDT log
- `FIRMSFeed.fetch()` → NASA FIRMS area/CSV API
- `_sample_met()` → ERA5 (`cdsapi`) or IMD gridded fields

## What the demo scenario shows

A crop-burning fire is planted at a location deliberately far from all
four simulated reference stations — this is "hidden" by construction,
directly targeting the finding in Bhardwaj et al. (arXiv 2410.04309) that
sparse official networks miss real hotspots. Wind carries the plume
across a 10 km x 10 km, 500 m-resolution grid, sampled by 40 sparse,
noisy citizen-mesh sensors (10% faulty, 5% malicious) plus the 4 far-away
reference stations.

**Result on this run (seed=7):**
- Hotspot detection: precision 1.0, recall 0.25 (vs. the cited baseline's
  0.95 / 0.88 on a real, much denser deployment). Recall is limited by
  how narrow a well-collimated Gaussian plume is relative to sensor
  spacing at this density — a legitimate, reportable finding, not a bug.
  It directly motivates two "should-have" items from the project plan:
  active sensing (recommend where to add sensors) and a smaller GP
  length-scale tuned for narrow point-source plumes vs. broad smog.
- Source attribution: 3/3 correctly labelled `crop_burning` on this run;
  classifier cross-validated macro-F1 ≈ 0.98 on the semi-synthetic
  training set (report this number as semi-synthetic, not as real-world
  accuracy — there is no real labelled dataset at this resolution).
- Source localisation: ~2.4 km error from only 2-4 nearby noisy readings.

**Do not present these as production accuracy numbers.** They are
honest results on a small semi-synthetic scenario, meant to demonstrate
that the pipeline runs end-to-end and that its evaluation methodology
(inject a known source, measure precision/recall/localisation error)
works — exactly the "semi-synthetic ground truth" approach the project
report recommends, and the same kind of validation a reviewer will ask
for.

## Tuning knobs

All the parameters from the project's parameter tables are exposed as
constructor arguments / dataclasses, not hardcoded:

- `fusion.GridConfig` — grid resolution, bounds
- `fusion.KalmanGridFusion` — process/observation noise
- `anomaly_detection.CusumAnomalyDetector` — k (slack), h (threshold)
- `hotspot_detection.HotspotConfig` — GP length-scale, hotspot threshold,
  z-threshold, persistence window, station-exclusion radius
- `calibration.CalibrationParams` — humidity kappa, drift rate

## Interface contract with the rest of the team

Every emitted `HotspotEvent` matches the schema fixed on day 1:

```python
{cell_id, ts, estimate, confidence, hotspot, hidden,
 source_type, source_confidence, source_location, emission_rate_g_s}
```

Person 3 (forecasting) should consume `estimate`/`confidence` per cell
per timestep. Person 4 (alerts/dashboard/guidance) should consume
`hotspot`, `source_type`, and `source_location` to drive both the
authority-facing alert and the citizen-facing personal-guidance engine.
