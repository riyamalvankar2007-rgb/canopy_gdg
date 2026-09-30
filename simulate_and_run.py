"""
simulate_and_run.py
---------------------
End-to-end demo of Person 2's full pipeline:

    1. Builds a semi-synthetic city (10 km x 10 km, 500 m grid cells).
    2. Plants a KNOWN hidden source (a crop-burning fire, off to one side,
       away from where the reference stations sit) — this is the "inject
       known emitters" evaluation strategy the project report calls for,
       so precision/recall/localisation-error numbers are measurable
       against ground truth, not just plausible-looking.
    3. Deploys a sparse set of reference (CPCB-style) stations and a
       denser, noisier citizen mesh, including faulty and malicious nodes.
    4. Runs several hourly timesteps through the full pipeline: calibrate
       -> fuse -> detect -> attribute -> localise.
    5. Prints the emitted event stream (the schema every other Canopy
       module consumes) and evaluation metrics, and saves a heatmap PNG.

Run:
    python -m canopy_person2.simulate_and_run
"""

from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data_ingestion import CPCBFeed, CitizenMeshFeed, FIRMSFeed
from .fusion import GridConfig
from .pipeline import Pipeline
from .source_attribution import generate_synthetic_training_data
from .hotspot_detection import evaluate_hotspot_detection


# ---------------------------------------------------------------------
# 1. Scenario setup
# ---------------------------------------------------------------------

SEED = 7
rng = np.random.default_rng(SEED)

CITY_SIZE_KM = 10.0
GRID_ORIGIN = (0.0, 0.0)   # local km coordinates; (lat, lon) fields in Reading are reused as km offsets
grid = GridConfig(xmin=0.0, xmax=CITY_SIZE_KM, ymin=0.0, ymax=CITY_SIZE_KM, cell_size_km=0.5)

# the ground-truth hidden source: a crop-burning fire at (8.2, 2.0) km,
# deliberately far from where we'll place the reference stations
TRUE_SOURCE_XY = (8.2, 2.0)
TRUE_SOURCE_Q = 40.0          # g/s emission rate
BACKGROUND_PM25 = 35.0        # ambient city PM2.5, µg/m3

# reference (CPCB-style) stations: sparse, clustered near the city centre,
# i.e. NOT near the true source -- this is what makes the source "hidden"
REF_STATIONS = [(2.0, 2.0, "cpcb-01"), (5.0, 5.0, "cpcb-02"),
                (7.5, 7.5, "cpcb-03"), (2.5, 7.0, "cpcb-04")]

FIRE_SITES = [(TRUE_SOURCE_XY[0], TRUE_SOURCE_XY[1], 0, 6 * 3600, 55.0)]  # active for 6 simulated hours

WIND_SPEED = 2.5        # m/s
WIND_DIR_FROM_DEG = 315.0   # wind from the NW -> plume drifts toward city centre (SE)


def true_field_fn(x_km, y_km, ts):
    """
    Ground-truth PM2.5 field used to generate what every sensor 'sees':
    background level + a Gaussian-plume contribution from the true source
    if the fire is currently active.
    """
    from .plume_localization import gaussian_plume_concentration
    val = BACKGROUND_PM25
    if 0 <= ts <= 6 * 3600:
        val += gaussian_plume_concentration((x_km, y_km), TRUE_SOURCE_XY, TRUE_SOURCE_Q,
                                             WIND_SPEED, WIND_DIR_FROM_DEG, stability="D")
    return dict(pm25=val, pm10=val / 0.7, no2=20 + rng.normal(0, 2),
                so2=8 + rng.normal(0, 1), co=400 + rng.normal(0, 30), o3=25 + rng.normal(0, 3))


def true_hotspot_cells() -> list:
    """Ground-truth hotspot cells: those within ~1.5km of the true source at t=3h."""
    cells = []
    for i in range(grid.nx):
        for j in range(grid.ny):
            cx, cy = grid.cell_center(i, j)
            from .plume_localization import gaussian_plume_concentration
            conc = BACKGROUND_PM25 + gaussian_plume_concentration(
                (cx, cy), TRUE_SOURCE_XY, TRUE_SOURCE_Q, WIND_SPEED, WIND_DIR_FROM_DEG, "D")
            if conc >= 90.0:
                cells.append((i, j))
    return cells


# ---------------------------------------------------------------------
# 2. Build feeds + pipeline
# ---------------------------------------------------------------------

def main():
    cpcb_feed = CPCBFeed(REF_STATIONS, rng)
    mesh_feed = CitizenMeshFeed(n_sensors=40, bounds=(0, CITY_SIZE_KM, 0, CITY_SIZE_KM),
                                 rng=rng, faulty_frac=0.1, malicious_frac=0.05)
    fire_feed = FIRMSFeed(FIRE_SITES, rng)

    pipeline = Pipeline(grid=grid, grid_origin_km=GRID_ORIGIN, fire_feed=fire_feed)

    print("Training source-attribution classifier on semi-synthetic data...")
    training_df = generate_synthetic_training_data(n_per_class=400, seed=SEED)
    train_metrics = pipeline.train_attributor(training_df)
    print(json.dumps(train_metrics, indent=2))
    print()

    station_ids = {sid for _, _, sid in REF_STATIONS}

    all_events_by_hour = []
    for hour in range(6):   # simulate 6 hourly timesteps while the fire is active
        ts = hour * 3600.0
        readings = cpcb_feed.fetch(ts, true_field_fn) + mesh_feed.fetch(ts, true_field_fn)

        events = pipeline.run_timestep(
            ts=ts, readings=readings, station_ids=station_ids,
            wind_speed=WIND_SPEED, wind_dir_deg=WIND_DIR_FROM_DEG,
            true_source_xy_km=TRUE_SOURCE_XY,
        )
        all_events_by_hour.append(events)

        hotspots_this_hour = [e for e in events if e.hotspot]
        print(f"--- t = hour {hour} --- {len(hotspots_this_hour)} hotspot cell(s) flagged")
        for e in hotspots_this_hour[:5]:
            print(f"   {e.cell_id}: est={e.estimate} µg/m3  conf={e.confidence}  "
                  f"hidden={e.hidden}  source={e.source_type} ({e.source_confidence})  "
                  f"loc={e.source_location}  Q={e.emission_rate_g_s} g/s")

    # ---------------------------------------------------------------------
    # 3. Evaluation against injected ground truth
    #    The true source is active for the whole 6-hour window and the
    #    wind is held fixed in this demo, so the ground-truth hotspot
    #    footprint doesn't move — evaluation aggregates detections across
    #    all 6 timesteps (a cell counts as "detected" if it was flagged at
    #    least once), which is the fairer test of the detector than any
    #    single noisy timestep.
    # ---------------------------------------------------------------------

    ever_detected = {}
    for hour_events in all_events_by_hour:
        for e in hour_events:
            if e.hotspot:
                key = (int(e.cell_id[1:4]), int(e.cell_id[5:8]))
                ever_detected[key] = dict(value=e.estimate, confidence=e.confidence)

    true_cells = true_hotspot_cells()
    all_cells = [(i, j) for i in range(grid.nx) for j in range(grid.ny)]

    metrics = evaluate_hotspot_detection(ever_detected, true_cells, all_cells)
    print(f"\nTrue hotspot cells (ground truth, {len(true_cells)} cells): {true_cells}")
    print(f"Ever-detected hotspot cells ({len(ever_detected)} cells): {list(ever_detected.keys())}")
    print("\n=== Hotspot detection evaluation (aggregated over 6-hour active window) ===")
    print(json.dumps(metrics, indent=2))
    print(f"(compare against Bhardwaj et al. baseline: precision 0.95, recall 0.88)")

    localized = [e for hour_events in all_events_by_hour for e in hour_events
                 if e.source_location is not None]
    if localized:
        errs = []
        for e in localized:
            sx, sy = e.source_location
            errs.append(float(np.hypot(sx - TRUE_SOURCE_XY[0], sy - TRUE_SOURCE_XY[1])))
        print(f"\nSource localisation error (km): mean={np.mean(errs):.3f}  "
              f"min={np.min(errs):.3f}  n_estimates={len(errs)}")
        attributed_correctly = sum(1 for e in localized if e.source_type == "crop_burning")
        print(f"Correctly attributed as crop_burning: {attributed_correctly}/{len(localized)}")

    # ---------------------------------------------------------------------
    # 4. Save the full event stream (this is Person 2's deliverable object)
    # ---------------------------------------------------------------------

    out_path = "canopy_person2_event_stream.json"
    serializable = [[e.to_dict() for e in hour_events] for hour_events in all_events_by_hour]
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nFull event stream saved to {out_path}")

    # ---------------------------------------------------------------------
    # 5. Visualisation: true field vs. detected hotspots (hour 3 snapshot,
    #    with markers for every cell EVER detected across the 6-hour window)
    # ---------------------------------------------------------------------

    eval_hour = 3
    events_h3 = {(int(e.cell_id[1:4]), int(e.cell_id[5:8])): e for e in all_events_by_hour[eval_hour]}
    detected_hotspots = ever_detected

    from .plume_localization import gaussian_plume_concentration
    true_grid = np.zeros((grid.nx, grid.ny))
    for i in range(grid.nx):
        for j in range(grid.ny):
            cx, cy = grid.cell_center(i, j)
            true_grid[i, j] = BACKGROUND_PM25 + gaussian_plume_concentration(
                (cx, cy), TRUE_SOURCE_XY, TRUE_SOURCE_Q, WIND_SPEED, WIND_DIR_FROM_DEG, "D")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    im0 = axes[0].imshow(true_grid.T, origin="lower", cmap="inferno",
                          extent=[0, CITY_SIZE_KM, 0, CITY_SIZE_KM])
    axes[0].scatter(*TRUE_SOURCE_XY, marker="*", s=250, c="cyan", edgecolor="black",
                     label="true source")
    axes[0].scatter([s[0] for s in REF_STATIONS], [s[1] for s in REF_STATIONS],
                     marker="^", s=80, c="white", edgecolor="black", label="reference stations")
    axes[0].set_title("Ground truth PM2.5 field (hour 3)")
    axes[0].legend(loc="upper left", fontsize=8)
    plt.colorbar(im0, ax=axes[0], label="µg/m3")

    est_grid = np.array([[events_h3.get((i, j)).estimate if events_h3.get((i, j)) else np.nan
                           for j in range(grid.ny)] for i in range(grid.nx)])
    im1 = axes[1].imshow(est_grid.T, origin="lower", cmap="inferno",
                          extent=[0, CITY_SIZE_KM, 0, CITY_SIZE_KM])
    hot_xy = [grid.cell_center(i, j) for (i, j) in detected_hotspots.keys()]
    if hot_xy:
        hx, hy = zip(*hot_xy)
        axes[1].scatter(hx, hy, marker="x", s=60, c="lime", label="detected hotspot")
    loc_xy = [e.source_location for e in localized]
    if loc_xy:
        lx, ly = zip(*loc_xy)
        axes[1].scatter(lx, ly, marker="P", s=120, c="red", edgecolor="black",
                         label="localised source estimate")
    axes[1].scatter(*TRUE_SOURCE_XY, marker="*", s=250, c="cyan", edgecolor="black")
    axes[1].set_title("Canopy fused estimate + detections (hour 3)")
    axes[1].legend(loc="upper left", fontsize=8)
    plt.colorbar(im1, ax=axes[1], label="µg/m3")

    plt.tight_layout()
    fig_path = "canopy_person2_hotspot_map.png"
    plt.savefig(fig_path, dpi=130)
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()
