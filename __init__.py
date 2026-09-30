"""
Canopy — Person 2 module: Fusion, Hotspot Detection & Source Attribution
=========================================================================

This package implements the full "sensing-to-source" pipeline owned by
Person 2 in the Canopy project:

    raw readings -> calibration -> physics-informed fusion
                 -> anomaly / hotspot detection (Gaussian Process)
                 -> source attribution (classifier)
                 -> source localisation (inverse Gaussian plume)
                 -> event stream: {cell_id, estimate, confidence,
                                    hotspot, source_type, source_location}

Every module here is written from scratch (NumPy/SciPy/scikit-learn used
only as numerical building blocks, never as an off-the-shelf black box
for the actual fusion/detection/localisation logic), per the project's
"hand-built core AI" requirement.
"""
