import os

STUDY_AREA_BUCKET = os.environ.get("BUCKET_PREFIX", "") + "climateiq-study-areas"
STUDY_AREA_CHUNKS_BUCKET = (
    os.environ.get("BUCKET_PREFIX", "") + "climateiq-study-area-chunks"
)
FEATURE_CHUNKS_BUCKET = (
    os.environ.get("BUCKET_PREFIX", "") + "climateiq-study-area-feature-chunks"
)
FLOOD_SIMULATION_INPUT_BUCKET = (
    os.environ.get("BUCKET_PREFIX", "") + "climateiq-flood-simulation-input"
)
