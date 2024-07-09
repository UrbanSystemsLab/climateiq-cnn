import os

STUDY_AREA_BUCKET = os.environ.get("BUCKET_PREFIX", "") + "climateiq-study-areas"
STUDY_AREA_CHUNKS_BUCKET = (
    os.environ.get("BUCKET_PREFIX", "") + "climateiq-study-area-chunks"
)
SIMULATION_CHUNKS_BUCKET = (
    os.environ.get("BUCKET_PREFIX", "") + "climateiq-simulation-chunks"
)
FEATURE_CHUNKS_BUCKET = (
    os.environ.get("BUCKET_PREFIX", "") + "climateiq-study-area-feature-chunks"
)
FLOOD_SIMULATION_INPUT_BUCKET = (
    os.environ.get("BUCKET_PREFIX", "") + "climateiq-flood-simulation-input"
)
FLOOD_SIMULATION_CONFIG_BUCKET = (
    os.environ.get("BUCKET_PREFIX", "") + "climateiq-flood-simulation-config"
)
LABEL_CHUNKS_BUCKET = (
    os.environ.get("BUCKET_PREFIX", "") + "climateiq-study-area-label-chunks"
)
