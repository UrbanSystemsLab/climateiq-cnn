# usl_pipeline

Takes a city from raw geodata to flood predictions on the ClimateIQ dashboard. Four stages run in order, and each one's output starts or feeds the next.

| # | Stage | Runs on | Started by | Code |
|---|---|---|---|---|
| 1 | Add area | Cloud Run job `add-area` | by hand | `add_area/`, `study_area_uploader/` |
| 2 | Preprocessing | 12 Cloud Functions | Stage 1's uploads | `cloud_functions/main.py` |
| 3 | Prediction | On-prem HPC cluster | by hand | `../usl_models/`, see `3_prediction/README.md` |
| 4 | Post-processing | 3 Cloud Functions | Stage 3's uploads | `cloud_functions/postprocessing/` |

`usl_lib/` is shared by stages 1, 2 and 4.

**Conventions**
- `<area>` is a study area name, `<ISO3>_<STATE>_<City>`, for example `USA_GA_Atlanta`.
- Bucket names here leave out `BUCKET_PREFIX`: `test-` in climateiq-test, empty in climateiq.
- `raw-data-h3index`, `climateiq-h3-pipeline-1770319032` and `climateiq-h3-tiles` never take a prefix.

## Hand-offs between stages

| From → to | Location | Signal |
|---|---|---|
| 1 → 2 | `climateiq-study-areas/<area>/header.json` | Upload |
| 1 → 2 | `climateiq-study-area-chunks/<area>/chunk_<x>_<y>.tar` | Upload |
| 1 → 2 | `climateiq-flood-simulation-config/<area>_config/Rainfall_Data_<N>.txt` | Upload |
| 2 → 3 | `climateiq-study-area-feature-chunks/<area>/scaled_chunk_<x>_<y>.npy`, plus Firestore records | `study_areas/<area>` state is `rescaling-done` |
| 3 → 4 | `climateiq-predictions/flood_predictions/...`, see `3_prediction/README.md` | Mosaics for all 9 scenarios uploaded |
| 4 → dashboard | `climateiq-h3-tiles/<area>/h3_<area>.geojson`, `USA_Combined/admin.geojson` | Upload starts the Mapbox publish |

## Stage 1: Add area

1. Someone starts the `add-area` job, or runs `add_area/main.py` locally, with a city and country.
2. It clips source layers from `gs://raw-data-h3index` (`Working_files/`, `CONUS_Data_H3Index/`) and downloads NOAA Atlas 14 rainfall.
3. It calls `study_area_uploader`, which writes:
   - `climateiq-study-areas/<area>/`: `header.json`, `boundaries.txt`, `elevation.tif`, `buildings.txt`, `green_areas.txt`, `soil_classes.txt`
   - `climateiq-study-area-chunks/<area>/chunk_<x>_<y>.tar`
4. It writes 9 rainfall scenarios to `climateiq-flood-simulation-config/<area>_config/`.
5. With `--export-to-citycat`, it also writes `climateiq-flood-simulation-input` and `climateiq-flood-simulation-input-chunked`. No function reacts to these; they feed CityCAT simulation runs.
6. With `--wait`, it polls until the study area reaches `rescaling-done`.

## Stage 2: Preprocessing

Each function runs on an upload to the bucket shown, unless marked otherwise.

| Function | Entry point | Runs on | Does |
|---|---|---|---|
| `write-study-area-metadata` | `write_study_area_metadata` | `climateiq-study-areas`, `header.json` | Writes `study_areas/<area>` to Firestore |
| `generate-flood-feature-matrix` | `build_flood_feature_matrix` | `climateiq-study-area-chunks`, `*.tar` | Writes `<area>/chunk_<x>_<y>.npy` to the feature bucket, and a chunk record |
| `rescale-feature-matrix` | `rescale_feature_matrices` | `climateiq-study-area-feature-chunks` | Acts once the last chunk lands: writes `<area>/scaled_chunk_<x>_<y>.npy` and sets `rescaling-done` |
| `write-citycat-config-data` | `write_flood_scenario_metadata_and_features` | `climateiq-flood-simulation-config` | Writes `rainfall/<area>_config/Rainfall_Data_<N>.npy` to the feature bucket, and a `city_cat_rainfall_configs` record |
| `delete-citycat-config-metadata` | `delete_flood_scenario_metadata` | Deletion, same bucket | Removes the scenario record |
| `write-processed-citycat-labels` | `process_citycat_outputs` | `climateiq-flood-simulation-output` | Writes training labels to `climateiq-study-area-label-chunks` |
| `generate-feature-matrix-http` | `build_feature_matrix_http` | HTTP | Re-runs either feature build by hand |

**Heat path.** These run alongside the flood path and depend on the `wrf-python` wheel in `cloud_functions/wheels/`.

| Function | Entry point | Runs on |
|---|---|---|
| `generate-feature-matrix` (heat only, despite the name) | `build_heat_feature_matrix` | `climateiq-study-area-chunks` |
| `upload-wrf-study-area-chunk` | `build_and_upload_study_area_chunk` | `climateiq-atmospheric-simulation-input` |
| `write-wrf-heat-config-metadata` | `write_heat_scenario_config_metadata` | `climateiq-atmospheric-simulation-config` |
| `build-wrf-label-matrix` | `build_wrf_label_matrix` | `climateiq-atmospheric-simulation-output` |
| `build-wrf-label-matrix-http` | `build_wrf_label_matrix_http` | HTTP |

## Stage 3: Prediction

Manual, and the least documented stage. See `3_prediction/README.md`.

## Stage 4: Post-processing

| Function | Entry point | Runs on |
|---|---|---|
| `h3-pipeline-tiff-trigger` | `process_h3_pipeline_on_tiff_upload` | Upload to `climateiq-predictions` |
| `h3-pipeline` | `process_h3_pipeline` | HTTP, `{city_folder, predictions_bucket}` |
| `h3-tile-publish` | `publish_tiles` | HTTP, called by the `h3-tile-publish` workflow |

1. `h3-pipeline-tiff-trigger` waits until mosaics exist for all 9 scenarios. It skips the city if `h3_csv_cache/<city>_max.csv` already exists in the pipeline bucket.
2. The pipeline reads the mosaics, plus `admin_boundaries/` from `climateiq-h3-pipeline-1770319032`.
3. It writes working files and merged results to that same bucket: `h3_csv_cache/`, `intermediates/`, `water_polygons/`, `output/`.
4. It publishes to `climateiq-h3-tiles`: `<area>/h3_<area>.geojson`, `<area>/h3_sampled.csv` and `USA_Combined/admin.geojson`.
5. It syncs results to MongoDB, database `climateiq_data`, collections `h3` and `admin_level`. If that fails, it logs the error and carries on.
6. When `TILESERVER_BUCKET` is set, it also writes `tileserver/latest/<tileset>.pmtiles` to that bucket.
7. An upload of `<area>/h3_<area>.geojson` or `USA_Combined/admin.geojson` starts the `h3-tile-publish` workflow, which publishes to Mapbox.
   - `<user>.climateiq-h3-<area>` has layers `admin-boundary-hazard-grid-h3-09` to `-12`.
   - `<user>.climateiq-admin-national` has layers `admin-boundary-hazard-6`, `-8` and `-10`.
   - Status is recorded in `tile_sources/<tileset>/publish.json`.
   - To publish again by hand: `gcloud workflows run h3-tile-publish --location us-central1 --data='{"force": true, "data": {"bucket": "climateiq-h3-tiles", "name": "<area>/h3_<area>.geojson"}}'`

**Still manual**
- The dashboard reads the US-wide tilesets `timonm.region-us-grid-12` and `timonm.region-us-admin-level-hazard-v1`. Someone uploads `output/*.geojson` to them by hand.
- `tileserver/latest/` is never served directly. No code copies it to `tileserver/files/` or writes `tileserver/releases/`.

**Older export pipeline.** Uploads to `climateiq-predictions` also start `trigger-export-pipeline` → `spatialize-chunk-predictions` → `merge-scenario-predictions` → `export-to-aws`. That code lives in the climateiq-frontend repo.

## Where do I change...

| To change | Look in |
|---|---|
| How a city gets its name | `usl_lib/usl_lib/shared/area_names.py` |
| Which source layers are clipped | `add_area/scripts/preprocess_h3cells.py` |
| Rainfall scenarios | `add_area/scripts/rainfall_scenario_generator.py` |
| Chunking and tar contents | `study_area_uploader/study_area_uploader/chunkers/study_area_chunkers.py` |
| Feature channels | `usl_lib/usl_lib/transformers/feature_raster_transformers.py` |
| Feature rescaling | `_start_feature_rescaling_if_ready` in `cloud_functions/main.py` |
| Bucket names | `usl_lib/usl_lib/storage/cloud_storage.py` |
| Firestore schema | `usl_lib/usl_lib/storage/metastore.py`, and `../usl_models/usl_models/flood_ml/metastore.py`, which copies the names by hand |
| The flood model | `../usl_models/usl_models/flood_ml/` |
| Stage 4 entry points | `cloud_functions/postprocessing/entry_points.py` |
| The steps of a stage 4 run | `cloud_functions/postprocessing/city_pipeline.py` |
| H3 aggregation | `cloud_functions/postprocessing/cli_preprocess_tiff_to_h3csv.py` |
| H3 resolutions and scenario names | `cloud_functions/postprocessing/city_config.py` |
| Mapbox tilesets and layers | `cloud_functions/postprocessing/tile_publish.py` |
| PMTiles | `cloud_functions/postprocessing/pmtiles_build.py` |
| MongoDB sync | `cloud_functions/postprocessing/mongo_sync.py` |
| What gets deployed, memory, timeouts | `../.github/deploy/functions.json`, plus climateiq-terraform |
| What ships in a function | `../scripts/stage_cloud_functions.sh`, plus the zip lists in climateiq-terraform |

## Known gaps

- A new `add_area` city can't reach the tilesets yet. `_city_output_folder` turns `USA_GA_Atlanta` into `USA_GA_USA_GA_Atlanta`, which fails the area name check in `publish_tiles`.
- Stage 3 output paths are built from legacy names like `Chicago_Predictions`, not `<area>`. See `3_prediction/README.md`.

## Development

Each package installs and tests separately. `../.github/ci-matrix.json` has the exact install command for each one, and CI runs these from the repo root:

```sh
flake8 <path> --show-source --statistics
black --check <path>
pytest <path>
mypy <path>
```

`add_area` and `study_area_uploader` need GDAL.

### Integration tests

Use `pytest -k 'not integration'` to skip them. Running them requires the [Firestore emulator](https://cloud.google.com/firestore/docs/emulator): start it, set `FIRESTORE_EMULATOR_HOST` as its documentation describes, and run `pytest`.
