# Stage 3: Prediction

Runs by hand on the on-prem HPC cluster. The model is in `../../usl_models/`. `run_predict_city.py`, which predicts a city and builds the mosaics

## Inputs

An area is ready when `study_areas/<area>` in Firestore has state `rescaling-done`.

| What | Where |
|---|---|
| Feature chunks | `climateiq-study-area-feature-chunks/<area>/scaled_chunk_<x>_<y>.npy` |
| Rainfall vectors | `climateiq-study-area-feature-chunks/rainfall/<area>_config/Rainfall_Data_<N>.npy` |
| Chunk records | Firestore `study_areas/<area>/chunks/*`, field `feature_matrix_path` |
| Scenario records | Firestore `city_cat_rainfall_configs`, document ID `<area>_config%2FRainfall_Data_<N>.txt` |
| Training labels | `climateiq-study-area-label-chunks`, training only |

## Steps

1. Pull the inputs onto the cluster.
2. Predict every chunk for each of the 9 rainfall scenarios.
3. Merge each scenario's chunks into one mosaic, reprojected to WGS84.
4. Upload the results to the predictions bucket.

## Outputs

Stage 4 reads exactly this layout, in `climateiq-predictions`. `%2F` is a literal part of the object name, not a folder separator.

```
flood_predictions/<folder>/<prefix>%2FRainfall_Data_<N>.txt/<folder>_mosaic_peak_wgs84.tif
```

- `<folder>` is the city folder
- `<prefix>` is `<folder>` without `_Predictions`
- `<N>` is 1 to 9. Stage 4 starts only when all 9 exist.
- Per-chunk TIFs can sit beside the mosaic. Stage 4 ignores any `.tif` with `chunk` in its name.

## Before you upload

- **Environments:** the climateiq-test trigger watches `test-climateiq-predictions`. Uploading to `climateiq-predictions` doesn't reach it.
- **Re-predicting a city:** delete `h3_csv_cache/<city>_max.csv` from `climateiq-h3-pipeline-1770319032` first, or stage 4 skips it.
- **New `add_area` cities:** rainfall configs use `<area>_config/`, but stage 4 expects `<prefix>%2F...`, with no `_config`. Check the path before uploading.
