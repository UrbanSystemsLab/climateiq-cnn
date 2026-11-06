# Stage 4: Post-processing

Turns prediction mosaics into H3 and admin results, then publishes them. `../README.md` lists the 3 Cloud Functions and what a run reads and writes.

## Where do I change...

| To change | Look in |
|---|---|
| Entry points | `postprocessing/entry_points.py` |
| The steps of a run | `postprocessing/city_pipeline.py` |
| H3 aggregation | `postprocessing/cli_preprocess_tiff_to_h3csv.py` |
| H3 resolutions and scenario names | `postprocessing/city_config.py` |
| Admin breadcrumbs and output folder names | `postprocessing/breadcrumbs.py` |
| Merged and published outputs | `postprocessing/combined_outputs.py` |
| Mapbox tilesets and layers | `postprocessing/tile_publish.py` |
| PMTiles | `postprocessing/pmtiles_build.py`, `postprocessing/pmtiles_publish.py` |
| MongoDB sync | `postprocessing/mongo_sync.py` |
| What gets deployed | `../../.github/deploy/functions.json`, group `postprocessing` |
| What ships in a function | `../../scripts/stage_postprocessing.sh` |

Keep `postprocessing/` flat: `pmtiles_build.py` re-runs itself with `python -m` from this folder, so it must sit one level inside the package.

## Running

From this folder, to run the pipeline by hand:

```sh
python -m postprocessing.cli_unified_pipeline --config cities_config.yaml
```

To run the checks, from the repository root: `scripts/test-local/postprocessing.sh`.
