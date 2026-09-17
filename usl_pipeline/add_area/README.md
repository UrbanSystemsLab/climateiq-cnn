Getting Started
===============

Script to add a new area to ClimateIQ.

```bash
pip install -r ../study_area_uploader/requirements.txt
pip install -e ../usl_lib -e ../study_area_uploader
```

Set your project and bucket prefix first. `BUCKET_PREFIX` is read at import
time, so an unset value silently targets production:

```bash
gcloud config set project climateiq-test
export BUCKET_PREFIX=test-
```

Reading `gs://` paths needs application default credentials:
`gcloud auth application-default login`.

Usage
=====

```bash
python usl_pipeline/add_area/main.py \
  --country "United States" --city "Kansas City" \
  --non-green-area-soil-classes 1 2 \
  --verbose
```

Siblings are imported by plain module name, so run `main.py` by path (or from
this directory) rather than with `python -m`. `preprocess_h3cells.py` is also
runnable on its own.

Areas are named `<ISO3>_<STATE>_<City>` with the city in PascalCase, for example
`USA_MO_KansasCity`. The ISO code and state are read from the urban areas
shapefile, which carries an `ISO` (alpha-3) and a `State_Abbr` column, so the
name always matches the source data. The command above resolves to
`USA_MO_KansasCity`. Pass `--country-iso` or `--state` only to override.

Outside the US the shapefile has no state and stores `XX`, so a non-US area
resolves to a name like `GBR_XX_London` and the command warns. Pass `--state`
to set a real subdivision code.

Start with `--dry-run` to check the derived name and target paths without
touching GCS.

Data flow
============

1. **Preprocess.** Clips the city out of the H3 source tiles in
   `gs://raw-data-h3index`, producing a DEM, buildings, soil, green areas and a
   city boundary.
2. **Upload and chunk.** Runs `study_area_uploader`, which writes the study area
   files to `climateiq-study-areas/<area>/` and chunk archives to
   `climateiq-study-area-chunks/<area>/`. The chunk archives are what trigger
   the feature matrix cloud functions.
3. **Rainfall.** Generates NOAA Atlas 14 design storms for the city centroid and
   uploads them to `climateiq-flood-simulation-config/<area>_config/`.

Use `--work-dir` to point that scratch at a specific location; it defaults to a
temporary directory.

`--wait` polls the metastore until the study area reaches `rescaling-done`,
reporting any chunk errors it finds along the way.

Layout
======

```
main.py                CLI, naming, and the three-stage flow
preprocess_h3cells.py  H3 tiles -> city rasters and vectors
scripts/
  rainfall_scenario_generator.py  NOAA Atlas 14 -> design storm files
```

Development
===========

```bash
flake8 . && black --check . && mypy .
```
