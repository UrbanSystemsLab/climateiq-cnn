Getting Started
===============

Script to add a new area to ClimateIQ.

```bash
pip install -r requirements.txt \
  --extra-index-url https://code.usgs.gov/api/v4/groups/859/-/packages/pypi/simple
pip install -e ../usl_lib -e ../study_area_uploader
```

* **`BUCKET_PREFIX`** chooses the buckets. `add_area` sets it to `test-` when
  unset, so the test environment is the default.
* **Project** selects the Firestore metastore. It comes from
  `GOOGLE_CLOUD_PROJECT`, or from `gcloud config` when that is unset.

```bash
# Test, the default
gcloud config set project climateiq-test
python usl_pipeline/add_area/main.py ...

# Production, both explicit
GOOGLE_CLOUD_PROJECT=climateiq BUCKET_PREFIX= python usl_pipeline/add_area/main.py ...
```

The derived bucket paths are printed at startup and by `--dry-run`, so check
them before a real run.

Reading `gs://` paths needs application default credentials:
`gcloud auth application-default login`.

`pfdf`, used by the rainfall stage, is served from a USGS package index rather
than PyPI, so install with the extra index shown above.

Usage
=====

```bash
python usl_pipeline/add_area/main.py \
  --country "United States" --city "Kansas City" \
  --non-green-area-soil-classes 1 2 \
  --verbose
```

`scripts/` is imported as a sibling package, so run `main.py` by path (or from
this directory) rather than with `python -m`. The stage scripts are also
runnable on their own.

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

Cloud Run Job
=============

Build and push:

```bash
gcloud builds submit --config=usl_pipeline/add_area/cloudbuild.yaml .
```

`BUCKET_PREFIX=test-` is the default and is set here; a production job
needs `--set-env-vars=BUCKET_PREFIX=` and the `climateiq` image and project:

```bash
gcloud run jobs create add-area \
  --image=us-central1-docker.pkg.dev/climateiq-test/usl-pipeline/add-area:dev \
  --region=us-central1 \
  --set-env-vars=BUCKET_PREFIX=test- \
  --set-env-vars=GOOGLE_CLOUD_PROJECT=climateiq-test \
  --service-account=<sa>@climateiq-test.iam.gserviceaccount.com \
  --memory=8Gi --cpu=4 \
  --task-timeout=3h --max-retries=0
```

Then run a city, overriding the args per execution:

```bash
gcloud run jobs execute add-area --region=us-central1 --wait \
  --args="--country=United States,--city=Watsonville,--state=CA,\
--non-green-area-soil-classes,0,--verbose"
```

The service account needs read on `gs://raw-data-h3index` (project:
`climateiq-test`), object admin on the study area, chunk, feature, label and
flood-config buckets, and `roles/datastore.user` for the metastore.

Layout
======

```
main.py    CLI, naming, and the three-stage flow
scripts/   stage scripts, also runnable on their own
  preprocess_h3cells.py           H3 tiles -> city rasters and vectors
  rainfall_scenario_generator.py  NOAA Atlas 14 -> design storm files
```

Development
===========

```bash
flake8 . && black --check . && mypy .
```
