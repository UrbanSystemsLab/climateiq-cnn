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

The file names below are from the `USA_CA_Watsonville` run, which spans the
single H3 cell `822837fffffffff` and produces 20 chunks.

1. **Preprocess.** Clips the city out of the H3 source tiles in
   `gs://raw-data-h3index`, producing a DEM, buildings, soil, green areas and a
   city boundary. The city footprint comes from the urban areas shapefile; each
   intersecting H3 cell contributes five layers.

   ```
   gs://raw-data-h3index/
     Working_files/01_urban_areas_simplified_with_state.shp
     CONUS_Data_H3Index/822837fffffffff/
       Elevation/DEM_with_buildings_822837fffffffff.tif
       Buildings/Buildings_822837fffffffff.shp
       Green_spaces/green_spaces_822837fffffffff.tif
       Soil/Soil_texture_822837fffffffff.shp
       Land_use/Landcover_822837fffffffff.tif
   ```

   The clipped result is written locally, named after the area. Each `.shp`
   carries the usual `.cpg` / `.dbf` / `.prj` / `.shx` sidecars.

   ```
   <work-dir>/USA_CA_Watsonville/
     DEM_USA_CA_Watsonville.tif
     Landcover_USA_CA_Watsonville.tif
     Buildings_USA_CA_Watsonville.shp
     Soil_USA_CA_Watsonville.shp
     green_spaces_USA_CA_Watsonville.shp
     green_spaces_USA_CA_Watsonville.tif
     City_boundary_USA_CA_Watsonville.shp
     H3_cells_USA_CA_Watsonville.shp
     h3_index_list.txt
   ```

2. **Generate rainfall locally.** Downloads NOAA Atlas 14 data for the city
   centroid and generates the design storms. A download or generation failure
   stops the run before any spatial data is uploaded.

   ```
   <work-dir>/rainfall/
     noaa-atlas14-mean-pds-depth-english.csv
     Rainfall_Data_1.txt ... Rainfall_Data_9.txt
     Rainfall_Scenario_Summary.csv
   ```

3. **Upload and chunk.** Runs `study_area_uploader`, which writes the study area
   files to `climateiq-study-areas/<area>/` and chunk archives to
   `climateiq-study-area-chunks/<area>/`. The chunk archives are what trigger
   the feature matrix cloud functions. Then uploads the generated rainfall
   files to `climateiq-flood-simulation-config/<area>_config/`.

   ```
   climateiq-study-areas/USA_CA_Watsonville/
     header.json       elevation.tif     buildings.txt
     green_areas.txt   soil_classes.txt  boundaries.txt

   climateiq-study-area-chunks/USA_CA_Watsonville/
     chunk_0_0.tar ... chunk_4_3.tar

   climateiq-flood-simulation-config/USA_CA_Watsonville_config/
     Rainfall_Data_1.txt ... Rainfall_Data_9.txt
   ```

   The cloud functions then write one scaled feature matrix per chunk.

   ```
   climateiq-study-area-feature-chunks/USA_CA_Watsonville/
     scaled_chunk_0_0.npy ... scaled_chunk_4_3.npy
   ```

`--skip-rainfall` omits rainfall generation and upload.

Use `--work-dir` to place preprocessing and rainfall scratch at a specific
location; it defaults to a temporary directory. The uploader uses a separate
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
  --args="--country=United States,--city=Watsonville,--state=CA,--verbose"
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
