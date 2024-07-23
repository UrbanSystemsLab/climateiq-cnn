Getting Started
===============

To install the upload_study_area script:
- pip install -r requirements.txt
- pip install -e .

You should now be able to run the `upload_study_area` command.

Usage
=====

You can run the script with the `upload_study_area`
command after installation, or equivalently
`python study_area_uploader/main.py`.

Run with the `--help` for documentation of the available options:
```bash
python study_area_uploader/main.py --help
```

Be sure to set your gcloud project to the appropriate environment and
set the BUCKET_PREFIX environmental variable to 'test-' if running in
the test environment.

If uploading to the production climateiq project, run with:
```shell
gcloud config set project climateiq
python study_area_uploader/main.py ...
```

if uploading to the development climateiq-test project, run with:
```shell
gcloud config set project climateiq-test
BUCKET_PREFIX=test- python study_area_uploader/main.py ...
```

Running the script requires providing many different files defining
the study area geography.
The script will upload these files to GCS.
Once they are written to GCS, this will automatically trigger
processing of the files into machine learning and simulation inputs.
You can monitor the metastore to view processing progress.
