# Batch predict on Google Cloud Batch

This folder contains everything required for building a container to run AtmoML predict on Google Cloud Batch.
To run batch prediction, use `usl_models/notebooks/atmoml_batch_predict.ipynb`.

## Contents

* `usl_models/batch_predict/atmo_ml/main.py` is the main program that runs when the Docker container.
* `usl_models/batch_predict/atmo_ml/Dockerfile` defines the Docker container.
* `cloudbuild.yaml` describes how to build the Docker container and push it to GCP.

## Build and push container

To build and push the Docker container:

```sh
gcloud builds submit --config="usl_models/batch_predict/atmo_ml/cloudbuild.yaml" "."
```

NOTE: this does not need to be run unless you have a code change!
