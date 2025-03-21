# Batch predict on Google Cloud Batch

This folder runs batch prediction on Google Cloud batch.

## Local testing

```sh
python usl_models/batch_predict/atmo_ml/main.py
```

## Build and push container

```sh
gcloud builds submit --config=usl_models/batch_predict/atmo_ml/cloudbuild.yaml .
```

## Run batch inference job

```sh

```
