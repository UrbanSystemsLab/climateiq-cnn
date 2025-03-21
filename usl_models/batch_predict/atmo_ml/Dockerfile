# docker build -f usl_models/batch_predict/atmo_ml/Dockerfile .
# gcloud builds submit --tag=us-central1-docker.pkg.dev/climateiq/usl-models/test-container:tag1 --region=us-central1 usl_models/batch_predict/atmo_ml/Dockerfile
FROM python:3.11

# Copy python and shell scripts
COPY usl_models /usl_models

# Install various dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r /usl_models/requirements.txt
