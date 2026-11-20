# Cloud Run job image for one city's H3 pipeline (postprocessing.city_pipeline).
#   docker build -f usl_pipeline/4_postprocessing/Dockerfile -t h3-pipeline .

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

COPY usl_pipeline/4_postprocessing/requirements.txt .
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt

COPY usl_pipeline/usl_lib/usl_lib ./usl_lib
COPY usl_pipeline/4_postprocessing/cities_config.yaml .
COPY usl_pipeline/4_postprocessing/postprocessing ./postprocessing

ENTRYPOINT ["python", "-m", "postprocessing.city_pipeline"]
