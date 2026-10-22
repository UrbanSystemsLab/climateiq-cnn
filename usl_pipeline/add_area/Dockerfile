# Cloud Run Job image for add_area.
#   podman build -f usl_pipeline/add_area/Dockerfile -t add-area .

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore
# Unbuffered so job logs stream rather than appearing only at exit.
ENV PYTHONUNBUFFERED=1

RUN apt-get update -qq \
    && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-gdal gdal-bin libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

COPY usl_pipeline /usl_pipeline

WORKDIR /usl_pipeline/add_area

RUN grep -v '^GDAL==' requirements.txt > /tmp/requirements.txt \
    && grep -v '^GDAL==' ../study_area_uploader/requirements.txt \
        > /tmp/uploader-requirements.txt \
    && sed -i 's|-r ../study_area_uploader/requirements.txt|-r /tmp/uploader-requirements.txt|' \
        /tmp/requirements.txt \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --ignore-installed \
        -r /tmp/requirements.txt \
        --extra-index-url https://code.usgs.gov/api/v4/groups/859/-/packages/pypi/simple \
    && python3 -m pip install --break-system-packages --no-deps \
        -e ../study_area_uploader

ENTRYPOINT ["python3", "main.py"]
