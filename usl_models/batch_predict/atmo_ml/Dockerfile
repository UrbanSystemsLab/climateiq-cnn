# docker build -f usl_models/batch_predict/atmo_ml/Dockerfile .
FROM python:3.11-slim

# Copy python and shell scripts
COPY usl_models /usl_models

# Set environment variables.
ENV PYTHONPATH="${PYTHONPATH}:/usl_models"
ENV PIP_ROOT_USER_ACTION="ignore"
ENV NVIDIA_VISIBLE_DEVICES="all"
ENV NVIDIA_DRIVER_CAPABILITIES="compute,utility"

# Install various dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r /usl_models/requirements.txt
