# docker build -f usl_models/batch_predict/atmo_ml/Dockerfile .
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Install Python
RUN apt-get update -y
RUN apt install python3.11 python3-pip -y

# Copy python and shell scripts
COPY usl_models /usl_models

# Set environment variables.
ENV PYTHONPATH="${PYTHONPATH}:/usl_models"
ENV PIP_ROOT_USER_ACTION="ignore"
ENV NVIDIA_VISIBLE_DEVICES="all"
ENV NVIDIA_DRIVER_CAPABILITIES="compute,utility"

# Install various dependencies
RUN python3.11 -m pip install --upgrade pip
RUN python3.11 -m pip install -r /usl_models/requirements.txt
