#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="usl_models",
    packages=find_packages(),
    install_requires=[
        # Match the versions already present in the docker image
        # us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-14.py310:latest
        # we use for training.
        "numpy==1.26.4",
        "matplotlib==3.9.2",
        "tensorflow==2.15.1",
        "keras==2.15.0",
        "google-cloud-aiplatform==1.43.0",
        "google-cloud-storage==2.15.0",
        # firestore is not present in the image, but we match the cloud-storage version.
        "google-cloud-firestore==2.15.0",
        "seaborn==0.13.2",
        "keras-tuner[bayesian]=1.4.7",
    ],
    extras_require={
        "dev": [
            "requests",
            "black~=24.0",
            "flake8~=7.0",
            "flake8-docstrings~=1.7",
            "pytest~=8.0",
            "mypy~=1.9",
        ]
    },
)
