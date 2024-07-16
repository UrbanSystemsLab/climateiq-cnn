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
        "tensorflow==2.14.1",
        "keras==2.14.0",
        "google-cloud-aiplatform==1.43.0",
        "google-cloud-storage==2.15.0",
        # firestore is not present in the image, but we match the cloud-storage version.
        "google-cloud-firestore==2.15.0",
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
