#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="usl_pipeline",
    packages=find_packages(),
    install_requires=[
        "functions-framework",
        "google-cloud-storage",
        "numpy",
        "rasterio",
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "pytest",
        ]
    },
)
