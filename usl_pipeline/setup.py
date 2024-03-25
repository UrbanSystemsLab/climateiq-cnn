#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name="usl_pipeline",
    packages=find_packages(),
    install_requires=['rasterio'],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "pytest",
        ]
    },
)
