#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="usl_lib",
    packages=find_packages(),
    install_requires=[
        "rasterio",
    ],
    extras_require={
        "dev": [
            "black~=24.0",
            "flake8~=7.0",
            "flake8-docstrings~=1.7",
            "pytest~=8.0",
        ]
    },
)
