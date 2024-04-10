#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="flood_ml",
    packages=find_packages(),
    extras_require={
        "dev": [
            "black~=24.0",
            "flake8~=7.0",
            "flake8-docstrings~=1.7",
            "pytest~=8.0",
            "mypy~=1.9",
        ]
    },
)
