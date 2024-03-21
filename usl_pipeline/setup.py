#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name="usl_pipeline",
    packages=find_packages(),
    extras_require={
        "dev": [
            "black",
            "flake8",
            "pytest",
        ]
    },
)
