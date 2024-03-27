#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name="usl_pipeline",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "upload_map = usl_pipeline.upload_map:main",
        ],
    },
    extras_require={
        "dev": [
            "black",
            "flake8",
            "pytest",
        ]
    },
)
