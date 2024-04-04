#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="map_uploader",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "upload_map = map_uploader.main:main",
        ],
    },
    extras_require={
        "dev": [
            "black~=24.0",
            "flake8~=7.0",
            "flake8-docstrings~=1.7",
            "pytest~=8.0",
        ]
    },
)
