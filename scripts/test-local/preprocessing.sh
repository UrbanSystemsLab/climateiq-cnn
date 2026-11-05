#!/usr/bin/env bash
# Run the pre-processing checks locally, from the repository root.
set -euo pipefail
flake8 usl_pipeline/2_preprocessing --show-source --statistics
black usl_pipeline/2_preprocessing --check
pytest usl_pipeline/2_preprocessing
MYPYPATH=usl_pipeline/usl_lib mypy usl_pipeline/2_preprocessing
