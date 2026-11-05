#!/usr/bin/env bash
# Run the post-processing checks locally, from the repository root.
set -euo pipefail
flake8 usl_pipeline/4_postprocessing --show-source --statistics
black usl_pipeline/4_postprocessing --check
pytest usl_pipeline/4_postprocessing
MYPYPATH=usl_pipeline/usl_lib mypy usl_pipeline/4_postprocessing
