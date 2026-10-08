#!/usr/bin/env bash
# Run the cloud_functions checks locally, from the repository root.
set -euo pipefail
flake8 usl_pipeline/cloud_functions --show-source --statistics
black usl_pipeline/cloud_functions --check
pytest usl_pipeline/cloud_functions
mypy usl_pipeline/cloud_functions
