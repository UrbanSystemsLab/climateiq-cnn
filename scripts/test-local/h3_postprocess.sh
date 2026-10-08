#!/usr/bin/env bash
# Run the h3_postprocess checks locally, from the repository root.
set -euo pipefail
flake8 usl_pipeline/h3_postprocess --show-source --statistics
black usl_pipeline/h3_postprocess --check
mypy usl_pipeline/h3_postprocess
