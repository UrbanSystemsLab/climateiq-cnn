#!/usr/bin/env bash
# Run the usl_models checks locally, from the repository root.
set -euo pipefail
flake8 usl_models --show-source --statistics
black usl_models --check
pytest usl_models -k "not integration"
mypy usl_models
