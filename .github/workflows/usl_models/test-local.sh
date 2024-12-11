# .github/workflows/usl_models/test-local.sh
# if it fails, we should give permission: chmod +x /home/elhajjas/climateiq-cnn/.github/workflows/usl_models/test-local.sh
# flake8 usl_models --show-source --statistics
# black usl_models --check
# pytest usl_models -k "not integration"
# mypy usl_models