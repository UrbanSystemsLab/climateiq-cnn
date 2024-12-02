# .github/workflows/cloud_functions/test-local.sh
flake8 usl_pipeline/cloud_functions --show-source --statistics
black usl_pipeline/cloud_functions --check
pytest usl_pipeline/cloud_functions
mypy usl_pipeline/cloud_functions