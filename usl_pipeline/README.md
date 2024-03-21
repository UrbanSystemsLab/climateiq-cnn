Getting Started
===============

To install the package and its requirements for local development:
- pip install -r dev-requirements.txt
- pip install -e .[dev]

To run the tests:
- py.test tests

To run the linter:
- flake8 usl_pipeline tests

To run the code auto-formatter:
- black usl_pipeline tests
