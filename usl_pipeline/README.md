Getting Started
===============

To install the package and its requirements for local development:
- pip install -r requirements.txt
- pip install -e usl_lib[dev] -e map_uploader[dev]

To run the tests:
- pytest

To run the linter:
- flake8 .

To run the code auto-formatter:
- black .

Package Layout
==============
- **map_uploader** contains a script for chunking and uploading geographies to
Google Cloud Storage.
- **cloud_functions** contains source code for Google Cloud Functions used in
the data pipeline.
- **usl_lib** contains library code common to both of the above.

Dependencies
============
`map_uploader` and `cloud_functions` each define their own `requirements.txt`
files.
The `gen_requirements_txt.sh` script creates a `requirements.txt` representing
the union of the two package's dependencies.
This resulting `requirements.txt` file must then be checked into the repository.

This `requirements.txt` file is used in GitHub's automated testing to ensure the
common `usl_lib` code works with both sets of requirements.
It also forces `map_uploader` and `cloud_functions` to use the same version of
libraries they both depend upon.
The GitHub build will fail with a mesage about conflicting dependencies if the
two require inconsistent versions of the same library.
