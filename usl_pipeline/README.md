Getting Started
===============

To install the package and its requirements for local development:
- System dependencies:
  - GDAL version 3.6.4 or later (see https://gdal.org/download.html for details)
- pip install -r requirements.txt
- pip install -e usl_lib[dev] -e study_area_uploader[dev]

To run the unit tests:
- pytest -k 'not integration'

To run the linter:
- flake8 .

To run the type checker:
- mypy --config-file ../mypy.ini .

To run the code auto-formatter:
- black .

Package Layout
==============
- **study_area_uploader** contains a script for chunking and uploading geographies to
Google Cloud Storage.
- **cloud_functions** contains source code for Google Cloud Functions used in
the data pipeline.
- **usl_lib** contains library code common to both of the above.

Integration Tests
=================
The instructions above run unit tests and skip integration tests with
`pytest -k 'not integration'`.
Running the integration tests requires the Firestore emulator.
Follow
[these instructions](https://cloud.google.com/firestore/docs/emulator)
to install and run the emulator.

Start the emulator and be sure to set the `FIRESTORE_EMULATOR_HOST`
environmental variable as described in the documentation.  Once you've
done so, you may now run the full test suite, including integration
tests, with `pytest` (without the `-k 'not integration'` bit.)
