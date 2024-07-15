Getting Started
===============

To install the package and its requirements for local development:
- pip install -r requirements.txt
- pip install -e .[dev]

To run the unit tests:
- pytest -k 'not integration'

To run the linter:
- flake8 .

To run the type checker:
- mypy --config-file ../mypy.ini .

To run the code auto-formatter:
- black .

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
