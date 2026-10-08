"""Guard that the CI deploy manifest matches the functions defined in main.py."""

import json
import pathlib

import main

_MANIFEST = (
    pathlib.Path(__file__).resolve().parents[2]
    / ".github"
    / "deploy"
    / "functions.json"
)


def _pipeline_functions() -> dict[str, dict[str, str]]:
    manifest = json.loads(_MANIFEST.read_text())
    return manifest["groups"]["pipeline"]["functions"]


def test_manifest_entry_points_exist_in_main():
    functions = _pipeline_functions()
    assert functions, "manifest lists no pipeline functions"
    for name, spec in functions.items():
        entry_point = spec["entry_point"]
        assert callable(
            getattr(main, entry_point, None)
        ), f"{name}: entry point {entry_point!r} is not defined in main.py"


def test_manifest_function_names_are_cloud_function_names():
    for name in _pipeline_functions():
        assert name == name.lower() and "_" not in name, (
            f"{name!r} should be the GCP function name (lowercase, hyphenated), "
            "not the Python entry point"
        )
