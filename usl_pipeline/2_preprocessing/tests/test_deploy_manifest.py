"""Guard that the CI config matches the repo and the functions defined in main.py."""

import json
import pathlib
import subprocess
import sys

import yaml

import main

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MANIFEST = _ROOT / ".github" / "deploy" / "functions.json"
_CI_MATRIX = _ROOT / ".github" / "ci-matrix.json"
_CI_WORKFLOW = _ROOT / ".github" / "workflows" / "ci.yml"

_CHECK_STAGED_SOURCE = """
import pathlib
import sys

import main
import preprocessing
import usl_lib

staged = pathlib.Path.cwd().resolve()
for module in (main, preprocessing, usl_lib):
    assert pathlib.Path(module.__file__).resolve().is_relative_to(staged), module
for entry_point in sys.argv[1:]:
    assert callable(getattr(main, entry_point, None)), entry_point
"""


def _groups() -> dict[str, dict]:
    return json.loads(_MANIFEST.read_text())["groups"]


def test_manifest_entry_points_exist_in_main():
    functions = _groups()["preprocessing"]["functions"]
    assert functions, "manifest lists no preprocessing functions"
    for name, spec in functions.items():
        entry_point = spec["entry_point"]
        assert callable(
            getattr(main, entry_point, None)
        ), f"{name}: entry point {entry_point!r} is not defined in main.py"


def test_manifest_function_names_are_cloud_function_names():
    for group in _groups().values():
        for name in group["functions"]:
            assert name == name.lower() and "_" not in name, (
                f"{name!r} should be the GCP function name (lowercase, hyphenated), "
                "not the Python entry point"
            )


def test_staged_source_resolves_every_entry_point(tmp_path):
    group = _groups()["preprocessing"]
    entry_points = [spec["entry_point"] for spec in group["functions"].values()]

    staged = tmp_path / "source"
    subprocess.run(
        ["bash", str(_ROOT / group["stage"]), str(staged)],
        check=True,
        capture_output=True,
    )

    result = subprocess.run(
        [sys.executable, "-c", _CHECK_STAGED_SOURCE, *entry_points],
        cwd=staged,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _ci_path_filters() -> dict[str, list[str]]:
    workflow = yaml.safe_load(_CI_WORKFLOW.read_text())
    steps = workflow["jobs"]["changes"]["steps"]
    filter_step = next(step for step in steps if step.get("id") == "filter")
    return yaml.safe_load(filter_step["with"]["filters"])


def test_ci_matrix_services_have_path_filters():
    filters = _ci_path_filters()
    for entry in json.loads(_CI_MATRIX.read_text()):
        service = entry["service"]
        assert service in filters, f"{service} has no path filter, so it never runs"
        assert (_ROOT / entry["path"]).is_dir(), f"{service}: {entry['path']} not found"


def test_ci_path_filters_match_existing_paths():
    for name, patterns in _ci_path_filters().items():
        for pattern in patterns:
            path = _ROOT / pattern.removesuffix("/**")
            assert path.exists(), f"{name}: {pattern} matches nothing"
