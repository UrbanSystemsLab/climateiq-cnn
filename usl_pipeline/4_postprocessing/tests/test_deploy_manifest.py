"""Guard that the manifest matches the functions defined in this deployable."""

import json
import pathlib
import subprocess
import sys

import main

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MANIFEST = _ROOT / ".github" / "deploy" / "functions.json"

_CHECK_STAGED_SOURCE = """
import pathlib
import sys

import main
import postprocessing
import usl_lib

staged = pathlib.Path.cwd().resolve()
for module in (main, postprocessing, usl_lib):
    assert pathlib.Path(module.__file__).resolve().is_relative_to(staged), module
for entry_point in sys.argv[1:]:
    assert callable(getattr(main, entry_point, None)), entry_point
"""


def _group() -> dict:
    return json.loads(_MANIFEST.read_text())["groups"]["postprocessing"]


def test_manifest_entry_points_exist_in_main():
    functions = _group()["functions"]
    assert functions, "manifest lists no postprocessing functions"
    for name, spec in functions.items():
        entry_point = spec["entry_point"]
        assert callable(
            getattr(main, entry_point, None)
        ), f"{name}: entry point {entry_point!r} is not defined in main.py"


def test_staged_source_resolves_every_entry_point(tmp_path):
    group = _group()
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
