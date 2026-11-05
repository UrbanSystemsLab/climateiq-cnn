from usl_h3_tools import pmtiles_build


def test_start_runs_the_build_in_a_new_process(tmp_path):
    out_path = tmp_path / "out.pmtiles"
    layer = pmtiles_build.LayerSource("layer", 0, 1, str(tmp_path / "missing.geojson"))

    process = pmtiles_build.start(str(out_path), "name", "description", [layer])

    assert process.wait(timeout=60) == 0
    assert not out_path.exists()
