import io
import json

import pytest

from usl_h3_tools import util_io

_FEATURES = [
    {"type": "Feature", "properties": {"admin_level": 109}, "geometry": None},
    {
        "type": "Feature",
        "properties": {"admin_level": 112, "name": "a b"},
        "geometry": {"type": "Point", "coordinates": [-84.4, 33.7]},
    },
]


@pytest.mark.parametrize("indent", [None, 2], ids=["single_line", "multi_line"])
def test_to_geojson_seq_writes_one_feature_per_line(indent):
    collection = {"type": "FeatureCollection", "features": _FEATURES}
    src = io.StringIO(json.dumps(collection, indent=indent))
    dst = io.StringIO()

    count = util_io.to_geojson_seq(src, dst, chunk_size=16)

    assert count == 2
    assert dst.getvalue().endswith("\n")
    assert [json.loads(line) for line in dst.getvalue().splitlines()] == _FEATURES
