import io
import json
from unittest import mock

from usl_h3_tools import publish

_FEATURES = [
    {"type": "Feature", "properties": {"admin_level": 109}, "geometry": None},
    {
        "type": "Feature",
        "properties": {"admin_level": 112, "name": "a b"},
        "geometry": {"type": "Point", "coordinates": [-84.4, 33.7]},
    },
]
_TILESET = publish.tileset_for_object(
    "bucket", "USA_GA_Atlanta/h3_USA_GA_Atlanta.geojson"
)


def test_mapbox_upload_source_streams_staged_file():
    content = b"".join(json.dumps(f).encode() + b"\n" for f in _FEATURES)
    blob = mock.MagicMock(size=len(content), generation=7)
    blob.open.return_value = io.BytesIO(content)
    storage_client = mock.MagicMock()
    storage_client.bucket.return_value.blob.return_value = blob
    session = mock.MagicMock()
    sent = {}

    def request(method, url, **kwargs):
        body = kwargs["data"]
        sent.update(method=method, url=url, length=len(body), body=body.read())
        sent["params"] = kwargs["params"]
        return mock.MagicMock(ok=True)

    session.request.side_effect = request

    publish.Mapbox("climateiq", "sk.token", storage_client, session).upload_source(
        _TILESET
    )

    storage_client.bucket.return_value.blob.assert_called_once_with(
        "tile_sources/climateiq-h3-USA_GA_Atlanta/source.geojsonl"
    )
    blob.open.assert_called_once_with("rb", if_generation_match=7)
    assert sent["method"] == "PUT"
    assert sent["url"] == (
        "https://api.mapbox.com/tilesets/v1/sources/climateiq/"
        "climateiq-h3-USA_GA_Atlanta"
    )
    assert sent["params"] == {"access_token": "sk.token"}
    assert content in sent["body"]
    assert sent["length"] == len(sent["body"])


def test_mapbox_publish_creates_tileset_and_starts_job():
    session = mock.MagicMock()
    session.request.side_effect = [
        mock.MagicMock(ok=True, **{"json.return_value": {"valid": True}}),
        mock.MagicMock(ok=False, status_code=404),
        mock.MagicMock(ok=True),
        mock.MagicMock(ok=True, **{"json.return_value": {"jobId": "job-1"}}),
    ]

    result = publish.Mapbox("climateiq", "sk.token", mock.MagicMock(), session).publish(
        _TILESET
    )

    tileset_url = (
        "https://api.mapbox.com/tilesets/v1/climateiq.climateiq-h3-USA_GA_Atlanta"
    )
    assert [c.args for c in session.request.call_args_list] == [
        ("PUT", "https://api.mapbox.com/tilesets/v1/validateRecipe"),
        ("GET", f"{tileset_url}/recipe"),
        ("POST", tileset_url),
        ("POST", f"{tileset_url}/publish"),
    ]
    assert result.tileset_id == "climateiq.climateiq-h3-USA_GA_Atlanta"
    assert result.job_id == "job-1"
