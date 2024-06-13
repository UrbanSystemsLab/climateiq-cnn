from unittest import mock

import rasterio

from usl_lib.shared import geo_data
from usl_lib.storage import metastore


def test_study_area_create():
    mock_db = mock.MagicMock()
    study_area = metastore.StudyArea(
        name="name",
        col_count=2,
        row_count=3,
        x_ll_corner=1.0,
        y_ll_corner=2.0,
        cell_size=3.0,
        crs="over there",
    )
    study_area.create(mock_db)
    mock_db.assert_has_calls(
        [
            mock.call.collection("study_areas"),
            mock.call.collection().document("name"),
            mock.call.collection()
            .document()
            .create(
                {
                    "col_count": 2,
                    "row_count": 3,
                    "x_ll_corner": 1.0,
                    "y_ll_corner": 2.0,
                    "cell_size": 3.0,
                    "crs": "over there",
                }
            ),
        ]
    )


def test_study_area_set():
    mock_db = mock.MagicMock()
    study_area = metastore.StudyArea(
        name="name",
        col_count=2,
        row_count=3,
        x_ll_corner=1.0,
        y_ll_corner=2.0,
        cell_size=3.0,
        crs=None,
    )
    study_area.set(mock_db)
    mock_db.assert_has_calls(
        [
            mock.call.collection("study_areas"),
            mock.call.collection().document("name"),
            mock.call.collection()
            .document()
            .set(
                {
                    "col_count": 2,
                    "row_count": 3,
                    "x_ll_corner": 1.0,
                    "y_ll_corner": 2.0,
                    "cell_size": 3.0,
                }
            ),
        ]
    )


def test_study_area_chunk_merge():
    mock_db = mock.MagicMock()
    chunk = metastore.StudyAreaChunk(
        id_="id",
        archive_path="archive",
        feature_matrix_path="matrix",
    )
    chunk.merge(mock_db, "study_area_name")
    mock_db.assert_has_calls(
        [
            mock.call.collection("study_areas"),
            mock.call.collection().document("study_area_name"),
            mock.call.collection().document().collection("chunks"),
            mock.call.collection().document().collection().document("id"),
            mock.call.collection()
            .document()
            .collection()
            .document()
            .set(
                {
                    "archive_path": "archive",
                    "feature_matrix_path": "matrix",
                },
                merge=True,
            ),
        ]
    )


def test_study_area_as_header():
    """Ensures the StudyArea as_header method works."""
    study_area = metastore.StudyArea(
        name="name",
        col_count=2,
        row_count=3,
        x_ll_corner=1.0,
        y_ll_corner=2.0,
        cell_size=3.0,
        crs="EPSG:3005",
    )
    assert study_area.as_header() == geo_data.ElevationHeader(
        col_count=2,
        row_count=3,
        x_ll_corner=1.0,
        y_ll_corner=2.0,
        cell_size=3.0,
        nodata_value=-9999.0,
        crs=rasterio.CRS.from_string("EPSG:3005"),
    )


def test_flood_scenario_config_set():
    mock_db = mock.MagicMock()
    metastore.FloodScenarioConfig(
        name="b/c",
        gcs_uri="a/b/c",
        as_vector_gcs_uri="d/e/f",
        parent_config_name="parent",
        rainfall_duration=5,
    ).set(mock_db)
    mock_db.assert_has_calls(
        [
            mock.call.collection("city_cat_rainfall_configs"),
            mock.call.collection().document("b%2Fc"),
            mock.call.collection()
            .document()
            .set(
                {
                    "parent_config_name": "parent",
                    "gcs_uri": "a/b/c",
                    "as_vector_gcs_uri": "d/e/f",
                    "rainfall_duration": 5,
                },
            ),
        ]
    )


def test_flood_scenario_config_delete():
    mock_db = mock.MagicMock()
    metastore.FloodScenarioConfig.delete(mock_db, "a/b/c")
    mock_db.assert_has_calls(
        [
            mock.call.collection("city_cat_rainfall_configs"),
            mock.call.collection().document("a%2Fb%2Fc"),
            mock.call.collection().document().delete(),
        ]
    )
