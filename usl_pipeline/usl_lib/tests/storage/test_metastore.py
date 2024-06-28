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


def test_study_area_update_chunk_info():
    mock_db = mock.MagicMock()
    mock_db.collection().document().get().to_dict.return_value = {
        "col_count": 10001,
        "row_count": 20999,
        "x_ll_corner": 0.0,
        "y_ll_corner": 0.0,
        "cell_size": 1.0,
    }

    metastore.StudyArea.update_chunk_info(mock_db, "TestArea", 1000)

    mock_db.collection().document().update.assert_called_once_with(
        {"chunk_size": 1000, "chunk_x_count": 11, "chunk_y_count": 21}
    )


def test_study_area_update_chunk_info_no_need():
    mock_db = mock.MagicMock()
    mock_db.collection().document().get().to_dict.return_value = {
        "col_count": 10500,
        "row_count": 20500,
        "x_ll_corner": 0.0,
        "y_ll_corner": 0.0,
        "cell_size": 1.0,
        "chunk_size": 1000,
    }

    metastore.StudyArea.update_chunk_info(mock_db, "TestArea", 1000)

    mock_db.transaction().update.assert_not_called()


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
