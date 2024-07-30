import datetime
from unittest import mock

from google.cloud import firestore
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


def test_study_area_delete_all_chunks():
    mock_db = mock.MagicMock()
    mock_doc_ref = mock.MagicMock()
    mock_db.collection().document().collection().list_documents.return_value = [
        mock_doc_ref,
        mock_doc_ref,
        mock_doc_ref,
    ]

    metastore.StudyArea.delete_all_chunks(mock_db, "TestStudyArea")
    mock_db.assert_has_calls(
        [
            mock.call.collection("study_areas"),
            mock.call.collection().document("TestStudyArea"),
            mock.call.collection().document().collection("chunks"),
            mock.call.collection()
            .document()
            .collection()
            .list_documents(page_size=None),
        ]
    )
    assert mock_doc_ref.delete.call_count == 3


def test_study_area_chunk_get_if_exists():
    mock_db = mock.MagicMock()
    chunk_doc_mock = mock_db.collection().document().collection().document().get()
    chunk_doc_mock.exists = True
    chunk_doc_mock.to_dict.return_value = {
        "raw_path": "a/b",
        "needs_scaling": True,
    }

    chunk = metastore.StudyAreaChunk.get_if_exists(mock_db, "TestStudyArea", "chunk_1")
    assert chunk == metastore.StudyAreaChunk(
        id_="chunk_1",
        raw_path="a/b",
        feature_matrix_path=None,
        needs_scaling=True,
    )
    mock_db.assert_has_calls(
        [
            mock.call.collection("study_areas"),
            mock.call.collection().document("TestStudyArea"),
            mock.call.collection().document().collection("chunks"),
            mock.call.collection().document().collection().document("chunk_1"),
            mock.call.collection().document().collection().document().get(),
        ]
    )


def test_study_area_chunk_update_scaling_done():
    mock_db = mock.MagicMock()
    metastore.StudyAreaChunk.update_scaling_done(
        mock_db, "TestStudyArea", "chunk_1", "c/d"
    )
    mock_db.assert_has_calls(
        [
            mock.call.collection("study_areas"),
            mock.call.collection().document("TestStudyArea"),
            mock.call.collection().document().collection("chunks"),
            mock.call.collection().document().collection().document("chunk_1"),
            mock.call.collection()
            .document()
            .collection()
            .document()
            .update(
                {
                    "state": metastore.StudyAreaChunkState.FEATURE_MATRIX_READY,
                    "needs_scaling": False,
                    "feature_matrix_path": "c/d",
                    "error": firestore.DELETE_FIELD,
                },
            ),
        ]
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


def test_heat_scenario_config_set():
    mock_db = mock.MagicMock()
    metastore.HeatScenarioConfig(
        name="b/c",
        parent_config_name="parent",
        gcs_uri="a/b/c",
        simulation_year=2012,
        simulation_months="JJA",
        percentile=99,
    ).set(mock_db)
    mock_db.assert_has_calls(
        [
            mock.call.collection("wrf_heat_configs"),
            mock.call.collection().document("b%2Fc"),
            mock.call.collection()
            .document()
            .set(
                {
                    "parent_config_name": "parent",
                    "gcs_uri": "a/b/c",
                    "simulation_year": 2012,
                    "simulation_months": "JJA",
                    "percentile": 99,
                },
            ),
        ]
    )


def test_heat_scenario_config_delete():
    mock_db = mock.MagicMock()
    metastore.HeatScenarioConfig.delete(mock_db, "a/b/c")
    mock_db.assert_has_calls(
        [
            mock.call.collection("wrf_heat_configs"),
            mock.call.collection().document("a%2Fb%2Fc"),
            mock.call.collection().document().delete(),
        ]
    )


def test_simulation_label_chunk_dataset_split_produce_right_split() -> None:
    """Ensure dataset_split produces a 60/20/20 split."""
    study_area = metastore.StudyArea(
        name="name",
        col_count=2,
        row_count=5,
        x_ll_corner=1.0,
        y_ll_corner=2.0,
        cell_size=3.0,
        crs="over there",
        chunk_size=1,
        chunk_x_count=2,
        chunk_y_count=5,
    )

    # Calculate the number of chunks in each set.
    chunks_dataset_split = [
        metastore.SimulationLabelSpatialChunk.dataset_split(study_area, "config", 0, 0),
        metastore.SimulationLabelSpatialChunk.dataset_split(study_area, "config", 0, 1),
        metastore.SimulationLabelSpatialChunk.dataset_split(study_area, "config", 0, 2),
        metastore.SimulationLabelSpatialChunk.dataset_split(study_area, "config", 0, 3),
        metastore.SimulationLabelSpatialChunk.dataset_split(study_area, "config", 0, 4),
        metastore.SimulationLabelSpatialChunk.dataset_split(study_area, "config", 1, 0),
        metastore.SimulationLabelSpatialChunk.dataset_split(study_area, "config", 1, 1),
        metastore.SimulationLabelSpatialChunk.dataset_split(study_area, "config", 1, 2),
        metastore.SimulationLabelSpatialChunk.dataset_split(study_area, "config", 1, 3),
        metastore.SimulationLabelSpatialChunk.dataset_split(study_area, "config", 1, 4),
    ]
    # Chunks should be split 6 / 2 / 2 in train / val / test.
    assert chunks_dataset_split.count(metastore.DatasetSplit.TRAIN) == 6
    assert chunks_dataset_split.count(metastore.DatasetSplit.VAL) == 2
    assert chunks_dataset_split.count(metastore.DatasetSplit.TEST) == 2


def test_simulation_label_chunk_dataset_split_is_deterministic() -> None:
    """Ensure dataset_split produces deterministic outputs."""
    study_area = metastore.StudyArea(
        name="name",
        col_count=2,
        row_count=5,
        x_ll_corner=1.0,
        y_ll_corner=2.0,
        cell_size=3.0,
        crs="over there",
        chunk_size=1,
        chunk_x_count=2,
        chunk_y_count=5,
    )

    # Call the function with the same inputs several times.
    results = [
        metastore.SimulationLabelSpatialChunk.dataset_split(study_area, "config", 0, 0)
        for _ in range(10)
    ]
    # Ensure they're all the same.
    assert all(res == results[0] for res in results)


def test_simulation_label_chunk_dataset_split_produces_different_splits() -> None:
    """Ensure dataset_split produces distinct splits for distinct simulations."""
    study_area = metastore.StudyArea(
        name="name",
        col_count=2,
        row_count=3,
        x_ll_corner=1.0,
        y_ll_corner=2.0,
        cell_size=3.0,
        crs="over there",
        chunk_size=1,
        chunk_x_count=2,
        chunk_y_count=5,
    )

    chunks_dataset_split_for_config_1 = [
        metastore.SimulationLabelSpatialChunk.dataset_split(study_area, "c1", 0, 0),
        metastore.SimulationLabelSpatialChunk.dataset_split(study_area, "c1", 0, 1),
        metastore.SimulationLabelSpatialChunk.dataset_split(study_area, "c1", 0, 2),
        metastore.SimulationLabelSpatialChunk.dataset_split(study_area, "c1", 1, 0),
        metastore.SimulationLabelSpatialChunk.dataset_split(study_area, "c1", 1, 1),
        metastore.SimulationLabelSpatialChunk.dataset_split(study_area, "c1", 1, 2),
    ]

    chunks_dataset_split_for_config_2 = [
        metastore.SimulationLabelSpatialChunk.dataset_split(study_area, "c2", 0, 0),
        metastore.SimulationLabelSpatialChunk.dataset_split(study_area, "c2", 0, 1),
        metastore.SimulationLabelSpatialChunk.dataset_split(study_area, "c2", 0, 2),
        metastore.SimulationLabelSpatialChunk.dataset_split(study_area, "c2", 1, 0),
        metastore.SimulationLabelSpatialChunk.dataset_split(study_area, "c2", 1, 1),
        metastore.SimulationLabelSpatialChunk.dataset_split(study_area, "c2", 1, 2),
    ]

    # Different simulation config names could potentially result in
    # the same set of chunks in the test set, but our use of
    # seeds ensures we will get the same sets for the same
    # configuration names, making this test reproducible.
    assert chunks_dataset_split_for_config_1 != chunks_dataset_split_for_config_2


def test_simulation_label_temporal_chunk_set():
    mock_db = mock.MagicMock()
    metastore.SimulationLabelTemporalChunk(
        gcs_uri="gs://bucket/study_area/config_group/Heat_Data_2012.txt",
        time=datetime.datetime(2012, 2, 2, 18, 0, 0),
    ).set(mock_db, "study_area", "config_group/Heat_Data_2012.txt")
    mock_db.assert_has_calls(
        [
            mock.call.collection("simulations"),
            mock.call.collection().document(
                "study_area-config_group%2FHeat_Data_2012.txt"
            ),
            mock.call.collection().document().collection("label_chunks"),
            mock.call.collection()
            .document()
            .collection()
            .document("2012-02-02 18:00:00"),
            mock.call.collection()
            .document()
            .collection()
            .document()
            .set(
                {
                    "gcs_uri": "gs://bucket/study_area/config_group/Heat_Data_2012.txt",
                    "time": datetime.datetime(2012, 2, 2, 18, 0, 0),
                },
            ),
        ]
    )
