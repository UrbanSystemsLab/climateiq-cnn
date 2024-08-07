import datetime
import os
import unittest

import pytest
import requests
from google.cloud import firestore

from usl_lib.storage import metastore

_GCP_PROJECT_ID = "integation-test"


@pytest.fixture
def firestore_db():
    """Creates a firestore DB client and clears the test database."""
    try:
        emulator_host = os.environ["FIRESTORE_EMULATOR_HOST"]
    except KeyError:
        raise RuntimeError(
            "To run these tests, you must start a firestore emulator and set the "
            "FIRESTORE_EMULATOR_HOST environmental variable. "
            "See the README file for details."
        )

    requests.delete(
        f"http://{emulator_host}/emulator/v1/projects/{_GCP_PROJECT_ID}/"
        "databases/(default)/documents"
    )

    return firestore.Client(project=_GCP_PROJECT_ID)


def test_create_get_study_area(firestore_db):
    """Ensures a study area can be created and retrieved."""
    study_area = metastore.StudyArea(
        name="test",
        col_count=1,
        row_count=2,
        x_ll_corner=3,
        y_ll_corner=4,
        cell_size=5,
        crs="crs",
    )
    study_area.create(firestore_db)
    assert study_area == metastore.StudyArea.get(firestore_db, "test")
    assert metastore.StudyArea.get_ref(firestore_db, "test").get().exists


def test_update_min_max_elevation(firestore_db):
    """Ensures a min & max elevation can be updated."""
    metastore.StudyArea(
        name="test",
        col_count=1,
        row_count=2,
        x_ll_corner=3,
        y_ll_corner=4,
        cell_size=5,
        crs="crs",
    ).create(firestore_db)

    # Ensure we can set the initial min & max.
    metastore.StudyArea.update_min_max_elevation(firestore_db, "test", 2, 20)
    study_area = metastore.StudyArea.get(firestore_db, "test")
    assert study_area.elevation_min == 2
    assert study_area.elevation_max == 20

    # Ensure we only update the max.
    metastore.StudyArea.update_min_max_elevation(firestore_db, "test", 10, 30)
    study_area = metastore.StudyArea.get(firestore_db, "test")
    assert study_area.elevation_min == 2
    assert study_area.elevation_max == 30

    # Ensure we only update the min.
    metastore.StudyArea.update_min_max_elevation(firestore_db, "test", 1, 10)
    study_area = metastore.StudyArea.get(firestore_db, "test")
    assert study_area.elevation_min == 1
    assert study_area.elevation_max == 30


def test_flood_scenario_config_get_set_delete(firestore_db):
    """Ensures we can create, retrieve and delete flood configs."""
    flood = metastore.FloodScenarioConfig(
        name="config/name.txt",
        gcs_uri="gs://config-bucket/config/name.txt",
        as_vector_gcs_uri="gs://feature-bucket/config/name.npy",
        parent_config_name="config",
        rainfall_duration=15,
    )
    flood.set(firestore_db)

    assert flood == metastore.FloodScenarioConfig.get(firestore_db, "config/name.txt")
    assert (
        metastore.FloodScenarioConfig.get_ref(firestore_db, "config/name.txt")
        .get()
        .exists
    )

    metastore.FloodScenarioConfig.delete(firestore_db, "config/name.txt")

    assert not (
        metastore.FloodScenarioConfig.get_ref(firestore_db, "config/name.txt")
        .get()
        .exists
    )
    with pytest.raises(ValueError):
        metastore.FloodScenarioConfig.get(firestore_db, "config/name.txt")


def test_heat_scenario_config_get_set_delete(firestore_db):
    """Ensures we can create, retrieve and delete heat configs."""
    heat = metastore.HeatScenarioConfig(
        name="config-group/Heat_Data_2012.txt",
        parent_config_name="config-group",
        gcs_uri="gs://config-bucket/config-group/Heat_Data_2012.txt",
        simulation_year=2012,
        simulation_months="JJA",
        percentile=99,
    )
    heat.set(firestore_db)

    assert heat == metastore.HeatScenarioConfig.get(
        firestore_db, "config-group/Heat_Data_2012.txt"
    )
    assert (
        metastore.HeatScenarioConfig.get_ref(
            firestore_db, "config-group/Heat_Data_2012.txt"
        )
        .get()
        .exists
    )

    metastore.HeatScenarioConfig.delete(firestore_db, "config-group/Heat_Data_2012.txt")

    assert not (
        metastore.HeatScenarioConfig.get_ref(
            firestore_db, "config-group/Heat_Data_2012.txt"
        )
        .get()
        .exists
    )
    with pytest.raises(ValueError):
        metastore.HeatScenarioConfig.get(
            firestore_db, "config-group/Heat_Data_2012.txt"
        )


def test_simulation_get_set_label_chunks(firestore_db):
    """Ensures we can save and retrieve simulations and their label chunks."""
    metastore.StudyArea(
        name="study-area",
        col_count=1,
        row_count=2,
        x_ll_corner=3,
        y_ll_corner=4,
        cell_size=5,
        crs="crs",
    ).create(firestore_db)

    metastore.FloodScenarioConfig(
        name="config/name.txt",
        gcs_uri="gs://config-bucket/config/name.txt",
        as_vector_gcs_uri="gs://feature-bucket/config/name.npy",
        parent_config_name="config",
        rainfall_duration=15,
    ).set(firestore_db)

    simulation = metastore.Simulation(
        gcs_prefix_uri="gs://sim-bucket/study-area/config/name.txt/",
        simulation_type=metastore.SimulationType.CITY_CAT,
        study_area=metastore.StudyArea.get_ref(firestore_db, "study-area"),
        configuration=metastore.FloodScenarioConfig.get_ref(
            firestore_db, "config/name.txt"
        ),
    )
    simulation.set(firestore_db)

    assert (
        metastore.Simulation.get(firestore_db, "study-area", "config/name.txt")
        == simulation
    )

    chunk_1 = metastore.SimulationLabelSpatialChunk(
        gcs_uri="gs://sim-chunks/study-area/config/name.txt/0_0.npy",
        x_index=0,
        y_index=0,
        dataset=metastore.DatasetSplit.TEST,
    )
    chunk_1.set(firestore_db, "study-area", "config/name.txt")

    chunk_2 = metastore.SimulationLabelSpatialChunk(
        gcs_uri="gs://sim-chunks/study-area/config/name.txt/0_1.npy",
        x_index=1,
        y_index=0,
        dataset=metastore.DatasetSplit.TRAIN,
    )
    chunk_2.set(firestore_db, "study-area", "config/name.txt")

    chunks = list(
        metastore.SimulationLabelSpatialChunk.list_chunks(
            firestore_db, "study-area", "config/name.txt"
        )
    )
    unittest.TestCase().assertCountEqual(chunks, [chunk_1, chunk_2])


def test_simulation_set_fails_with_bad_study_area(firestore_db):
    """Ensures we raise an error when a simulation is given a bad study area ref."""
    metastore.FloodScenarioConfig(
        name="config/name.txt",
        gcs_uri="gs://config-bucket/config/name.txt",
        as_vector_gcs_uri="gs://feature-bucket/config/name.npy",
        parent_config_name="config",
        rainfall_duration=15,
    ).set(firestore_db)

    simulation = metastore.Simulation(
        gcs_prefix_uri="gs://sim-bucket/study-area/config/name.txt/",
        simulation_type=metastore.SimulationType.CITY_CAT,
        study_area=metastore.StudyArea.get_ref(firestore_db, "missing-area"),
        configuration=metastore.FloodScenarioConfig.get_ref(
            firestore_db, "config/name.txt"
        ),
    )
    with pytest.raises(ValueError) as excinfo:
        simulation.set(firestore_db)
    assert "No such study area exists" in str(excinfo.value)


def test_simulation_set_fails_with_bad_simulation_config(firestore_db):
    """Ensures we raise an error when a simulation is given a bad config ref."""
    metastore.StudyArea(
        name="study-area",
        col_count=1,
        row_count=2,
        x_ll_corner=3,
        y_ll_corner=4,
        cell_size=5,
        crs="crs",
    ).create(firestore_db)

    simulation = metastore.Simulation(
        gcs_prefix_uri="gs://sim-bucket/study-area/config/name.txt/",
        simulation_type=metastore.SimulationType.CITY_CAT,
        study_area=metastore.StudyArea.get_ref(firestore_db, "study-area"),
        configuration=metastore.FloodScenarioConfig.get_ref(
            firestore_db, "config/name.txt"
        ),
    )

    with pytest.raises(ValueError) as excinfo:
        simulation.set(firestore_db)
    assert "No such configuration exists" in str(excinfo.value)


def test_get_simulation_bad_name_raises_error(firestore_db):
    """Ensure an error is raised when retrieving non-existent simulations."""
    with pytest.raises(ValueError):
        metastore.Simulation.get(
            firestore_db, "missing-study-area", "missing-config/name.txt"
        )


def test_create_get_study_area_spatial_chunk(firestore_db):
    """Ensures a study area spatial chunk can be created and retrieved."""
    study_area = metastore.StudyArea(
        name="study_area_name",
        col_count=1,
        row_count=2,
        x_ll_corner=3,
        y_ll_corner=4,
        cell_size=5,
        crs="crs",
    )
    study_area.create(firestore_db)

    chunk = metastore.StudyAreaSpatialChunk(
        id_="chunk_name",
        raw_path="gcs://raw_file.tar",
        feature_matrix_path="gcs://feautre_file.npy",
        x_index=1,
        y_index=2,
    )
    chunk.merge(firestore_db, "study_area_name")

    assert (
        metastore.StudyAreaSpatialChunk.get(
            firestore_db, "study_area_name", "chunk_name"
        )
        == chunk
    )


def test_spacial_chunk_update(firestore_db):
    """Ensures a study area spatial chunk can be updated."""
    study_area = metastore.StudyArea(
        name="study_area_name",
        col_count=1,
        row_count=2,
        x_ll_corner=3,
        y_ll_corner=4,
        cell_size=5,
        crs="crs",
    )
    study_area.create(firestore_db)

    chunk = metastore.StudyAreaSpatialChunk(
        id_="chunk_name",
        state=metastore.StudyAreaChunkState.FEATURE_MATRIX_PROCESSING,
        raw_path="gcs://raw_file.tar",
        needs_scaling=True,
        x_index=1,
        y_index=2,
        error="Old error",
    )
    chunk.merge(firestore_db, "study_area_name")

    metastore.StudyAreaSpatialChunk.update_scaling_done(
        firestore_db, "study_area_name", "chunk_name", "gcs://feature_file.npy"
    )

    assert metastore.StudyAreaSpatialChunk.get(
        firestore_db, "study_area_name", "chunk_name"
    ) == metastore.StudyAreaSpatialChunk(
        id_="chunk_name",
        state=metastore.StudyAreaChunkState.FEATURE_MATRIX_READY,
        raw_path="gcs://raw_file.tar",
        feature_matrix_path="gcs://feature_file.npy",
        needs_scaling=False,
        x_index=1,
        y_index=2,
    )


def test_delete_all_study_area_chunks(firestore_db):
    """Ensures all study area chunks can be deleted for a given study area."""
    study_area = metastore.StudyArea(
        name="study_area_name",
        col_count=1,
        row_count=2,
        x_ll_corner=3,
        y_ll_corner=4,
        cell_size=5,
        crs="crs",
    )
    study_area.create(firestore_db)

    for i in range(20):
        metastore.StudyAreaChunk(id_=f"chunk_{i}", raw_path="tar!").merge(
            firestore_db, "study_area_name"
        )

    for i in range(20):
        assert (
            metastore.StudyAreaChunk.get_if_exists(
                firestore_db, "study_area_name", f"chunk_{i}"
            )
            is not None
        )

    metastore.StudyArea.delete_all_chunks(firestore_db, study_area.name, page_size=10)

    for i in range(20):
        assert (
            metastore.StudyAreaChunk.get_if_exists(
                firestore_db, "study_area_name", f"chunk_{i}"
            )
            is None
        )


def test_create_get_study_area_temporal_chunk(firestore_db):
    """Ensures a study area spatial chunk can be created and retrieved."""
    study_area = metastore.StudyArea(
        name="study_area_name",
        col_count=1,
        row_count=2,
        x_ll_corner=3,
        y_ll_corner=4,
        cell_size=5,
        crs="crs",
    )
    study_area.create(firestore_db)

    chunk = metastore.StudyAreaTemporalChunk(
        id_="chunk_name",
        raw_path="gcs://raw_file.nc",
        feature_matrix_path="gcs://feautre_file.npy",
        time=datetime.datetime(2012, 12, 21, 0, 0, 0, tzinfo=datetime.timezone.utc),
    )
    chunk.merge(firestore_db, "study_area_name")

    assert (
        metastore.StudyAreaTemporalChunk.get(
            firestore_db, "study_area_name", "chunk_name"
        )
        == chunk
    )


def test_study_area_list_all_chunk_refs(firestore_db):
    """Ensures all study area chunks can be listed for a given study area."""
    study_area = metastore.StudyArea(
        name="study_area_name",
        col_count=1,
        row_count=2,
        x_ll_corner=3,
        y_ll_corner=4,
        cell_size=5,
        crs="crs",
    )
    study_area.create(firestore_db)

    for i in range(3):
        metastore.StudyAreaChunk(id_=f"chunk_{i}", raw_path=f"tar_{i}!").merge(
            firestore_db, "study_area_name"
        )

    ref_list = metastore.StudyArea.list_all_chunk_refs(firestore_db, "study_area_name")
    chunk_list = [metastore.StudyAreaChunk.from_ref(ref) for ref in ref_list]
    assert chunk_list == [
        metastore.StudyAreaChunk(id_="chunk_0", raw_path="tar_0!"),
        metastore.StudyAreaChunk(id_="chunk_1", raw_path="tar_1!"),
        metastore.StudyAreaChunk(id_="chunk_2", raw_path="tar_2!"),
    ]


def test_study_area_update_state(firestore_db):
    """Ensures all study area chunks can be listed for a given study area."""
    study_area = metastore.StudyArea(
        name="StudyArea1",
        col_count=1,
        row_count=2,
        x_ll_corner=3,
        y_ll_corner=4,
        cell_size=5,
        crs="crs",
    )
    study_area.create(firestore_db)

    metastore.StudyArea.update_state(
        firestore_db, "StudyArea1", metastore.StudyAreaState.INIT
    )

    assert metastore.StudyArea.get(firestore_db, "StudyArea1") == metastore.StudyArea(
        name="StudyArea1",
        col_count=1,
        row_count=2,
        x_ll_corner=3,
        y_ll_corner=4,
        cell_size=5,
        crs="crs",
        state=metastore.StudyAreaState.INIT,
    )
