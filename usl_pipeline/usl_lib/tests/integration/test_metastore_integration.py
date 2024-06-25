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

    chunk_1 = metastore.SimulationLabelChunk(
        gcs_uri="gs://sim-chunks/study-area/config/name.txt/0_0.npy",
        x_index=0,
        y_index=0,
    )
    chunk_1.set(firestore_db, "study-area", "config/name.txt")

    chunk_2 = metastore.SimulationLabelChunk(
        gcs_uri="gs://sim-chunks/study-area/config/name.txt/0_1.npy",
        x_index=1,
        y_index=0,
    )
    chunk_2.set(firestore_db, "study-area", "config/name.txt")

    chunks = list(
        metastore.SimulationLabelChunk.list_chunks(
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


def test_update_chunk_info(firestore_db):
    metastore.StudyArea(
        name="study-area",
        col_count=10,
        row_count=4,
        x_ll_corner=3,
        y_ll_corner=4,
        cell_size=5,
        crs="crs",
    ).create(firestore_db)

    metastore.StudyArea.update_chunk_info(firestore_db, "study-area", 2, seed=2)
    assert metastore.StudyArea.get(firestore_db, "study-area") == metastore.StudyArea(
        name="study-area",
        col_count=10,
        row_count=4,
        x_ll_corner=3,
        y_ll_corner=4,
        cell_size=5,
        crs="crs",
        chunk_x_count=5,  # 10 / 2
        chunk_y_count=2,  # 4 / 2
        chunks_size=2,
        #  8 / 10 chunks for training.
        train_indices=[
            {"x_index": 0, "y_index": 0},
            {"x_index": 0, "y_index": 1},
            {"x_index": 1, "y_index": 0},
            {"x_index": 1, "y_index": 1},
            {"x_index": 2, "y_index": 0},
            {"x_index": 3, "y_index": 0},
            {"x_index": 3, "y_index": 1},
            {"x_index": 4, "y_index": 0},
        ],
        #  2 / 10 chunks for test.
        test_indices=[{"x_index": 2, "y_index": 1}, {"x_index": 4, "y_index": 1}],
    )
