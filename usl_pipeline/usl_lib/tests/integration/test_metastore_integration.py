import os

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
