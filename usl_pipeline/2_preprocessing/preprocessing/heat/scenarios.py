"""Heat scenario config records."""

import pathlib
import re
from google.cloud import firestore
from google.cloud import storage
import functions_framework
from usl_lib.readers import config_readers
from usl_lib.shared.function_runtime import (
    retry_and_report_errors as _retry_and_report_errors,
)
from usl_lib.storage import file_names
from usl_lib.storage import metastore


@functions_framework.cloud_event
@_retry_and_report_errors()
def write_heat_scenario_config_metadata(
    cloud_event: functions_framework.CloudEvent,
) -> None:
    """Writes metadata and features for uploaded heat scenario config.

    This function is triggered when files for heat configuration are uploaded to
    GCS. It writes an entry to the metastore for the configuration.
    """
    file_name = pathlib.PurePosixPath(cloud_event.data["name"])
    # For AtmoML, use `Heat_Data_` files to trigger HeatScenarioConfig write
    # to metastore
    if re.search(file_names.HEAT_CONFIG_TXT_REGEX, file_name.name):
        storage_client = storage.Client()
        db = firestore.Client()

        config_blob = storage_client.bucket(cloud_event.data["bucket"]).blob(
            cloud_event.data["name"]
        )
        with config_blob.open("rt") as txt_fd:
            config_dict = config_readers.read_key_value_pairs(txt_fd)

        metastore.HeatScenarioConfig(
            name=config_blob.name,
            # File names should be in the form:
            # <study_area>/<parent_config_name>/<file_name>
            # Ex: NYC_Heat/Summer_Config/Heat_Data_2012.txt
            parent_config_name=file_name.parent.name,
            gcs_uri=f"gs://{config_blob.bucket.name}/{config_blob.name}",
            simulation_year=int(config_dict["simulation_year"]),
            simulation_months=config_dict["simulation_months"],
            percentile=int(config_dict["percentile"]),
        ).set(db)
