"""Records a study area when its header is uploaded."""

import pathlib
from google.cloud import firestore
from google.cloud import storage
import functions_framework
from usl_lib.readers import elevation_readers
from usl_lib.shared.function_runtime import (
    retry_and_report_errors as _retry_and_report_errors,
)
from usl_lib.storage import file_names
from usl_lib.storage import metastore


@functions_framework.cloud_event
@_retry_and_report_errors()
def write_study_area_metadata(cloud_event: functions_framework.CloudEvent) -> None:
    """Writes metadata for a study area on upload.

    This function is triggered when files containing raw geo data for an entire study
    area are uploaded to GCS. It writes an entry to the metastore for the new study
    area. It also calculates min & max values needed for feature scaling of the relevant
    files.
    """
    file_name = pathlib.PurePosixPath(cloud_event.data["name"])
    if file_name.name == file_names.HEADER_JSON:
        storage_client = storage.Client()
        db = firestore.Client()

        bucket = storage_client.bucket(cloud_event.data["bucket"])
        blob = bucket.blob(str(file_name))

        with blob.open("rb") as fd:
            # We're only reading the header, so reading the first MB is plenty.
            header = elevation_readers.read_header_from_json_file(fd)
        # File names should be in the form <study_area_name>/<file_name>
        study_area_name = file_name.parent.name

        metastore.StudyArea.delete_all_chunks(db, study_area_name)
        study_area = metastore.StudyArea(
            name=study_area_name,
            state=metastore.StudyAreaState.INIT,
            col_count=header.col_count,
            row_count=header.row_count,
            x_ll_corner=header.x_ll_corner,
            y_ll_corner=header.y_ll_corner,
            cell_size=header.cell_size,
            crs=header.crs.to_string() if header.crs is not None else None,
        )
        study_area.set(db)
