"""Uploads WRF study area chunks."""

import pathlib
import re
from google.cloud import firestore
from google.cloud import storage
import flask
import functions_framework
import netCDF4
import wrf
from usl_lib.shared.function_runtime import error_to_response as _error_to_response
from usl_lib.shared.function_runtime import (
    retry_and_report_errors as _retry_and_report_errors,
)
from usl_lib.storage import cloud_storage
from usl_lib.storage import file_names
from usl_lib.storage import metastore


@functions_framework.cloud_event
@_retry_and_report_errors()
def build_and_upload_study_area_chunk(
    cloud_event: functions_framework.CloudEvent,
) -> None:
    return _build_and_upload_study_area_chunk(
        file_name=pathlib.PurePosixPath(cloud_event.data["name"]),
        bucket_name=cloud_event.data["bucket"],
    )


@functions_framework.http
@_error_to_response
def build_and_upload_study_area_chunk_http(
    request: flask.Request,
) -> flask.Response:
    request_json = request.get_json()
    _build_and_upload_study_area_chunk(
        file_name=pathlib.PurePosixPath(request_json["name"]),
        bucket_name=request_json["bucket"],
    )
    return flask.jsonify({"message", "Upload complete."})


def _build_and_upload_study_area_chunk(
    file_name: pathlib.PurePosixPath, bucket_name: str
):
    """Creates a study area chunk, uploads to bucket, and writes to metastore.

    This function is triggered when files containing raw geo data for a study area
    are uploaded to GCS. It will load the file into study area chunks bucket and
    write a new study area to metastore.
    """
    # For AtmoML, only 3rd domain (500m) output files will be used
    if re.search(file_names.WPS_DOMAIN3_NC_REGEX, file_name.name):
        storage_client = storage.Client()
        db = firestore.Client()

        bucket = storage_client.bucket(bucket_name)
        file_blob = bucket.blob(str(file_name))

        # For WRF, we can treat each snapshot output file as its own chunk. If
        # multiple snapshots must be processed together, we can choose to group
        # the files together in a TAR before uploading.
        with file_blob.open("rb") as fd:
            study_area_chunk_bucket = storage_client.bucket(
                cloud_storage.STUDY_AREA_CHUNKS_BUCKET
            )
            # TODO: Consider cheaper alt: `copy_blob()` here since file is a straight
            # copy
            study_area_chunk_bucket.blob(str(file_name)).upload_from_file(fd)

            # File names can be in the form <study_area_name>/<file_name>
            # or <study_area_name>/<some_sub_folder>/<file_name>
            study_area_name = file_name.parts[0]

            fd.seek(0)  # Seek to the beginning so the file can be read.
            nc_bytes = fd.read()
            # create a netCDF in-memory dataset from the bytes object.
            with netCDF4.Dataset("in_memory.nc", memory=nc_bytes) as ds:
                study_area = metastore.StudyArea(
                    name=study_area_name,
                    # Grid dimension will always be 200x200 for all cities
                    col_count=200,
                    row_count=200,
                    x_ll_corner=float(ds.getncattr("corner_lons")[0]),
                    y_ll_corner=float(ds.getncattr("corner_lats")[0]),
                    # https://www2.mmm.ucar.edu/wrf/users/namelist_best_prac_wps.html#dx_dy
                    cell_size=int(ds.getncattr("DX")),
                    # https://www2.mmm.ucar.edu/wrf/users/namelist_best_prac_wps.html#map_proj
                    crs=_get_crs_from_wps(ds),
                )
                study_area.set(db)


def _get_crs_from_wps(nc_dataset: netCDF4.Dataset) -> str:
    # The var used here doesn't matter - as long as it is a var that has coordinate
    # attributes
    wps_var = wrf.getvar(wrfin=nc_dataset, varname="XLAT_M")
    cart_proj = wrf.get_cartopy(wps_var)
    source_crs = cart_proj.source_crs.to_json_dict()["conversion"]["method"]["id"]
    crs = source_crs["authority"] + ":" + str(source_crs["code"])
    return crs
