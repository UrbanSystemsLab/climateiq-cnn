import dataclasses
import datetime
import functools
import io
import pathlib
import logging
import re
import tarfile
from typing import BinaryIO, Callable, IO, Optional, TextIO, Tuple

from google.api_core import exceptions
from google.cloud import error_reporting
from google.cloud import firestore
from google.cloud import storage
import functions_framework
import netCDF4
import numpy
from numpy.typing import NDArray
import xarray
import rasterio
from shapely import geometry
import wrf

from usl_lib.chunkers import raster_chunkers
from usl_lib.readers import simulation_readers
from usl_lib.readers import config_readers
from usl_lib.readers import elevation_readers, polygon_readers
from usl_lib.shared import geo_data, wps_data
from usl_lib.storage import cloud_storage
from usl_lib.storage import file_names
from usl_lib.storage import metastore
from usl_lib.transformers import feature_raster_transformers

# How long to accept cloud function invocations after the triggering
# event. It's up to GCP how frequently withing this period the cloud
# function is retried. This also needs to include lag for the length
# of time between when an event is triggered and when the cloud
# function actually runs, which can be a bit of time if there are a
# lot of triggers.
_MAX_RETRY_SECONDS = 60 * 60

# The vector is long enough to express rainfall every five minutes for up to three days.
_RAINFALL_VECTOR_LENGTH = (60 // 5) * 24 * 3


@dataclasses.dataclass(slots=True)
class FeatureMetadata:
    """Additional information about the extracted features.

    Attributes:
        elevation_min: The lowest elevation height encountered.
        elevation_max: The highest elevation height encountered.
    """

    elevation_min: Optional[float] = None
    elevation_max: Optional[float] = None
    chunk_size: Optional[int] = None
    time: Optional[datetime.datetime] = None


def _retry_and_report_errors(
    error_reporter_func: Optional[
        Callable[[functions_framework.CloudEvent, Exception], None]
    ] = None
) -> Callable[
    [Callable[[functions_framework.CloudEvent], None]],
    Callable[[functions_framework.CloudEvent], None],
]:
    """Adds error reporting and retry logic to a cloud function.

    If the decorated function raises an exception, the decorator will:
    - Log the exception to stderr, which will make the error appear in cloud function
      logs.
    - Log the exception to the GCP Error Reporter service.
    - If the caller supplies an `error_reporter_func` argument, calls that function with
      the cloud event and exception object, allowing callers to add additional custom
      error handling logic.

    If enough time has passed since the cloud function's initial invocation, then the
    decorated function will return rather than attempt to do work. This is to prevent
    retries of cloud functions from retrying indefinitely.

    Args:
      error_reporter_func: A function the caller may define to perform additional error
        handling. The funciont accepts a CloudEvent and Exception object. It may perform
        additional error logic, such as writing the error to the metastore.
    """

    def decorator(
        func: Callable[[functions_framework.CloudEvent], None]
    ) -> Callable[[functions_framework.CloudEvent], None]:
        """Decorator which adds error handling and retries to a cloud function."""

        @functools.wraps(func)
        def wrapper(cloud_event: functions_framework.CloudEvent) -> None:
            logging.basicConfig(level=logging.INFO)

            # Utilities like `gcloud storage cp` create temporary files for parallel
            # uploads and then combine the chunks in one final write. We want to avoid
            # the intermediary chunk writes.
            if cloud_event.data.get("name", "").startswith(
                "gcloud/tmp/parallel_composite_uploads"
            ):
                logging.debug("Skipping tmp upload file %s", cloud_event.data["name"])
                return

            # To handle retries, check the time created and return if the event is too
            # old. GCP retries until the function returns successfully, so the pattern
            # is to report exceptions to have the function retried and return normally
            # to stop retries.
            # https://cloud.google.com/functions/docs/bestpractices/retries
            event_time = datetime.datetime.fromisoformat(
                cloud_event.data["timeCreated"]
            )
            event_age = (
                datetime.datetime.now(datetime.timezone.utc) - event_time
            ).total_seconds()
            if event_age > _MAX_RETRY_SECONDS:
                logging.error(
                    "Dropped event id: %s, source: %s, name: %s after %s seconds",
                    cloud_event["id"],
                    cloud_event["source"],
                    cloud_event.data.get("name"),
                    _MAX_RETRY_SECONDS,
                )
                return

            # Catch exceptions and report them with GCP error reporting.
            try:
                func(cloud_event)
            except Exception as exc:  # noqa
                # Report the error in GCP Error Reporter.
                error_reporting.Client().report_exception()
                # Perform any custom error handling.
                if error_reporter_func is not None:
                    error_reporter_func(cloud_event, exc)
                # Raise the exception to indicate failure and allow GCP to retry.
                raise

        return wrapper

    return decorator


@functions_framework.cloud_event
@_retry_and_report_errors()
def build_and_upload_study_area_chunk(
    cloud_event: functions_framework.CloudEvent,
) -> None:
    """Creates a study area chunk and uploads it to GCS.

    This function is triggered when files containing raw geo data for a study area
    are uploaded to GCS. It will group the files into TAR-files and upload it to a
    bucket.
    """
    file_name = cloud_event.data["name"]
    bucket_name = cloud_event.data["bucket"]

    if re.search(file_names.WPS_DOMAIN3_NC_REGEX, file_name):
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        file_blob = bucket.blob(file_name)

        # For WRF, we can treat each snapshot output file as its own chunk. If
        # multiple snapshots must be processed together, we can choose to group
        # the files together in a TAR before uploading.
        with file_blob.open("rb") as f:
            study_area_chunk_bucket = storage_client.bucket(
                cloud_storage.STUDY_AREA_CHUNKS_BUCKET
            )
            study_area_chunk_bucket.blob(file_name).upload_from_file(f)

        # TODO: Write to metastore - create new chunk entry


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

        study_area = metastore.StudyArea(
            name=study_area_name,
            col_count=header.col_count,
            row_count=header.row_count,
            x_ll_corner=header.x_ll_corner,
            y_ll_corner=header.y_ll_corner,
            cell_size=header.cell_size,
            crs=header.crs.to_string() if header.crs is not None else None,
        )
        study_area.set(db)


@functions_framework.cloud_event
@_retry_and_report_errors()
def write_flood_scenario_metadata_and_features(
    cloud_event: functions_framework.CloudEvent,
) -> None:
    """Writes metadata and features for uploaded flood scenario config.

    This function is triggered when files for rainfall configuration are uploaded to
    GCS. It writes an entry to the metastore for the configuration and writes the
    rainfall configuration as a feature vector into the GCS features bucket.
    """
    file_name = pathlib.PurePosixPath(cloud_event.data["name"])
    # Distinguish between Rainfall_Data and CityCat_Config files, the latter of which we
    # don't need to record.
    if file_name.name.startswith("Rainfall_Data_"):
        storage_client = storage.Client()
        db = firestore.Client()

        config_blob = storage_client.bucket(cloud_event.data["bucket"]).blob(
            cloud_event.data["name"]
        )
        with config_blob.open("rt") as rain_fd:
            as_vector, length = _rain_config_as_vector(rain_fd)

        vector_blob = storage_client.bucket(cloud_storage.FEATURE_CHUNKS_BUCKET).blob(
            f"rainfall/{file_name.with_suffix('.npy')}"
        )
        _write_as_npy(vector_blob, as_vector)

        metastore.FloodScenarioConfig(
            name=config_blob.name,
            gcs_uri=f"gs://{config_blob.bucket.name}/{config_blob.name}",
            as_vector_gcs_uri=f"gs://{vector_blob.bucket.name}/{vector_blob.name}",
            rainfall_duration=length,
            # File names should be in the form <parent_config_name>/<file_name>
            parent_config_name=file_name.parent.name,
        ).set(db)


def _rain_config_as_vector(rain_fd: TextIO) -> Tuple[NDArray, int]:
    """Converts the rainfall config contents into a vector describing the rainfall.

    Args:
      rain_fd: The file containing the CityCAT rainfall configuration.

    Returns:
      A tuple of (rainfall, length) where `rainfall` contains
      the rainfall pattern as a numpy array, padded to an agreed-upon
      length, and `length` represent the un-padded number entries in
      the array (i.e. the length the array would be if it was not
      padded.)

    """
    entries = config_readers.read_rainfall_amounts(rain_fd)

    # The uploader should prevent configurations over this length.
    if len(entries) > _RAINFALL_VECTOR_LENGTH:
        raise ValueError(
            f"Rainfall configuration has unexpected length {len(entries)}. "
            f"Max allowed length is {_RAINFALL_VECTOR_LENGTH}."
        )

    return numpy.pad(
        numpy.array(entries, dtype=numpy.float32),
        (0, _RAINFALL_VECTOR_LENGTH - len(entries)),
    ), len(entries)


@functions_framework.cloud_event
@_retry_and_report_errors()
def delete_flood_scenario_metadata(cloud_event: functions_framework.CloudEvent) -> None:
    """Delete metadata for uploaded flood scenario config.

    This function is triggered when files for rainfall configuration are deleted from
    GCS and likewise removes them from the metastore.
    """
    file_name = pathlib.PurePosixPath(cloud_event.data["name"])
    # Distinguish between Rainfall_Data and CityCat_Config files, the latter of which we
    # don't need to record.
    if file_name.name.startswith("Rainfall_Data_"):
        db = firestore.Client()

        metastore.FloodScenarioConfig.delete(db, cloud_event.data["name"])


@functions_framework.cloud_event
@_retry_and_report_errors()
def build_label_matrix(cloud_event: functions_framework.CloudEvent) -> None:
    """Builds a label matrix when a set of simulation output files is uploaded.

    This function is triggered when files containing simulation data are uploaded to
    GCS. It produces a label matrix for the sim data and writes that label matrix to
    another GCS bucket for eventual use in model training and prediction.
    """
    _build_label_matrix(
        cloud_event.data["bucket"],
        cloud_event.data["name"],
        cloud_storage.LABEL_CHUNKS_BUCKET,
    )


def _build_label_matrix(bucket_name: str, chunk_name: str, output_bucket: str) -> None:
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    chunk_blob = bucket.blob(chunk_name)

    with chunk_blob.open("rb") as chunk:
        # Heat (WRF) - treat one WRF output file as one chunk
        if re.search(file_names.WRF_DOMAIN3_NC_REGEX, chunk_name):
            label_matrix, metadata = _build_wrf_label_matrix(chunk)

            label_file_name = chunk_name + ".npy"
            label_blob = storage_client.bucket(output_bucket).blob(str(label_file_name))

            _write_as_npy(label_blob, label_matrix)
            _write_wrf_label_chunk_metastore_entry(chunk_blob, label_blob, metadata)
        else:
            raise ValueError(f"Unexpected file {chunk_name}")


def _build_wrf_label_matrix(fd: IO[bytes]) -> Tuple[NDArray, FeatureMetadata]:
    label_components = []
    with netCDF4.Dataset("ncfile", mode="r", memory=fd.read()) as nc:
        nc.set_auto_mask(False)
        snapshot_time = nc.variables["Times"][:][0]
        # Variable names are listed here according to the formatting
        # specified by wrf-python documentation
        # https://wrf-python.readthedocs.io/en/latest/user_api/generated/wrf.getvar.html#wrf-getvar
        vars_to_derive = ["rh2", "T2", "wspd_wdir10"]
        for var in vars_to_derive:
            if var == "wspd_wdir10":
                # TODO: Split wind dir to sin/cos components
                wspd10, wdir10 = wrf.getvar(nc, "wspd_wdir10")
                label_components.append(numpy.swapaxes(wspd10, 0, 1))
                label_components.append(numpy.swapaxes(wdir10, 0, 1))
            else:
                label = wrf.getvar(nc, var)
                # Ensure order of dimension axes are: (west_east, south_north)
                label_components.append(numpy.swapaxes(label, 0, 1))

    labels_matrix = numpy.dstack(label_components)
    return labels_matrix, FeatureMetadata(
        time=datetime.datetime.fromisoformat(str(snapshot_time))
    )


def _write_wrf_label_chunk_metastore_entry(
    chunk_blob: storage.Blob, label_blob: storage.Blob, metadata: FeatureMetadata
) -> None:
    """Updates the metastore with new information for the given wrf chunks.

    Args:
      chunk_blob: The GCS blob containing the raw chunk.
      label_blob: The GCS blob containing the label tensor for the chunk.
      metadata: Additional metadata describing the features.
    """
    db = firestore.Client()
    study_area_name = _parse_chunk_path(chunk_blob.name)[0]

    metastore.SimulationLabelTemporalChunk(
        gcs_uri=f"gs://{label_blob.bucket.name}/{label_blob.name}",
        time=metadata.time,
    ).set(db, study_area_name, config_path=None)


@functions_framework.cloud_event
@_retry_and_report_errors(
    lambda cloud_event, exc: _write_chunk_metastore_error(
        cloud_event.data["name"], str(exc)
    )
)
def build_feature_matrix(cloud_event: functions_framework.CloudEvent) -> None:
    """Builds a feature matrix when an a set of geo files is uploaded.

    This function is triggered when files containing geo data are uploaded to
    GCS. It produces a feature matrix for the geo data and writes that feature matrix to
    another GCS bucket for eventual use in model training and prediction.
    """
    _build_feature_matrix(
        cloud_event.data["bucket"],
        cloud_event.data["name"],
        cloud_storage.FEATURE_CHUNKS_BUCKET,
    )


def _build_feature_matrix(
    bucket_name: str, chunk_name: str, output_bucket: str
) -> None:
    """Builds a feature matrix when a set of geo files is uploaded."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    chunk_blob = bucket.blob(chunk_name)

    feature_file_name = pathlib.PurePosixPath(chunk_name).with_suffix(".npy")
    feature_blob = storage_client.bucket(output_bucket).blob(str(feature_file_name))

    # TODO: Refactor to better handle both tar'ed + un-tarred chunks
    with chunk_blob.open("rb") as chunk:
        # Flood (CityCat)
        if chunk_name.endswith(".tar"):
            feature_matrix, metadata = _build_flood_feature_matrix_from_archive(chunk)

            if feature_matrix is None:
                raise ValueError(f"Empty archive found in {chunk_blob}")

            # Updating min/max elevation in the study area metadata first before storing
            # feature matrix file that will trigger rescaling post-processing.
            _update_study_area_metastore_entry(chunk_blob, metadata)
            _write_as_npy(feature_blob, feature_matrix)
            _write_flood_chunk_metastore_entry(chunk_blob, feature_blob)

        # Heat (WRF) - treat one WPS outout file as one chunk
        elif re.search(file_names.WPS_DOMAIN3_NC_REGEX, chunk_name):
            feature_matrix, metadata = _build_wps_feature_matrix(chunk)
            _write_as_npy(feature_blob, feature_matrix)
            _write_wps_chunk_metastore_entry(chunk_blob, feature_blob, metadata)
        else:
            raise ValueError(f"Unexpected file {chunk_name}")


@functions_framework.cloud_event
@_retry_and_report_errors()
def process_citycat_outputs(cloud_event: functions_framework.CloudEvent) -> None:
    """Creates training labels as CityCAT result files are uploaded.

    CityCAT creates one .rsl file for each time step in the simulation, representing
    simulated flood heights for that timestep. For each timestep's .rsl file:
    - Rasterize the output file and break it into chunks, as we do with features. The
      labels need to be chunked identically to the features.
    - Write each chunk as a numpy matrix for the timestep into paths like
      <x_chunk_index>_<y_chunk_index>/<timestep_number>.npy

    This same function will then be triggered for each chunk upload. For each chunk:
    - Look up the expected number of timesteps in the metastore associated with this
      simulation's rainfall configuration.
    - Check to see if corresponding chunks for the same <x_chunk_index> and
      <y_chunk_index> have been uploaded for all expected timesteps.
    - If chunks for all timesteps are present, collapse them and write the resulting
      combined tensor into the labels bucket.
    - If not, return and let another chunk upload trigger the above collapsing logic.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(cloud_event.data["bucket"])
    blob = bucket.blob(cloud_event.data["name"])
    blob_path = pathlib.PurePosixPath(cloud_event.data["name"])

    if blob_path.suffix == ".rsl":
        _write_city_cat_output_chunks(blob, blob_path)
    elif blob_path.suffix == ".npy":
        _collapse_city_cat_output_chunks(
            blob, blob_path, storage_client.bucket(cloud_storage.LABEL_CHUNKS_BUCKET)
        )
    else:
        logging.error("Unexpected file %s", blob_path)


def _write_city_cat_output_chunks(
    blob: storage.Blob, blob_path: pathlib.PurePosixPath, chunk_size: int = 1000
) -> None:
    """Writes each chunk of the CityCAT result file for further processing.

    Breaks the given CityCAT .rsl result file for one timestep into chunks and writes
    them into GCS as numpy arrays. Writing these arrays re-triggers the
    `process_citycat_outputs` cloud function, which will then call
    `_collapse_city_cat_output_chunks` to collapse the chunks for each timestep.

    See the documentation for process_citycat_outputs for more details.

    Args:
      blob: The blob containing the CityCAT .rsl result file.
      blob_path: The path to the blob within its bucket.
      chunk_size: The number of cells in each side of the chunk square.
    """
    # CityCAT output file paths are of the form:
    # <study_area_name>/<config/path>/R11_C1_T<x>_y>min.rsl
    study_area_name = blob_path.parts[0]
    config_path = str(pathlib.PurePosixPath(*blob_path.parts[1:-1]))
    logging.info(
        "Processing file %s for study area %s simulation config %s",
        blob,
        study_area_name,
        config_path,
    )

    # Enter the simulation in our metastore.
    db = firestore.Client()
    metastore.Simulation(
        gcs_prefix_uri=f"gs://{blob.bucket.name}/{blob_path.parent}",
        simulation_type=metastore.SimulationType.CITY_CAT,
        study_area=metastore.StudyArea.get_ref(db, study_area_name),
        configuration=metastore.FloodScenarioConfig.get_ref(db, config_path),
    ).set(db)

    # Retrieve geography information for the study area to rasterize CityCAT results.
    header = metastore.StudyArea.get(db, study_area_name).as_header()

    chunk_prefix = pathlib.PurePosixPath("timestep_parts") / blob_path.parent
    timestep = _timestep_number_from_rsl_path(blob_path.name)
    logging.info(
        "Writing chunks of %s for timestep %s to prefix %s",
        blob,
        timestep,
        chunk_prefix,
    )

    with blob.open("rt") as citycat_result:
        depths = simulation_readers.read_city_cat_result_as_raster(
            citycat_result, header
        )
    logging.info("%s loaded, now being chunked.", blob)

    for x_index, y_index, chunk in raster_chunkers.split_raster_into_chunks(
        chunk_size, depths
    ):
        # Write chunks to paths like <x_index>_<y_index>/<timestep>.npy so that chunks
        # at every timestep for the same location can be found under the same
        # <x_index>_<y_index>/ prefix
        chunk_name = chunk_prefix / f"{x_index}_{y_index}" / f"{timestep}.npy"
        with blob.bucket.blob(str(chunk_name)).open("wb") as chunk_file:
            numpy.save(chunk_file, chunk)

    logging.info("Processed all chunks for file %s", blob)


def _collapse_city_cat_output_chunks(
    blob: storage.Blob, blob_path: pathlib.PurePosixPath, labels_bucket: storage.Bucket
) -> None:
    """Collects chunks for all timesteps at the same region and writes them as labels.

    When a chunk file is uploaded, see if other chunks for the same region have been
    uploaded for all timesteps. If so, collapse them into a single 3D tensor and write
    them into the labels bucket.

    See the documentation for process_citycat_outputs for more details.

    Args:
      blob: The blob containing the chunk .npy file.
      blob_path: The path to the blob within its bucket.
      labels_bucket: The bucket in which to write the label tensor.
    """
    # Chunks are of the form
    # timestep_parts/<study_area_name>/<config/path>/<x_index>_<y_index>/<timestep>.npy
    study_area_name = blob_path.parts[1]
    # Note that the config name be multiple 'folders' e.g. config_name/rainfall_4.txt
    config_path = str(pathlib.PurePosixPath(*blob_path.parts[2:-2]))
    x_index, y_index = blob_path.parts[-2].split("_")
    logging.info("Processing chunk %s for config %s", blob, config_path)

    # Retrieve the number of timesteps configured for this result file's simulation.
    db = firestore.Client()
    timesteps = metastore.FloodScenarioConfig.get(db, config_path).rainfall_duration

    # See if a file is present in GCS for each timestep.
    expected_names = set(
        str(blob_path.with_stem(str(timestep))) for timestep in range(timesteps)
    )
    timestep_blobs = list(blob.bucket.list_blobs(prefix=f"{blob_path.parent}/"))
    timestep_blob_names = set(b.name for b in timestep_blobs)
    if not expected_names == timestep_blob_names:
        logging.info(
            "Chunks for some timesteps missing, halting. Expected %s got %s.",
            expected_names,
            timestep_blob_names,
        )
        return

    # Sort the blobs by their timestep number so we assemble them in order.
    timestep_blobs.sort(key=lambda b: int(pathlib.PurePosixPath(b.name).stem))
    chunk_timestep_parts = []
    for ts_blob in timestep_blobs:
        try:
            with ts_blob.open("rb") as fd:
                chunk_timestep_parts.append(numpy.load(fd))
        except exceptions.NotFound:
            # Another invocation of this function has collapsed the chunks while this
            # invocation was running.
            logging.info("Timestep chunk deletion in process, halting.")
            return

    # Stack the matrices at each timestep together and write them to the labels bucket.
    full_chunk = numpy.dstack(chunk_timestep_parts)
    label_path = pathlib.PurePosixPath(*blob_path.parts[1:-1]).with_suffix(".npy")
    with labels_bucket.blob(str(label_path)).open("wb") as fd:
        numpy.save(fd, full_chunk)
        logging.info("Wrote full chunk after processing %s.", blob)

    # Delete the chunks now that we're done with them.
    for ts_blob in timestep_blobs:
        try:
            ts_blob.delete()
        except exceptions.NotFound:
            # Another invocation of this function has collapsed the chunks while this
            # invocation was running.
            continue

    metastore.SimulationLabelSpatialChunk(
        gcs_uri=f"gs://{labels_bucket.name}/{label_path}",
        x_index=int(x_index),
        y_index=int(y_index),
    ).set(db, study_area_name, config_path)

    logging.info(
        "Deleted timestep chunks %s after processing %s.", timestep_blobs, blob
    )


def _timestep_number_from_rsl_path(rsl_path: str) -> str:
    """Extracts the timestep number x from paths like R11_C1_T<x>_<y>min.rsl."""
    time_step_num = re.search(r"R\d+_C\d+_T(\d+)", rsl_path)
    if time_step_num is None:
        raise ValueError(
            f"Unexpected path {rsl_path} expected path of the form R<x>_C<y>_T<n>_"
        )

    return time_step_num.group(1)


def _read_polygons_from_byte_stream(
    stream: IO[bytes],
) -> list[Tuple[geometry.Polygon, int]]:
    """Reads polygons from textual file content provided in form of a byte stream."""
    return list(polygon_readers.read_polygons_from_text_file(io.TextIOWrapper(stream)))


def _build_flood_feature_matrix_from_archive(
    archive: BinaryIO,
) -> Tuple[Optional[NDArray], FeatureMetadata]:
    """Builds a feature matrix for the given archive.

    Args:
      archive: The file containing the archive.

    Returns:
      A tuple of (array, metadata) for the feature matrix and
      metadata describing the feature matrix.
    """
    metadata: FeatureMetadata = FeatureMetadata()
    elevation: Optional[geo_data.Elevation] = None
    boundaries: Optional[list[Tuple[geometry.Polygon, int]]] = None
    buildings: Optional[list[Tuple[geometry.Polygon, int]]] = None
    green_areas: Optional[list[Tuple[geometry.Polygon, int]]] = None
    soil_classes: Optional[list[Tuple[geometry.Polygon, int]]] = None
    files_in_tar = []

    with tarfile.TarFile(fileobj=archive) as tar:
        for member in tar:
            fd = tar.extractfile(member)
            if fd is None:
                continue

            name = pathlib.PurePosixPath(member.name).name
            files_in_tar.append(name)
            # Handle flood model input files (CityCat)
            if name == file_names.ELEVATION_TIF:
                elevation, metadata = _read_elevation_features(fd)
            elif name == file_names.BOUNDARIES_TXT:
                boundaries = _read_polygons_from_byte_stream(fd)
            elif name == file_names.BUILDINGS_TXT:
                buildings = _read_polygons_from_byte_stream(fd)
            elif name == file_names.GREEN_AREAS_TXT:
                green_areas = _read_polygons_from_byte_stream(fd)
            elif name == file_names.SOIL_CLASSES_TXT:
                soil_classes = _read_polygons_from_byte_stream(fd)
            else:
                logging.warning(f"Unexpected member name: {name}")

    flood_files_present = [
        file is not None for file in (elevation, buildings, green_areas, soil_classes)
    ]
    if any(flood_files_present):
        if not all(flood_files_present):
            raise ValueError(
                f"Some flood simulation data missing (see tar list: {files_in_tar})"
            )
        # MyPy can't figure out that the all() call above prevents arguments in the
        # following call from being None.
        feature_matrix = feature_raster_transformers.transform_to_feature_raster_layers(
            elevation,  # type: ignore
            boundaries,
            buildings,  # type: ignore
            green_areas,  # type: ignore
            soil_classes,  # type: ignore
            geo_data.DEFAULT_INFILTRATION_CONFIGURATION,
        )
        return feature_matrix, metadata

    return None, FeatureMetadata()


def _build_wps_feature_matrix(fd: IO[bytes]) -> Tuple[NDArray, FeatureMetadata]:
    # Ignore type checker error - BytesIO inherits from expected type BufferedIOBase
    # https://shorturl.at/lk4om
    with xarray.open_dataset(fd, engine="h5netcdf") as ds:  # type: ignore
        # Assign Time coordinates so datetime is associated with each data array
        ds = ds.assign_coords(Time=ds.Times)
        # Derive non-native variables and assign to dataset
        ds = _compute_custom_wps_variables(ds)

        # Apply feature engineering and build features matrix
        features_components = []
        for var_name in wps_data.ML_REQUIRED_VARS_REPO.keys():
            var_config = wps_data.ML_REQUIRED_VARS_REPO[var_name]
            feature = _process_wps_feature(
                feature=ds.data_vars[var_name], var_config=var_config
            )
            features_components.append(feature)

        features_matrix = numpy.dstack(features_components)

    return features_matrix, FeatureMetadata(
        time=datetime.datetime.fromisoformat(str(ds.Times.values[0]))
    )


def _compute_custom_wps_variables(dataset: xarray.Dataset) -> xarray.Dataset:
    """Computes additional, non-native variables for WPS datasets.

    Args:
      dataset: The xarray dataset containing all WPS data and its metadata.

    Returns:
      A new xarray dataset with newly derived variables.
    """
    # Derive wind components: WSPD10 (wind speed), WDIR10 (wind direction)
    if all(var in dataset.keys() for var in ["UU", "VV"]):
        uu = dataset.data_vars["UU"]
        vv = dataset.data_vars["VV"]

        # Interpolate UU and VV to the common 200x200 grid
        uu_centered = (uu[:, :, :, :-1] + uu[:, :, :, 1:]) / 2
        vv_centered = (vv[:, :, :-1, :] + vv[:, :, 1:, :]) / 2

        wind_speed = numpy.sqrt(uu_centered.values**2 + vv_centered.values**2)
        wind_direction = (
            270 - numpy.degrees(numpy.arctan2(vv_centered.values, uu_centered.values))
        ) % 360

        new_dims = ["Time", "num_metgrid_levels", "south_north", "west_east"]
        dataset = dataset.assign(WSPD10=(new_dims, wind_speed))
        dataset = dataset.assign(WDIR10=(new_dims, wind_direction))

    return dataset


def _process_wps_feature(feature: xarray.DataArray, var_config: dict) -> NDArray:
    """Performs a series of data transforms on a WPS variable.

    Args:
      feature: The xarray.DataArray containing the feature, its dimensions,
      and metadata
      var_config: The dict entry to ML_REQUIRED_VARS_REPO from wps_data.py containing
      the metadata that will be used to determine what feature engineering processing
      should be applied

    Returns:
      A new numpy array with transforms applied according to rules
      for each variable defined in: https://shorturl.at/W6nzY
    """
    snapshot_time = feature.coords.get("Time")
    # Ensure order of dimension axes are: (west_east, south_north, <other spatial dims>)
    # to stay consistent with rest of data pipeline
    feature = feature.transpose("west_east", "south_north", ...)

    # FNL-derived var - extract to only first level
    if "num_metgrid_levels" in feature.dims:
        feature = feature.isel(num_metgrid_levels=0)

    # Monthly climatology var - extract to only the month of the file's datestamp
    if "z-dimension0012" in feature.dims and snapshot_time:
        snapshot_datetime = datetime.datetime.strptime(
            snapshot_time.values[0].decode(), "%Y-%m-%d_%H:%M:%S"
        )
        feature = feature.isel({"z-dimension0012": snapshot_datetime.month - 1})

    # At this point, we can remove the Time axis from current feature DataArray
    # since we no longer need it for processing or in the feature matrix for ML
    # model. xarray.DataArry.isel() will return new DataArray with axis dropped.
    feature = feature.isel(Time=0)

    feature_values = feature.values

    # Convert percentage-based units to decimal
    if var_config.get("unit") == wps_data.Unit.PERCENTAGE:
        feature_values = _convert_to_decimal(feature_values)

    # If var config requires global scaling, normalize feature values
    scaling_config = var_config.get("scaling")
    if (
        scaling_config is not None
        and scaling_config.get("type") == wps_data.ScalingType.GLOBAL
    ):
        min = scaling_config.get("min")
        max = scaling_config.get("max")
        feature_values = _apply_minmax_scaler(feature_values, min, max)

    return feature_values


def _convert_to_decimal(x):
    return x / 100


def _apply_minmax_scaler(x, minx, maxx):
    # Clip values if they go out of bounds of the specified min and max
    x[x > maxx] = maxx
    x[x < minx] = minx

    return (x - minx) / (maxx - minx)


def _calculate_metadata_for_elevation(elevation: geo_data.Elevation) -> FeatureMetadata:
    """Calculates metadata for the elevation data."""
    elevation_data = elevation.data

    if elevation_data is None:
        raise ValueError("Elevation file unexpectedly empty.")

    # NODATA cells should be excluded from min/max calculation
    present_data = elevation_data[elevation_data != elevation.header.nodata_value]

    if present_data.size == 0:
        # All the cells in the study area have NODATA values.
        return FeatureMetadata()

    return FeatureMetadata(
        elevation_min=float(present_data.min()),
        elevation_max=float(present_data.max()),
        chunk_size=elevation.header.row_count,
    )


def _read_elevation_features(
    fd: IO[bytes],
) -> Tuple[geo_data.Elevation, FeatureMetadata]:
    """Reads the elevation file into a matrix and returns metadata for the matrix.

    Args:
      fd: The file containing elevation data.

    Returns:
      A tuple of (Elevation, metadata) for the elevation data and metadata describing
      the feature matrix.
    """
    elevation = elevation_readers.read_from_geotiff(rasterio.io.MemoryFile(fd.read()))

    return elevation, _calculate_metadata_for_elevation(elevation)


def _update_study_area_metastore_entry(
    chunk_blob: storage.Blob, metadata: FeatureMetadata
) -> None:
    """Updates the study area metadata with new information for a given chunk.

    Updates elevation_min and elevation_max values supplied in the metadata for the
    whole study area.

    Args:
      chunk_blob: The GCS blob containing the raw chunk archive.
      metadata: Additional metadata describing the features.
    """
    db = firestore.Client()
    study_area_name, _ = _parse_chunk_path(chunk_blob.name)

    if metadata.elevation_min is not None and metadata.elevation_max is not None:
        metastore.StudyArea.update_min_max_elevation(
            db,
            study_area_name,
            min_=metadata.elevation_min,
            max_=metadata.elevation_max,
        )
    if metadata.chunk_size is not None:
        metastore.StudyArea.update_chunk_info(db, study_area_name, metadata.chunk_size)


def _write_flood_chunk_metastore_entry(
    chunk_blob: storage.Blob, feature_blob: storage.Blob
) -> None:
    """Updates the metastore with new information for the given flood chunks.

    Writes a Firestore entry for the new feature matrix chunk.

    Args:
      chunk_blob: The GCS blob containing the raw chunk.
      feature_blob: The GCS blob containing the feature tensor for the chunk.
    """
    db = firestore.Client()
    study_area_name, chunk_name = _parse_chunk_path(chunk_blob.name)
    x_index, y_index = _parse_spatial_chunk_indices_from_name(chunk_blob.name)

    metastore.StudyAreaSpatialChunk(
        id_=chunk_name,
        raw_path=f"gs://{chunk_blob.bucket.name}/{chunk_blob.name}",
        feature_matrix_path=f"gs://{feature_blob.bucket.name}/{feature_blob.name}",
        x_index=x_index,
        y_index=y_index,
        # Remove any errors from previous failed retries which have now succeeded.
        error=firestore.DELETE_FIELD,
    ).merge(db, study_area_name)


def _write_wps_chunk_metastore_entry(
    chunk_blob: storage.Blob, feature_blob: storage.Blob, metadata: FeatureMetadata
) -> None:
    """Updates the metastore with new information for the given wps chunks.

    Writes a Firestore entry for the new feature matrix chunk.

    Args:
      chunk_blob: The GCS blob containing the raw chunk.
      feature_blob: The GCS blob containing the feature tensor for the chunk.
      metadata: Additional metadata describing the features.
    """
    db = firestore.Client()
    study_area_name, chunk_name = _parse_chunk_path(chunk_blob.name)

    metastore.StudyAreaTemporalChunk(
        id_=chunk_name,
        raw_path=f"gs://{chunk_blob.bucket.name}/{chunk_blob.name}",
        feature_matrix_path=f"gs://{feature_blob.bucket.name}/{feature_blob.name}",
        time=metadata.time,
        # Remove any errors from previous failed retries which have now succeeded.
        error=firestore.DELETE_FIELD,
    ).merge(db, study_area_name)


def _write_chunk_metastore_error(chunk_path: str, error_message: str) -> None:
    """Updates the metastore with an error message for this chunk.

    Args:
      chunk_path: The GCS path to the chunk being processed.
      error_message: The error message to write into Firestore.
    """
    db = firestore.Client()
    study_area_name, chunk_name = _parse_chunk_path(chunk_path)
    metastore.StudyAreaChunk(id_=chunk_name, error=error_message).merge(
        db, study_area_name
    )


def _parse_chunk_path(chunk_path: str) -> Tuple[str, str]:
    """Parses a GCS chunk path into (study_area, chunk) components.

    Args:
      chunk_path: The GCS path of a chunk to parse.

    Returns:
      Given a GCS path in the form "study_area_name/chunk_name" returns a tuple
      (study_area_name, chunk_name)
    """
    as_path = pathlib.PurePosixPath(chunk_path)
    return str(as_path.parent), as_path.stem


def _parse_spatial_chunk_indices_from_name(chunk_name: str) -> Tuple[int, int]:
    """Extract the x, y indices from a name like "chunk_x_y.tar"."""
    indices = re.search(r"(\d+)_(\d+)", chunk_name)
    if indices is None:
        raise ValueError(
            f"Unexpected chunk name {chunk_name} does not contain chunk indices"
        )
    return (int(indices.group(1)), int(indices.group(2)))


def _write_as_npy(blob: storage.Blob, array: NDArray) -> None:
    """Writes `array` into the GCS `blob` in the numpy binary array format.

    Args:
      blob: The blob to write the array into.
      array: The array to upload.
    """
    npy_file = io.BytesIO()
    numpy.save(npy_file, array)
    # Seek to the beginning so the file can be read.
    npy_file.flush()
    npy_file.seek(0)
    blob.upload_from_file(npy_file)
