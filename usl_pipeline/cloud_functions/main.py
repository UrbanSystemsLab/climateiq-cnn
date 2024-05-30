import dataclasses
import datetime
import functools
import itertools
import io
import pathlib
import logging
import re
import tarfile
from typing import BinaryIO, Callable, Dict, IO, Optional, Sequence, TextIO, Tuple
import zipfile

from google.cloud import error_reporting
from google.cloud import firestore
from google.cloud import storage
import functions_framework
import numpy
from numpy.typing import NDArray
import rasterio

from usl_lib.chunkers import raster_chunkers
from usl_lib.readers import simulation_readers
from usl_lib.readers import config_readers
from usl_lib.readers import elevation_readers
from usl_lib.shared import geo_data
from usl_lib.storage import cloud_storage
from usl_lib.storage import file_names
from usl_lib.storage import metastore

_MAX_RETRY_SECONDS = 60 * 2

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
            logging.basicConfig(level=logging.WARN)

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
                    "Dropped event id %s source %s after %s seconds",
                    cloud_event["id"],
                    cloud_event["source"],
                    _MAX_RETRY_SECONDS,
                )
                return

            # Catch exceptions and report them with GCP error reporting.
            try:
                func(cloud_event)
            except Exception as exc:  # noqa
                # Log the exception to display in Cloud Function logs.
                logging.exception(exc)
                # Report the error in GCP Error Reporter.
                error_reporting.Client().report_exception()
                # Perform any custom error handling.
                if error_reporter_func is not None:
                    error_reporter_func(cloud_event, exc)

        return wrapper

    return decorator


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
    if file_name.name == file_names.ELEVATION_TIF:
        storage_client = storage.Client()
        db = firestore.Client()

        bucket = storage_client.bucket(cloud_event.data["bucket"])
        blob = bucket.blob(str(file_name))

        with blob.open("rb") as fd:
            # We're only reading the header, so reading the first MB is plenty.
            header = elevation_readers.read_from_geotiff(
                rasterio.io.MemoryFile(fd.read(1048576)), header_only=True
            ).header
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
            gcs_uri=f"gs://{config_blob.bucket.name}/{config_blob.name}",
            as_vector_gcs_uri=f"gs://{vector_blob.bucket.name}/{vector_blob.name}",
            num_rainfall_entries=length,
            # File names should be in the form <parent_config_name>/<file_name>
            parent_config_name=file_name.parent.name,
        ).set(db, config_blob.name)


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
@_retry_and_report_errors(
    lambda cloud_event, exc: _write_chunk_metastore_error(
        cloud_event.data["name"], str(exc)
    )
)
def build_feature_matrix(cloud_event: functions_framework.CloudEvent) -> None:
    """Builds a feature matrix when an archive of geo files is uploaded.

    This function is triggered when archive files containing geo data are uploaded to
    GCS. It produces a feature matrix for the geo data and writes that feature matrix to
    another GCS bucket for eventual use in model training and prediction.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(cloud_event.data["bucket"])
    chunk_name = cloud_event.data["name"]
    chunk_blob = bucket.blob(chunk_name)

    with chunk_blob.open("rb") as archive:
        feature_matrix, metadata = _build_feature_matrix_from_archive(archive)
        if feature_matrix is None:
            raise ValueError(f"Empty archive found in {chunk_blob}")

    feature_file_name = pathlib.PurePosixPath(chunk_name).with_suffix(".npy")
    feature_blob = storage_client.bucket(cloud_storage.FEATURE_CHUNKS_BUCKET).blob(
        str(feature_file_name)
    )

    _write_as_npy(feature_blob, feature_matrix)
    _write_metastore_entry(chunk_blob, feature_blob, metadata)


@functions_framework.cloud_event
@_retry_and_report_errors()
def process_citycat_outputs(cloud_event: functions_framework.CloudEvent) -> None:
    storage_client = storage.Client()
    bucket = storage_client.bucket(cloud_event.data["bucket"])
    blob = bucket.blob(cloud_event.data["name"])

    db = firestore.Client()
    path = pathlib.PurePosixPath(cloud_event.data["name"])
    study_area_name = path.parts[0]
    header = metastore.StudyArea.get(db, study_area_name).as_header()

    config_name = pathlib.PurePosixPath(*path.parts[1:-1])
    timesteps = metastore.FloodScenarioConfig.get(config_name).num_rainfall_entries

    labels_bucket = storage_client.bucket(cloud_storage.LABEL_CHUNKS_BUCKET)
    with blob.open("rb") as fd:
        _write_outputs(labels_bucket, path, fd, header)


def _write_outputs(
    labels_bucket: storage.Bucket,
    city_cat_results_path: pathlib.PurePosixPath,
    city_cat_results_zip: BinaryIO,
    header: geo_data.ElevationHeader,
):
    buf_prefix = city_cat_results_path.parent / "buffers"
    parts: Dict[Tuple[int, int], storage.Blob] = {}
    with zipfile.ZipFile(city_cat_results_zip) as timesteps:
        names = [path for path in timesteps.namelist() if path.endswith(".rsl")]
        names.sort(key=_timestep_number_from_rsl_path)
        for i, name in enumerate(names):
            print(f"{datetime.datetime.now()} processing {name}")
            with timesteps.open(name) as timestep:
                depths = simulation_readers.read_city_cat_result_as_raster(
                    timestep, header
                )
                print(f"{datetime.datetime.now()} read {name}")
                for x_index, y_index, chunk in raster_chunkers.split_raster_into_chunks(
                    1000, depths
                ):
                    print(
                        f"{datetime.datetime.now()} processing chunk {x_index} {y_index}"
                    )
                    key = (x_index, y_index)
                    part_blob = labels_bucket.blob(
                        str(buf_prefix / f"{i}_{x_index}_{y_index}.npy")
                    )

                    print(
                        f"{datetime.datetime.now()} writing chunk {x_index} {y_index}"
                    )
                    with part_blob.open("wb") as chunk_file:
                        numpy.save(chunk_file, chunk)

                    print(f"{datetime.datetime.now()} wrote chunk {x_index} {y_index}")
                    parts.setdefault(key, []).append(part_blob)

    for (x_index, y_index), chunk_parts in parts.items():
        print(f"{datetime.datetime.now()} collapsing chunk {x_index}_{y_index}")
        part_arrays = []
        for part_blob in chunk_parts:
            with part_blob.open("rb") as fd:
                part_arrays.append(numpy.load(fd))

        full_chunk = numpy.dstack(part_arrays)
        with labels_bucket.blob(
            city_cat_results_path.parent / f"label_chunk_{x_index}_{y_index}.npy"
        ).open("wb") as fd:
            numpy.save(fd, full_chunk)

    labels_bucket.delete_blobs(itertools.chain.from_iterable(parts.keys()))


def _create_buffer_for(
    bucket: storage.Bucket, prefix: pathlib.PurePosixPath, index: Tuple[int, int]
) -> TextIO:
    return bucket.open(str(prefix / str(index)), "wt")


def _timestep_number_from_rsl_path(rsl_path: str) -> int:
    # The files have paths like R11_C1_T0_0min.rsl, R11_C1_T1_5min.rsl
    # We want to extract the _T0_, _T1_ aspects of the paths.
    time_step_num = re.search("R11_C1_T(\d+)", rsl_path)
    if time_step_num is None:
        raise ValueError(
            f"Unexpected path {rsl_path} expect path of the form R11_C1_T<n>_"
        )

    return int(time_step_num.group(1))


def _build_feature_matrix_from_archive(
    archive: BinaryIO,
) -> Tuple[Optional[NDArray], FeatureMetadata]:
    """Builds a feature matrix for the given archive.

    Args:
      archive: The file containing the archive.

    Returns:
      A tuple of (array, metadata) for the feature matrix and
      metadata describing the feature matrix.
    """
    with tarfile.TarFile(fileobj=archive) as tar:
        for member in tar:
            fd = tar.extractfile(member)
            if fd is None:
                continue

            name = pathlib.PurePosixPath(member.name).name
            if name == file_names.ELEVATION_TIF:
                return _read_elevation_features(fd)
            # TODO: handle additional archive members.
            else:
                logging.warning(f"Unexpected member name: {name}")

    return None, FeatureMetadata()


def _read_elevation_features(fd: IO[bytes]) -> Tuple[NDArray, FeatureMetadata]:
    """Reads the elevation file into a matrix and returns metadata for the matrix.

    Args:
      fd: The file containing elevation data.

    Returns:
      A tuple of (array, metadata) for the elevation matrix and
      metadata describing the feature matrix.
    """
    elevation = elevation_readers.read_from_geotiff(
        rasterio.io.MemoryFile(fd.read())
    ).data

    if elevation is None:
        raise ValueError("Elevation file unexpectedly empty.")

    metadata = FeatureMetadata(
        elevation_min=float(elevation.min()),
        elevation_max=float(elevation.max()),
    )

    return elevation, metadata


def _write_metastore_entry(
    chunk_blob: storage.Blob, feature_blob: storage.Blob, metadata: FeatureMetadata
) -> None:
    """Updates the metastore with new information for the given chunks.

    Writes a Firestore entry for the new feature matrix chunk. Updates elvation_min and
    elevation_max values supplied in the metadata for the whole study area.

    Args:
      chunk_blob: The GCS blob containing the raw chunk archive.
      feature_blob: The GCS blob containing the feature tensor for the chunk.
      metadata: Additional metadata describing the features.
    """
    db = firestore.Client()
    study_area_name, chunk_name = _parse_chunk_path(chunk_blob.name)

    metastore.StudyAreaChunk(
        id_=chunk_name,
        archive_path=f"gs://{chunk_blob.bucket.name}/{chunk_blob.name}",
        feature_matrix_path=f"gs://{feature_blob.bucket.name}/{feature_blob.name}",
        # Remove any errors from previous failed retries which have now succeeded.
        error=firestore.DELETE_FIELD,
    ).merge(db, study_area_name)

    if metadata.elevation_min is not None and metadata.elevation_max is not None:
        metastore.StudyArea.update_min_max_elevation(
            db,
            study_area_name,
            min_=metadata.elevation_min,
            max_=metadata.elevation_max,
        )


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


# storage_client = storage.Client()
# labels_bucket = storage_client.bucket(cloud_storage.LABEL_CHUNKS_BUCKET)
# header = elevation_readers.read_from_esri_ascii(
#     open("/home/waltaskew/studyarea_1/inputs/Domain_DEM.asc"), header_only=True
# ).header
# blob = storage_client.bucket("citycat-output-test").blob(
#     "studyarea_1/R11C1_StudyArea1.zip"
# )
# _write_outputs(
#     labels_bucket,
#     pathlib.PurePosixPath("studyarea_1/R11C1_StudyArea1.zip"),
#     blob.open("rb"),
#     header,
# )
