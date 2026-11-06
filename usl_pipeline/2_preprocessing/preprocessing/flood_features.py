"""Builds flood feature matrices from study area chunks."""

import io
import logging
import pathlib
import tarfile
import time
from typing import BinaryIO, IO, Tuple
from google.cloud import firestore
from google.cloud import storage
import functions_framework
from numpy.typing import NDArray
import rasterio
from shapely import geometry
from usl_lib.readers import elevation_readers, polygon_readers
from usl_lib.shared import geo_data
from usl_lib.shared.function_runtime import (
    retry_and_report_errors as _retry_and_report_errors,
)
from usl_lib.storage import cloud_storage
from usl_lib.storage import file_names
from usl_lib.storage import metastore
from usl_lib.transformers import feature_raster_transformers
from preprocessing.chunks import (
    _parse_chunk_path,
    _parse_spatial_chunk_indices_from_name,
    _write_as_npy,
    _write_chunk_metastore_error,
)
from preprocessing.feature_metadata import FeatureMetadata


@functions_framework.cloud_event
@_retry_and_report_errors(
    lambda cloud_event, exc: _write_chunk_metastore_error(
        cloud_event.data["name"], str(exc)
    )
)
def build_flood_feature_matrix(cloud_event: functions_framework.CloudEvent) -> None:
    """Builds a flood feature matrix when a chunk archive is uploaded."""
    chunk_path = cloud_event.data["name"]
    if not chunk_path.endswith(".tar"):
        logging.debug("Skipping non-flood file %s", chunk_path)
        return

    _build_flood_feature_matrix(
        cloud_event.data["bucket"],
        chunk_path,
        cloud_storage.FEATURE_CHUNKS_BUCKET,
    )


def _build_flood_feature_matrix(
    bucket_name: str, chunk_path: str, output_bucket: str
) -> None:
    """Builds a flood (CityCat) feature matrix from an uploaded chunk archive."""
    storage_client = storage.Client()
    chunk_blob = storage_client.bucket(bucket_name).blob(chunk_path)

    with chunk_blob.open("rb") as chunk:
        study_area_name, chunk_name = _parse_chunk_path(chunk_path)
        feature_file_name = pathlib.PurePosixPath(chunk_path).with_suffix(".npy")
        feature_blob = storage_client.bucket(output_bucket).blob(str(feature_file_name))
        chunk_metadata = metastore.StudyAreaSpatialChunk.get_if_exists(
            firestore.Client(), study_area_name, chunk_name
        )
        # Let's check if the chunk metadata object is present and has the state
        # different from None (which means it's either FEATURE_MATRIX_PROCESSING
        # meaning that unscaled feature matrix is stored and this CF succeeded, or
        # it's FEATURE_MATRIX_READY and the downstream rescaling CF is also done).
        # If metadata object is not present it means we're in the first execution
        # attempt. None state means that we're in the retry and either this CF
        # crashed during previous execution attempt or it finished with an error.
        if chunk_metadata is not None and chunk_metadata.state is not None:
            logging.info(
                "Flood feature matrix for chunk %s was already generated",
                chunk_path,
            )
            return

        start_time = time.time()
        logging.info("Start generating flood feature matrix for chunk %s", chunk_path)
        metastore.StudyAreaSpatialChunk(
            id_=chunk_name,
            error=firestore.DELETE_FIELD,
        ).merge(firestore.Client(), study_area_name)
        feature_matrix, metadata, header = _build_flood_feature_matrix_from_archive(
            chunk
        )

        if feature_matrix is None or header is None:
            raise ValueError(f"Empty archive found in {chunk_blob}")

        # Updating min/max elevation in the study area metadata first before storing
        # feature matrix file that will trigger rescaling post-processing.
        _update_study_area_metastore_entry(chunk_blob, metadata)
        _write_as_npy(feature_blob, feature_matrix)
        logging.info(
            "Flood feature matrix file was generated for chunk %s in %s seconds",
            chunk_path,
            time.time() - start_time,
        )
        _write_flood_chunk_metastore_entry(chunk_blob, header)


def _read_polygons_from_byte_stream(
    stream: IO[bytes],
) -> list[Tuple[geometry.Polygon, int]]:
    """Reads polygons from textual file content provided in form of a byte stream."""
    return list(polygon_readers.read_polygons_from_text_file(io.TextIOWrapper(stream)))


def _build_flood_feature_matrix_from_archive(
    archive: BinaryIO,
) -> Tuple[NDArray | None, FeatureMetadata, geo_data.ElevationHeader | None]:
    """Builds a feature matrix for the given archive.

    Args:
      archive: The file containing the archive.

    Returns:
      A tuple of (array, metadata, elevation-header) for the feature matrix, metadata
      describing the feature matrix and elevation header describing coordinate grid. In
      case there are no files related to study area in the archive optional items in the
      returning tuple will be None.
    """
    metadata: FeatureMetadata = FeatureMetadata()
    elevation: geo_data.Elevation | None = None
    boundaries: list[Tuple[geometry.Polygon, int]] | None = None
    buildings: list[Tuple[geometry.Polygon, int]] | None = None
    green_areas: list[Tuple[geometry.Polygon, int]] | None = None
    soil_classes: list[Tuple[geometry.Polygon, int]] | None = None
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
        if (
            elevation is None
            or buildings is None
            or green_areas is None
            or soil_classes is None
        ):
            raise ValueError(
                f"Some flood simulation data missing (see tar list: {files_in_tar})"
            )
        feature_matrix = feature_raster_transformers.transform_to_feature_raster_layers(
            elevation,
            boundaries,
            buildings,
            green_areas,
            soil_classes,
            geo_data.DEFAULT_INFILTRATION_CONFIGURATION,
        )

        slope_layer = feature_matrix[:, :, 8]
        nodata = elevation.header.nodata_value
        valid_slope = slope_layer[slope_layer != nodata]
        if valid_slope.size > 0:
            metadata.slope_min = float(valid_slope.min())
            metadata.slope_max = float(valid_slope.max())
        else:
            logging.warning(
                "No valid slope values found for study area (all cells nodata); "
                "leaving slope_min/slope_max unset rather than using the nodata "
                "sentinel, which would otherwise be treated as a real bound by "
                "rescale_feature_matrix and never be displaceable by a genuine "
                "value once persisted."
            )

        return feature_matrix, metadata, elevation.header

    return None, FeatureMetadata(), None


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
    if metadata.slope_min is not None and metadata.slope_max is not None:
        metastore.StudyArea.update_min_max_slope(
            db,
            study_area_name,
            min_=metadata.slope_min,
            max_=metadata.slope_max,
        )
    if metadata.chunk_size is not None:
        metastore.StudyArea.update_chunk_info(db, study_area_name, metadata.chunk_size)

        # Let's update study area state to CHUNKS_UPLOADED if it's still INIT and all
        # the chunks are present in metadata (in any state) meaning that every chunk
        # TAR-file was placed in the chunk bucket and did trigger processing event.
        study_area_metadata = metastore.StudyArea.get(db, study_area_name)
        if (
            study_area_metadata.state == metastore.StudyAreaState.INIT
            and study_area_metadata.chunk_x_count is not None
            and study_area_metadata.chunk_y_count is not None
        ):
            expected_chunk_count = (
                study_area_metadata.chunk_x_count * study_area_metadata.chunk_y_count
            )
            current_chunk_count = len(
                metastore.StudyArea.list_all_chunk_refs(db, study_area_name)
            )
            if current_chunk_count == expected_chunk_count:
                # All the chunk metadata objects were registered, let's update the state
                # of the study area.
                metastore.StudyArea.update_state(
                    db, study_area_name, metastore.StudyAreaState.CHUNKS_UPLOADED
                )


def _write_flood_chunk_metastore_entry(
    chunk_blob: storage.Blob,
    header: geo_data.ElevationHeader,
) -> None:
    """Updates the metastore with new information for the given flood chunks.

    Writes a Firestore entry for the new feature matrix chunk.

    Args:
        chunk_blob: The GCS blob containing the raw chunk archive.
        header: The elevation header describing coordinate grid of the chunk area.
    """
    db = firestore.Client()
    study_area_name, chunk_name = _parse_chunk_path(chunk_blob.name)
    # Match against the file name only. The regex is unanchored, so passing the
    # full path would let a study area name containing "<digits>_<digits>" supply
    # the indices instead of the chunk itself.
    x_index, y_index = _parse_spatial_chunk_indices_from_name(
        pathlib.PurePosixPath(chunk_blob.name).name
    )

    metastore.StudyAreaSpatialChunk(
        id_=chunk_name,
        state=metastore.StudyAreaChunkState.FEATURE_MATRIX_PROCESSING,
        raw_path=f"gs://{chunk_blob.bucket.name}/{chunk_blob.name}",
        needs_scaling=True,
        x_index=x_index,
        y_index=y_index,
        col_count=header.col_count,
        row_count=header.row_count,
        x_ll_corner=header.x_ll_corner,
        y_ll_corner=header.y_ll_corner,
        # Remove any errors from previous failed retries which have now succeeded.
        error=firestore.DELETE_FIELD,
    ).merge(db, study_area_name)
