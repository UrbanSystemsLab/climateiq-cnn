import io
import logging
import pathlib
import shutil
import tarfile
from typing import Tuple

from google.cloud import storage
import numpy
from shapely import geometry

from study_area_uploader.chunkers import elevation_chunkers
from study_area_uploader.transformers import study_area_transformers
from usl_lib.chunkers import polygon_chunkers
from usl_lib.readers import elevation_readers
from usl_lib.shared import geo_data
from usl_lib.storage import file_names
from usl_lib.transformers import soil_classes_transformers
from usl_lib.writers import elevation_writers
import numpy as np


def _chunk_polygons_if_present(
    elevation_header: geo_data.ElevationHeader,
    chunk_size: int,
    polygon_masks: list[Tuple[geometry.Polygon, int]] | None,
    work_dir: pathlib.Path,
    object_type_name: str,
    support_mask_values: bool = False,
    chunk_additional_border_cells: int = 0,
) -> pathlib.Path | None:
    """Splits optional list of polygons with masks (if present) into chunk files."""
    logging.info("Preparing %s chunks...", object_type_name)
    chunks_dir = None
    if polygon_masks is not None:
        chunks_dir = work_dir / f"{object_type_name}_chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        polygon_chunkers.split_polygons_into_chunks(
            elevation_header,
            chunk_size,
            polygon_masks,
            chunks_dir,
            support_mask_values=support_mask_values,
            chunk_additional_border_cells=chunk_additional_border_cells,
        )
    return chunks_dir


def _add_chunk_to_tar_if_present(
    chunks_dir: pathlib.Path | None,
    chunk_file_name: str,
    tar_entry_file_name: str,
    tar: tarfile.TarFile,
):
    """Copy new entry to TAR-file from a chunk file if the folder with chunks exists."""
    if chunks_dir is not None:
        tar.add(str(chunks_dir / chunk_file_name), arcname=tar_entry_file_name)


def build_and_upload_chunks(
    study_area_name: str,
    input_data: study_area_transformers.PreparedInputData,
    work_dir: pathlib.Path,
    study_area_chunk_bucket: storage.Bucket,
    chunk_size: int,
    extend_soil_classes_chunk_border: bool = True,
    input_elevation_band: int = 1,
    default_nodata_value: float = -9999.0,
) -> None:
    """Builds chunks of input data groups them into TAR-files and uploads to a bucket.

    Args:
        study_area_name: Name of the study area that is used as a folder name when
            storing files to storage bucket.
        input_data: Input data for chunking.
        work_dir: Folder where intermediate temporary files can be created.
        study_area_chunk_bucket: Storage bucket to store files to.
        chunk_size: Size of chunks in cells (over both axes).
        extend_soil_classes_chunk_border: Indicates that chunk areas for soil classes
            should be extended by chunk_size in all directions around each chunk.
        input_elevation_band: Band number in input GeoTIFF elevation file that elevation
            data should be read from.
        default_nodata_value: NODATA value that should be used to store output elevation
            chunks.
    """
    logging.info("Preparing elevation chunks...")
    elevation_chunks_dir = work_dir / "elevation_chunks"
    elevation_chunks_dir.mkdir(parents=True, exist_ok=True)
    elevation_chunk_descriptors = elevation_chunkers.split_geotiff_into_chunks(
        input_data.elevation_file_path, chunk_size, elevation_chunks_dir
    )
    with open(input_data.elevation_file_path, "rb") as input_fd:
        header = elevation_readers.read_from_geotiff(input_fd, header_only=True).header

    # Chunking all present polygon layers
    boundaries_chunks_dir = _chunk_polygons_if_present(
        header, chunk_size, input_data.boundaries_polygons, work_dir, "boundaries"
    )
    buildings_chunks_dir = _chunk_polygons_if_present(
        header, chunk_size, input_data.buildings_polygons, work_dir, "buildings"
    )
    green_areas_chunks_dir = _chunk_polygons_if_present(
        header, chunk_size, input_data.green_areas_polygons, work_dir, "green_areas"
    )
    soil_classes_chunk_additional_border_cells = (
        chunk_size if extend_soil_classes_chunk_border else 0
    )
    soil_classes_chunks_dir = _chunk_polygons_if_present(
        header,
        chunk_size,
        input_data.soil_classes_polygons,
        work_dir,
        "soil_classes",
        support_mask_values=True,
        chunk_additional_border_cells=soil_classes_chunk_additional_border_cells,
    )

    for elevation_chunk_descriptor in elevation_chunk_descriptors:
        y_chunk_index = elevation_chunk_descriptor.y_chunk_index
        x_chunk_index = elevation_chunk_descriptor.x_chunk_index
        chunk_file_name = f"chunk_{x_chunk_index}_{y_chunk_index}"
        logging.info("Exporting %s...", chunk_file_name)
        elevation_chunk_file_path = elevation_chunk_descriptor.path
        # Let's standardize the band number (to 1) and the NODATA_value (to default one)
        with open(elevation_chunk_file_path, "rb") as input_file:
            chunk_elevation = elevation_readers.read_from_geotiff(
                input_file,
                band=input_elevation_band,
                no_data_value=default_nodata_value,
            )
        # Padding the chunk size to guarantee chunk_size defined by a caller for both
        # X and Y axis (padding is done by filling in NODATA values).
        if (
            chunk_elevation.header.row_count < chunk_size
            or chunk_elevation.header.col_count < chunk_size
        ):
            y_pad = chunk_size - chunk_elevation.header.row_count
            x_pad = chunk_size - chunk_elevation.header.col_count
            chunk_elevation.data = numpy.pad(
                chunk_elevation.data,
                numpy.array(((0, y_pad), (0, x_pad))),
                mode="constant",
                constant_values=chunk_elevation.header.nodata_value,
            )
            chunk_elevation.header.row_count = chunk_size
            chunk_elevation.header.col_count = chunk_size
            chunk_elevation.header.y_ll_corner = (
                chunk_elevation.header.y_ll_corner
                - y_pad * chunk_elevation.header.cell_size
            )

        elevation_writers.write_to_geotiff(chunk_elevation, elevation_chunk_file_path)

        tar_fd = io.BytesIO()
        with tarfile.open(mode="w", fileobj=tar_fd) as tar:
            tar.add(str(elevation_chunk_file_path), arcname=file_names.ELEVATION_TIF)
            _add_chunk_to_tar_if_present(
                boundaries_chunks_dir,
                chunk_file_name,
                file_names.BOUNDARIES_TXT,
                tar,
            )
            _add_chunk_to_tar_if_present(
                buildings_chunks_dir,
                chunk_file_name,
                file_names.BUILDINGS_TXT,
                tar,
            )
            _add_chunk_to_tar_if_present(
                green_areas_chunks_dir,
                chunk_file_name,
                file_names.GREEN_AREAS_TXT,
                tar,
            )
            _add_chunk_to_tar_if_present(
                soil_classes_chunks_dir,
                chunk_file_name,
                file_names.SOIL_CLASSES_TXT,
                tar,
            )
        tar_fd.flush()
        tar_fd.seek(0)
        study_area_chunk_bucket.blob(
            f"{study_area_name}/{chunk_file_name}.tar"
        ).upload_from_file(tar_fd)


def _add_chunk_to_dir_if_present(
    chunks_dir: pathlib.Path | None,
    chunk_dir_name: str,
    file_name: str,
    destination_path: pathlib.Path,
    study_area_name: str,
    study_area_chunk_bucket: storage.Bucket,
):
    """Copy new entry to folder."""
    if chunks_dir is not None:
        shutil.copy(
            str(chunks_dir / chunk_dir_name), destination_path.absolute().as_posix()
        )
        blob_name = f"{study_area_name}/{chunk_dir_name}/{file_name}"
        blob = study_area_chunk_bucket.blob(blob_name)
        blob.upload_from_filename(str(chunks_dir / chunk_dir_name))


def check_threshold(chunk_elevation_data, chunk_size, nodata_value):
    count = np.count_nonzero(chunk_elevation_data == nodata_value)
    total_count = chunk_size**chunk_elevation_data.ndim
    return count / total_count


def build_and_upload_chunks_citycat(
    study_area_name: str,
    input_data: study_area_transformers.PreparedInputData,
    work_dir: pathlib.Path,
    study_area_chunk_bucket: storage.Bucket,
    chunk_size: int,
    extend_soil_classes_chunk_border: bool = True,
    input_elevation_band: int = 1,
    default_nodata_value: float = -9999.0,
) -> None:
    """Builds chunks of input data groups them into TAR-files and uploads to a bucket.

    Args:
        study_area_name: Name of the study area that is used as a folder name when
            storing files to storage bucket.
        input_data: Input data for chunking.
        work_dir: Folder where intermediate temporary files can be created.
        study_area_chunk_bucket: Storage bucket to store files to.
        chunk_size: Size of chunks in cells (over both axes).
        extend_soil_classes_chunk_border: Indicates that chunk areas for soil classes
            should be extended by chunk_size in all directions around each chunk.
        input_elevation_band: Band number in input GeoTIFF elevation file that elevation
            data should be read from.
        default_nodata_value: NODATA value that should be used to store output elevation
            chunks.
    """
    logging.info("Preparing elevation chunks...")
    elevation_chunks_dir = work_dir / "elevation_chunks"
    elevation_chunks_dir.mkdir(parents=True, exist_ok=True)
    elevation_chunk_descriptors = elevation_chunkers.split_geotiff_into_chunks(
        input_data.elevation_file_path, chunk_size, elevation_chunks_dir
    )
    chunked_output_dir = work_dir / "chunked_output"
    chunked_output_dir.mkdir(parents=True, exist_ok=True)
    with open(input_data.elevation_file_path, "rb") as input_fd:
        header = elevation_readers.read_from_geotiff(input_fd, header_only=True).header

    # Chunking all present polygon layers
    boundaries_chunks_dir = _chunk_polygons_if_present(
        header, chunk_size, input_data.boundaries_polygons, work_dir, "boundaries"
    )
    buildings_chunks_dir = _chunk_polygons_if_present(
        header, chunk_size, input_data.buildings_polygons, work_dir, "buildings"
    )
    green_areas_chunks_dir = _chunk_polygons_if_present(
        header, chunk_size, input_data.green_areas_polygons, work_dir, "green_areas"
    )
    soil_classes_chunk_additional_border_cells = (
        chunk_size if extend_soil_classes_chunk_border else 0
    )
    soil_classes_chunks_dir = _chunk_polygons_if_present(
        header,
        chunk_size,
        input_data.soil_classes_polygons,
        work_dir,
        "soil_classes",
        support_mask_values=True,
        chunk_additional_border_cells=soil_classes_chunk_additional_border_cells,
    )

    green_areas_polygons_transformed = set(
        soil_classes_transformers.transform_soil_classes_as_green_areas(
            header,
            input_data.green_areas_polygons,
            input_data.soil_classes_polygons,
            non_green_area_soil_classes={
                study_area_transformers.DEFAULT_NON_GREEN_AREA_SOIL_CLASS
            },
        )
    )
    green_areas_transformed_chunks_dir = _chunk_polygons_if_present(
        header,
        chunk_size,
        list(green_areas_polygons_transformed),
        work_dir,
        "green_areas_transformed",
        True,
    )

    for elevation_chunk_descriptor in elevation_chunk_descriptors:
        y_chunk_index = elevation_chunk_descriptor.y_chunk_index
        x_chunk_index = elevation_chunk_descriptor.x_chunk_index
        chunk_file_name = f"chunk_{x_chunk_index}_{y_chunk_index}"
        logging.info("Exporting %s...", chunk_file_name)
        elevation_chunk_file_path = elevation_chunk_descriptor.path
        # Let's standardize the band number (to 1) and the NODATA_value (to default one)
        with open(elevation_chunk_file_path, "rb") as input_file:
            chunk_elevation = elevation_readers.read_from_geotiff(
                input_file,
                band=input_elevation_band,
                no_data_value=default_nodata_value,
            )
        null_ratio = check_threshold(
            chunk_elevation.data, chunk_size, default_nodata_value
        )
        # Padding the chunk size to guarantee chunk_size defined by a caller for both
        # X and Y axis (padding is done by filling in NODATA values).
        if (
            chunk_elevation.header.row_count < chunk_size
            or chunk_elevation.header.col_count < chunk_size
        ):
            y_pad = chunk_size - chunk_elevation.header.row_count
            x_pad = chunk_size - chunk_elevation.header.col_count
            chunk_elevation.data = numpy.pad(
                chunk_elevation.data,
                numpy.array(((0, y_pad), (0, x_pad))),
                mode="constant",
                constant_values=chunk_elevation.header.nodata_value,
            )
            chunk_elevation.header.row_count = chunk_size
            chunk_elevation.header.col_count = chunk_size
            chunk_elevation.header.y_ll_corner = (
                chunk_elevation.header.y_ll_corner
                - y_pad * chunk_elevation.header.cell_size
            )

        if null_ratio < 0.9:
            elevation_asc_file_path = elevation_chunks_dir.joinpath(chunk_file_name)
            with open(elevation_asc_file_path, "wt") as output_file:
                elevation_writers.write_to_esri_ascii_raster_file(
                    chunk_elevation, output_file
                )
            chunk_output_dir = chunked_output_dir / chunk_file_name
            chunk_output_dir.mkdir(parents=True, exist_ok=True)

            _add_chunk_to_dir_if_present(
                elevation_chunks_dir,
                chunk_file_name,
                file_names.CITYCAT_ELEVATION_ASC,
                chunk_output_dir.joinpath(file_names.CITYCAT_ELEVATION_ASC),
                study_area_name,
                study_area_chunk_bucket,
            )
            _add_chunk_to_dir_if_present(
                boundaries_chunks_dir,
                chunk_file_name,
                file_names.BOUNDARIES_TXT,
                chunk_output_dir.joinpath(file_names.BOUNDARIES_TXT),
                study_area_name,
                study_area_chunk_bucket,
            )
            _add_chunk_to_dir_if_present(
                buildings_chunks_dir,
                chunk_file_name,
                file_names.BUILDINGS_TXT,
                chunk_output_dir.joinpath(file_names.BUILDINGS_TXT),
                study_area_name,
                study_area_chunk_bucket,
            )
            # _add_chunk_to_dir_if_present(
            #     green_areas_chunks_dir,
            #     chunk_file_name,
            #     file_names.GREEN_AREAS_TXT,
            #     chunk_output_dir.joinpath(file_names.GREEN_AREAS_TXT),
            #     study_area_name,
            #     study_area_chunk_bucket,
            # )
            # _add_chunk_to_dir_if_present(
            #     soil_classes_chunks_dir,
            #     chunk_file_name,
            #     file_names.SOIL_CLASSES_TXT,
            #     chunk_output_dir.joinpath(file_names.SOIL_CLASSES_TXT),
            #     study_area_name,
            #     study_area_chunk_bucket,
            # )
            green_areas_file_name = (
                file_names.CITYCAT_SPATIAL_GREEN_AREAS_TXT
                if soil_classes_chunks_dir
                else file_names.GREEN_AREAS_TXT
            )

            # Add the appropriate green areas file
            _add_chunk_to_dir_if_present(
                green_areas_chunks_dir,
                chunk_file_name,
                green_areas_file_name,
                chunk_output_dir.joinpath(green_areas_file_name),
                study_area_name,
                study_area_chunk_bucket,
            )
