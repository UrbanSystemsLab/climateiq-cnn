import io
import logging
import pathlib
import tarfile
from typing import Optional, Tuple

from google.cloud import storage
from shapely import geometry

from study_area_uploader.chunkers import elevation_chunkers
from study_area_uploader.transformers import study_area_transformers
from usl_lib.chunkers import polygon_chunkers
from usl_lib.readers import elevation_readers
from usl_lib.shared import geo_data
from usl_lib.storage import file_names
from usl_lib.writers import elevation_writers


def _chunk_polygons_if_present(
    elevation_header: geo_data.ElevationHeader,
    chunk_size: int,
    polygon_masks: Optional[list[Tuple[geometry.Polygon, int]]],
    work_dir: pathlib.Path,
    object_type_name: str,
    support_mask_values: bool = False,
    chunk_additional_border_cells: int = 0,
) -> Optional[pathlib.Path]:
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
    chunks_dir: Optional[pathlib.Path],
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
    chunk_size: int = 1000,
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
        chunk_file_name = f"chunk_{y_chunk_index}_{x_chunk_index}"
        logging.info("Exporting %s...", chunk_file_name)
        elevation_chunk_file_path = elevation_chunk_descriptor.path
        # Let's standardize the band number (to 1) and the NODATA_value (to default one)
        with open(elevation_chunk_file_path, "rb") as input_file:
            chunk_elevation = elevation_readers.read_from_geotiff(
                input_file,
                band=input_elevation_band,
                no_data_value=default_nodata_value,
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
