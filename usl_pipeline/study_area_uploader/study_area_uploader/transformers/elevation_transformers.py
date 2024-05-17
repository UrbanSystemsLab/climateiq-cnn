import math
import pathlib
import typing
from typing import Optional, Tuple

import numpy
from osgeo import gdal
import rasterio
import rasterio.features
from shapely import geometry

from usl_lib.readers import elevation_readers
from usl_lib.shared import geo_data
from usl_lib.transformers import polygon_transformers
from usl_lib.writers import elevation_writers


def crop_geotiff_to_sub_area(
    source_elevation_file_path: str | pathlib.Path,
    target_elevation_file_path: str | pathlib.Path,
    sub_area_bounding_box: geo_data.BoundingBox,
    border_cell_count: int = 1,
) -> None:
    """Crops the elevation data from input file according to sub-area bounding box.

    Args:
        source_elevation_file_path: Path to the input elevation file.
        target_elevation_file_path: Path to the output elevation file that will be
            created.
        sub_area_bounding_box: Sub-area bounding box defined in coordinate system of the
            source elevation data.
        border_cell_count: Number of cells to add on borders of a new area to include to
            output elevation data.
    """
    with pathlib.Path(source_elevation_file_path).open("rb") as input_file:
        header = elevation_readers.read_from_geotiff(
            input_file, header_only=True
        ).header

    # This transformation is backward one mapping raster cells back to original
    # coordinates (original_x = min_x + raster_x * step, y is similar).
    bck_transform = rasterio.Affine(
        header.cell_size,
        0,
        header.x_ll_corner,
        0,
        -header.cell_size,
        header.y_ll_corner + header.row_count * header.cell_size,
    )
    # The forward transformation maps X/Y coordinates to raster column/row.
    fwd_transform = ~bck_transform

    left_col, lower_row = fwd_transform * (
        sub_area_bounding_box.min_x,
        sub_area_bounding_box.min_y,
    )

    right_col, upper_row = fwd_transform * (
        sub_area_bounding_box.max_x,
        sub_area_bounding_box.max_y,
    )

    col_start = max(0, int(left_col) - border_cell_count)
    col_end = min(header.col_count, int(math.ceil(right_col)) + border_cell_count)
    row_start = max(0, int(upper_row) - border_cell_count)
    row_end = min(header.row_count, int(math.ceil(lower_row)) + border_cell_count)

    ds = gdal.Open(str(source_elevation_file_path))
    gdal.Translate(
        str(target_elevation_file_path),
        ds,
        srcWin=[col_start, row_start, col_end - col_start, row_end - row_start],
    )


def transform_geotiff_with_boundaries_to_esri_ascii(
    input_file_path: str | pathlib.Path,
    temp_buffer_file_path: str | pathlib.Path,
    output_file: typing.TextIO,
    band: int = 1,
    no_data_value: Optional[float] = None,
    boundaries_polygons: Optional[list[Tuple[geometry.Polygon, int]]] = None,
    row_buffer_size: int = 500,
) -> None:
    """Transforms elevation raster data from GeoTIFF file stream to Esri ASCII one.

    Transformation optionally includes a stage of zeroing-out values (setting No-data)
    for cells outside the polygon boundaries.

    Args:
        input_file_path: Path to GeoTIFF elevation file to load from.
        temp_buffer_file_path: Path to a place where temporary buffering file will be
            created.
        output_file: Textual stream to write to.
        band: Index of a band that should be loaded from GeoTIFF.
        no_data_value: Optional value to set in the returned data to indicate absence of
            data. If not supplied, the no-data value defined in the TIFF file itself
            will be used.
        boundaries_polygons: Optional polygon boundaries that if defined would be used
            to set No-data for elevation raster outside of boundaries.
        row_buffer_size: Size of buffer in rows that will be used to iterate over
            elevation data so that the whole data is not loaded in memory at once.
    """
    with rasterio.open(input_file_path, driver="GTiff") as src:
        header = elevation_readers.read_header_from_rasterio_dataset_reader(
            src, no_data_value
        )
        elevation_writers.write_header_to_esri_ascii_raster_file(header, output_file)

    global_row_count = header.row_count
    y_chunk_count = (global_row_count + row_buffer_size - 1) // row_buffer_size

    ds = gdal.Open(str(input_file_path))
    for y_chunk_index in range(y_chunk_count):
        row_start = y_chunk_index * row_buffer_size
        row_count = min(global_row_count - row_start, row_buffer_size)
        gdal.Translate(
            str(temp_buffer_file_path),
            ds,
            srcWin=[0, row_start, header.col_count, row_count],
        )
        _transform_geotiff_chunk_data_with_boundaries_to_esri_ascii(
            temp_buffer_file_path,
            output_file,
            band=band,
            no_data_value=no_data_value,
            boundaries_polygons=boundaries_polygons,
        )


def _transform_geotiff_chunk_data_with_boundaries_to_esri_ascii(
    elevation_buffer_file_path: str | pathlib.Path,
    output_file: typing.TextIO,
    band: int = 1,
    no_data_value: Optional[float] = None,
    boundaries_polygons: Optional[list[Tuple[geometry.Polygon, int]]] = None,
):
    """Transforms part of elevation data from GeoTIFF buffer file to Esri ASCII."""
    with open(elevation_buffer_file_path, "rb") as input_file:
        elevation = elevation_readers.read_from_geotiff(
            input_file, band=band, no_data_value=no_data_value
        )
    data = elevation.data

    if boundaries_polygons is not None:
        boundaries_raster = polygon_transformers.rasterize_polygons(
            elevation.header, boundaries_polygons
        )
        data[boundaries_raster == 0] = no_data_value

    numpy.savetxt(output_file, data, delimiter=" ", fmt="%s")
