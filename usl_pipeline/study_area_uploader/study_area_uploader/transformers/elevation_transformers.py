import math
import pathlib

from osgeo import gdal
import rasterio

from usl_lib.readers import elevation_readers
from usl_lib.shared import geo_data


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
