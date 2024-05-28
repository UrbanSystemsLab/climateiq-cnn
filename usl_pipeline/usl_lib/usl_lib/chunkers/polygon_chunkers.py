import pathlib
from typing import Iterable, Tuple

from shapely import geometry

from usl_lib.chunkers import chunkers_data
from usl_lib.shared import geo_data
from usl_lib.writers import polygon_writers


def _get_chunk_bounding_box(
    elevation_header: geo_data.ElevationHeader,
    chunk_size: int,
    y_chunk_index: int,
    x_chunk_index: int,
) -> geo_data.BoundingBox:
    """Calculates bounding box in coordinate values corresponding to a chunk region.

    Args:
        elevation_header: Information about cell grid projection to coordinate system.
        chunk_size: Size of chunk in cells.
        y_chunk_index: Index of the chunk along the Y-axis
        x_chunk_index: Index of the chunk along the X-axis

    Returns:
        The bounding box of the chunk represented by a tuple (min-x, min-y, max-x,
        max-y).
    """
    global_min_x = elevation_header.x_ll_corner
    cell_size = elevation_header.cell_size
    global_col_count = elevation_header.col_count
    global_row_count = elevation_header.row_count
    global_max_y = elevation_header.y_ll_corner + cell_size * global_row_count

    # X-axis goes from left to right (from smaller cell index to greater one).
    chunk_min_x = global_min_x + cell_size * x_chunk_index * chunk_size
    # Y-axis goes from bottom up (from greater cell index to smaller one).
    chunk_max_y = global_max_y - cell_size * y_chunk_index * chunk_size

    chunk_col_count = min(chunk_size, global_col_count - x_chunk_index * chunk_size)
    chunk_row_count = min(chunk_size, global_row_count - y_chunk_index * chunk_size)
    chunk_max_x = chunk_min_x + chunk_col_count * cell_size
    chunk_min_y = chunk_max_y - chunk_row_count * cell_size
    return geo_data.BoundingBox.from_tuple(
        (chunk_min_x, chunk_min_y, chunk_max_x, chunk_max_y)
    )


def split_polygons_into_chunks(
    elevation_header: geo_data.ElevationHeader,
    chunk_size: int,
    polygon_masks: Iterable[Tuple[geometry.Polygon, int]],
    output_dir_path: str | pathlib.Path,
    chunk_file_name_pattern: str = "chunk_{y}_{x}",
    support_mask_values: bool = False,
    chunk_additional_border_cells: int = 0
) -> list[chunkers_data.ChunkDescriptor]:
    """Writes polygon chunk files based on source with polygons and chunk structure.

    Args:
        elevation_header: Information about cell grid projection to coordinate system.
        chunk_size: Size of each chunk in cells.
        polygon_masks: Source of polygons with associated mask values.
        output_dir_path: Path to the directory where chunk files will be stored.
        chunk_file_name_pattern: Format pattern used to generate chunk file names.
        support_mask_values: Optional indicator that masks should be added into output
            as additional first column.
        chunk_additional_border_cells: Number of cells that the chunk area is extended
            by in all 4 sides (up, down, left, right).

    Returns:
        List of descriptors for produced chunk files.
    """
    global_col_count = elevation_header.col_count
    global_row_count = elevation_header.row_count
    x_chunk_count = (global_col_count + chunk_size - 1) // chunk_size
    y_chunk_count = (global_row_count + chunk_size - 1) // chunk_size

    step = elevation_header.cell_size
    chunk_bboxes: list[list[geo_data.BoundingBox]] = []
    chunk_polygons: list[list[list[Tuple[geometry.Polygon, int]]]] = []
    for y_chunk_index in range(0, y_chunk_count):
        chunk_bboxes.append([])
        chunk_polygons.append([])
        for x_chunk_index in range(0, x_chunk_count):
            bbox = _get_chunk_bounding_box(
                elevation_header, chunk_size, y_chunk_index, x_chunk_index
            )
            chunk_bboxes[y_chunk_index].append(
                geo_data.BoundingBox(
                    bbox.min_x - step - chunk_additional_border_cells,
                    bbox.min_y - step - chunk_additional_border_cells,
                    bbox.max_x + step + chunk_additional_border_cells,
                    bbox.max_y + step + chunk_additional_border_cells,
                )
            )
            chunk_polygons[y_chunk_index].append([])

    for polygon_mask in polygon_masks:
        pol_bbox = geo_data.BoundingBox.from_tuple(polygon_mask[0].bounds)
        for y_chunk_index in range(0, y_chunk_count):
            for x_chunk_index in range(0, x_chunk_count):
                chunk_bbox = chunk_bboxes[y_chunk_index][x_chunk_index]
                if chunk_bbox.intersects(pol_bbox):
                    chunk_polygons[y_chunk_index][x_chunk_index].append(polygon_mask)

    chunk_descriptors: list[chunkers_data.ChunkDescriptor] = []
    for y_chunk_index in range(0, y_chunk_count):
        for x_chunk_index in range(0, x_chunk_count):
            polygons_for_chunk = chunk_polygons[y_chunk_index][x_chunk_index]
            chunk_file_path = pathlib.Path(
                output_dir_path
            ) / chunk_file_name_pattern.format(y=y_chunk_index, x=x_chunk_index)
            with chunk_file_path.open("wt") as output_file:
                polygon_writers.write_polygons_to_text_file(
                    polygons_for_chunk,
                    output_file,
                    support_mask_values=support_mask_values,
                )
            chunk_descriptors.append(
                chunkers_data.ChunkDescriptor(
                    y_chunk_index=y_chunk_index,
                    x_chunk_index=x_chunk_index,
                    path=chunk_file_path,
                )
            )
    return chunk_descriptors
