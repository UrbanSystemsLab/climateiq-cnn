import pathlib
import tempfile

from shapely import geometry

from usl_lib.chunkers import polygon_chunkers
from usl_lib.readers import polygon_readers
from usl_lib.shared import geo_data


def assert_chunk_polygons(
    dir_path,
    y_chunk_index,
    x_chunk_index,
    expected_polygon_masks,
):
    chunk_file = pathlib.Path(dir_path) / f"chunk_{y_chunk_index}_{x_chunk_index}"
    with chunk_file.open("rt") as file:
        stored_polygon_masks = list(polygon_readers.read_polygons_from_text_file(file))
    assert stored_polygon_masks == expected_polygon_masks


def test_write_polygons_to_text_file_no_masks():
    header = geo_data.ElevationHeader(
        col_count=18,
        row_count=15,
        x_ll_corner=0.0,
        y_ll_corner=0.0,
        cell_size=1.0,
        nodata_value=0.0,
    )
    p1 = geometry.Polygon([(1, 1), (2, 1), (2, 2), (1, 1)])
    p2 = geometry.Polygon([(15, 1), (16, 1), (16, 2), (15, 1)])
    p3 = geometry.Polygon([(10, 5), (11, 5), (11, 6), (10, 5)])
    p4 = geometry.Polygon([(5, 15), (15, 15), (15, 16), (5, 15)])
    p5 = geometry.Polygon([(30, 30), (31, 30), (30, 31), (30, 30)])
    with tempfile.TemporaryDirectory() as temp_dir:
        polygon_chunkers.split_polygons_into_chunks(
            header,
            10,
            [(p1, 1), (p2, 2), (p3, 3), (p4, 4), (p5, 5)],
            temp_dir,
        )
        assert_chunk_polygons(temp_dir, 0, 0, [(p3, 1), (p4, 1)])
        assert_chunk_polygons(temp_dir, 0, 1, [(p3, 1), (p4, 1)])
        assert_chunk_polygons(temp_dir, 1, 0, [(p1, 1), (p3, 1)])
        assert_chunk_polygons(temp_dir, 1, 1, [(p2, 1), (p3, 1)])


def test_write_polygons_to_text_file_with_masks():
    header = geo_data.ElevationHeader(
        col_count=18,
        row_count=15,
        x_ll_corner=0.0,
        y_ll_corner=0.0,
        cell_size=1.0,
        nodata_value=0.0,
    )
    p1 = geometry.Polygon([(1, 1), (2, 1), (2, 2), (1, 1)])
    p2 = geometry.Polygon([(15, 1), (16, 1), (16, 2), (15, 1)])
    p3 = geometry.Polygon([(10, 5), (11, 5), (11, 6), (10, 5)])
    p4 = geometry.Polygon([(5, 15), (15, 15), (15, 16), (5, 15)])
    p5 = geometry.Polygon([(30, 30), (31, 30), (30, 31), (30, 30)])
    with tempfile.TemporaryDirectory() as temp_dir:
        polygon_chunkers.split_polygons_into_chunks(
            header,
            10,
            [(p1, 1), (p2, 2), (p3, 3), (p4, 4), (p5, 5)],
            temp_dir,
            support_mask_values=True,
        )
        assert_chunk_polygons(temp_dir, 0, 0, [(p3, 3), (p4, 4)])
        assert_chunk_polygons(temp_dir, 0, 1, [(p3, 3), (p4, 4)])
        assert_chunk_polygons(temp_dir, 1, 0, [(p1, 1), (p3, 3)])
        assert_chunk_polygons(temp_dir, 1, 1, [(p2, 2), (p3, 3)])
