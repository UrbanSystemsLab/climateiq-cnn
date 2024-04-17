from io import StringIO

from shapely import geometry

from usl_lib.writers import polygon_writers


def test_write_polygons_to_text_file_no_masks():
    buffer = StringIO()
    polygon_writers.write_polygons_to_text_file(
        [
            (geometry.Polygon([(0.5, 0.0), (0.5, 1.0), (1.5, 0.0), (0.5, 0.0)]), 1),
            (geometry.Polygon([(5.5, 10.0), (6.5, 10.0), (5.5, 20.0), (5.5, 10.0)]), 1),
        ],
        buffer,
    )
    assert buffer.getvalue() == (
        "2\n"
        + "4 0.5 0.5 1.5 0.5 0.0 1.0 0.0 0.0\n"
        + "4 5.5 6.5 5.5 5.5 10.0 10.0 20.0 10.0\n"
    )


def test_write_polygons_to_text_file_with_masks():
    buffer = StringIO()
    polygon_writers.write_polygons_to_text_file(
        [
            (geometry.Polygon([(0.5, 0.0), (0.5, 1.0), (1.5, 0.0), (0.5, 0.0)]), 7),
            (geometry.Polygon([(5.5, 10.0), (6.5, 10.0), (5.5, 20.0), (5.5, 10.0)]), 8),
        ],
        buffer,
        support_mask_values=True,
    )
    assert buffer.getvalue() == (
        "2\n"
        + "7 4 0.5 0.5 1.5 0.5 0.0 1.0 0.0 0.0\n"
        + "8 4 5.5 6.5 5.5 5.5 10.0 10.0 20.0 10.0\n"
    )
