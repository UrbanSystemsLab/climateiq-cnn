import io

from shapely import geometry

from usl_lib.readers import polygon_readers


def test_read_polygons_from_text_file_no_masks():
    lines = [
        "2",
        "4 0.1 0.2 0.2 0.1 1.1 1.1 1.2 1.1  # unused part",
        "5 1.0 2.0 2.0 1.0 1.0 10.0 10.0 20.0 20.0 10.0",
        "# Unused line",
    ]
    reader = io.BufferedReader(io.BytesIO("\n".join(lines).encode("utf-8")))
    with io.TextIOWrapper(reader) as file:
        polygon_masks = polygon_readers.read_polygons_from_text_file(file)
        assert polygon_masks == [
            (geometry.Polygon([(0.1, 1.1), (0.2, 1.1), (0.2, 1.2), (0.1, 1.1)]), 1),
            (geometry.Polygon([(1, 10), (2, 10), (2, 20), (1, 20), (1, 10)]), 1),
        ]


def test_read_polygons_from_text_file_with_masks():
    lines = [
        "2",
        "8 4 0.1 0.2 0.2 0.1 1.1 1.1 1.2 1.1  # unused part",
        "9 5 1.0 2.0 2.0 1.0 1.0 10.0 10.0 20.0 20.0 10.0",
        "# Unused line",
    ]
    reader = io.BufferedReader(io.BytesIO("\n".join(lines).encode("utf-8")))
    with io.TextIOWrapper(reader) as file:
        polygon_masks = polygon_readers.read_polygons_from_text_file(
            file, support_mask_values=True
        )
        assert polygon_masks == [
            (geometry.Polygon([(0.1, 1.1), (0.2, 1.1), (0.2, 1.2), (0.1, 1.1)]), 8),
            (geometry.Polygon([(1, 10), (2, 10), (2, 20), (1, 20), (1, 10)]), 9),
        ]
