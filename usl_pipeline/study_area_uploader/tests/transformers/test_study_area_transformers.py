import pathlib
import tempfile
from typing import Tuple

import fiona
from shapely import geometry

from study_area_uploader.transformers import study_area_transformers
from usl_lib.shared import geo_data


def polygon_points(polygon: geometry.Polygon) -> list[Tuple[float, float]]:
    xx, yy = polygon.exterior.coords.xy
    return list(zip(xx, yy))


def prepare_shape_file(
    file_path: pathlib.Path,
    polygon_masks: list[Tuple[geometry.Polygon, int]],
    soil_class_prop: str = "unused_one",
) -> None:
    crs = "EPSG:32618"
    schema = {"geometry": "Polygon", "properties": {soil_class_prop: "int"}}
    with fiona.open(
        file_path,
        "w",
        crs=crs,
        driver="ESRI Shapefile",
        schema=schema,
    ) as output:
        for polygon_mask in polygon_masks:
            output.write(
                {
                    "geometry": {
                        "coordinates": [polygon_points(polygon_mask[0])],
                        "type": "Polygon",
                    },
                    "id": "1",
                    "properties": {soil_class_prop: polygon_mask[1]},
                    "type": "Feature",
                }
            )


def bbox_polygon(x1: float, y1: float, x2: float, y2: float) -> geometry.Polygon:
    return geometry.Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)])


def test_transform_shape_file_no_polygon_masks():
    with (tempfile.TemporaryDirectory() as temp_dir,):
        input_dir = pathlib.Path(temp_dir)
        buildings_input_file_path = input_dir / "buildings.shp"
        prepare_shape_file(
            buildings_input_file_path,
            [
                (bbox_polygon(2, 1, 4, 3), 1),
            ],
        )

        cropped_buildings = list(
            study_area_transformers.transform_shape_file(
                buildings_input_file_path,
                geo_data.BoundingBox(min_x=1, min_y=1, max_x=3, max_y=3),
                "EPSG:32618",
            )
        )

        assert len(cropped_buildings) == 1
        assert cropped_buildings[0][0].equals(bbox_polygon(2, 1, 3, 3))


def test_transform_shape_file_with_polygon_masks():
    with (tempfile.TemporaryDirectory() as temp_dir,):
        input_dir = pathlib.Path(temp_dir)
        soil_classes_input_file_path = input_dir / "soil_classes.shp"
        soil_class_prop = "soil_class"
        prepare_shape_file(
            soil_classes_input_file_path,
            [
                (bbox_polygon(0, 0, 5, 5), 0),  # It has 0 mask and will be skipped
                (bbox_polygon(2, 1, 4, 3), 8),
                (bbox_polygon(1, 0, 3, 2), 9),
            ],
            soil_class_prop=soil_class_prop,
        )

        soil_classes = list(
            study_area_transformers.transform_shape_file(
                soil_classes_input_file_path,
                None,
                "EPSG:32618",
                mask_value_feature_property=soil_class_prop,
            )
        )

        assert len(soil_classes) == 2
        assert soil_classes[0][0].equals(bbox_polygon(2, 1, 4, 3))
        assert soil_classes[0][1] == 8
        assert soil_classes[1][0].equals(bbox_polygon(1, 0, 3, 2))
        assert soil_classes[1][1] == 9
