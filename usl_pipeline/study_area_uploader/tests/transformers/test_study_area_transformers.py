import pathlib
import tempfile
from typing import Tuple

import fiona
import numpy
from numpy import testing
import rasterio
from shapely import geometry

from study_area_uploader.transformers import study_area_transformers
from usl_lib.readers import elevation_readers, polygon_readers
from usl_lib.shared import geo_data


def prepare_elevation_data(
    elevation_file_path: pathlib.Path, data: list[list[float]]
) -> None:
    with rasterio.open(
        str(elevation_file_path),
        "w",
        driver="GTiff",
        dtype=rasterio.float32,
        nodata=0.0,
        width=5,
        height=5,
        count=1,
        crs=rasterio.CRS.from_epsg(32618),
        # The lower-left corner in the following transform is (0, 0) and cell_size is 1.
        transform=rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 5.0),
    ) as raster:
        raster.write(numpy.array(data).astype(rasterio.float32), 1)


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


def test_transform_study_area_files_crop_boundaries_no_soil_classes():
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        tempfile.TemporaryDirectory() as temp_dir2,
    ):
        input_dir = pathlib.Path(temp_dir)
        output_dir = pathlib.Path(temp_dir2)
        elevation_input_file_path = input_dir / "elevation_input.tif"
        prepare_elevation_data(
            elevation_input_file_path,
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0, 24.0],
            ],
        )

        boundaries_input_file_path = input_dir / "boundaries.shp"
        prepare_shape_file(boundaries_input_file_path, [(bbox_polygon(1, 1, 3, 3), 1)])

        buildings_input_file_path = input_dir / "buildings.shp"
        prepare_shape_file(buildings_input_file_path, [(bbox_polygon(2, 1, 4, 3), 1)])

        green_areas_input_file_path = input_dir / "green_areas.shp"
        prepare_shape_file(green_areas_input_file_path, [(bbox_polygon(1, 0, 3, 2), 1)])

        output_file_path = study_area_transformers.transform_study_area_files(
            output_dir,
            elevation_input_file_path,
            sub_area_boundaries_shape_file_path=boundaries_input_file_path,
            buildings_shape_file_path=buildings_input_file_path,
            green_areas_shape_file_path=green_areas_input_file_path,
        )

        assert output_file_path == output_dir / "elevation.tif"

        with output_file_path.open("rb") as input_file:
            elevation = elevation_readers.read_from_geotiff(input_file)
        assert elevation.header == geo_data.ElevationHeader(
            col_count=4,
            row_count=4,
            x_ll_corner=0.0,
            y_ll_corner=0.0,
            cell_size=1.0,
            nodata_value=0.0,
            crs=rasterio.CRS({"init": "EPSG:32618"}),
        )
        testing.assert_array_equal(
            elevation.data,
            [
                [5.0, 6.0, 7.0, 8.0],
                [10.0, 11.0, 12.0, 13.0],
                [15.0, 16.0, 17.0, 18.0],
                [20.0, 21.0, 22.0, 23.0],
            ],
        )

        with open(output_dir / "buildings.txt", "r") as input_file:
            cropped_buildings = list(
                polygon_readers.read_polygons_from_text_file(input_file)
            )
            assert len(cropped_buildings) == 1
            assert cropped_buildings[0][0].equals(bbox_polygon(2, 1, 3, 3))

        with open(output_dir / "green_areas.txt", "r") as input_file:
            cropped_buildings = list(
                polygon_readers.read_polygons_from_text_file(input_file)
            )
            assert len(cropped_buildings) == 1
            assert cropped_buildings[0][0].equals(bbox_polygon(1, 1, 3, 2))


def test_transform_study_area_files_no_crop_boundaries_soil_classes_only():
    input_elevation_data = [
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0, 13.0, 14.0],
        [15.0, 16.0, 17.0, 18.0, 19.0],
        [20.0, 21.0, 22.0, 23.0, 24.0],
    ]

    with (
        tempfile.TemporaryDirectory() as temp_dir,
        tempfile.TemporaryDirectory() as temp_dir2,
    ):
        input_dir = pathlib.Path(temp_dir)
        output_dir = pathlib.Path(temp_dir2)
        elevation_input_file_path = input_dir / "elevation_input.tif"
        prepare_elevation_data(elevation_input_file_path, input_elevation_data)

        soil_classes_input_file_path = input_dir / "soil_classes.shp"
        soil_class_prop = "soil_class"
        prepare_shape_file(
            soil_classes_input_file_path,
            [(bbox_polygon(2, 1, 4, 3), 8), (bbox_polygon(1, 0, 3, 2), 9)],
            soil_class_prop=soil_class_prop,
        )

        output_file_path = study_area_transformers.transform_study_area_files(
            output_dir,
            elevation_input_file_path,
            soil_classes_shape_file_path=soil_classes_input_file_path,
            soil_classes_shape_file_property=soil_class_prop,
        )

        assert output_file_path == elevation_input_file_path

        with output_file_path.open("rb") as input_file:
            elevation = elevation_readers.read_from_geotiff(input_file)
        assert elevation.header == geo_data.ElevationHeader(
            col_count=5,
            row_count=5,
            x_ll_corner=0.0,
            y_ll_corner=0.0,
            cell_size=1.0,
            nodata_value=0.0,
            crs=rasterio.CRS({"init": "EPSG:32618"}),
        )
        testing.assert_array_equal(elevation.data, input_elevation_data)

        with open(output_dir / "soil_classes.txt", "r") as input_file:
            soil_classes = list(
                polygon_readers.read_polygons_from_text_file(input_file)
            )
            assert len(soil_classes) == 2
            assert soil_classes[0][0].equals(bbox_polygon(2, 1, 4, 3))
            assert soil_classes[0][1] == 8
            assert soil_classes[1][0].equals(bbox_polygon(1, 0, 3, 2))
            assert soil_classes[1][1] == 9
