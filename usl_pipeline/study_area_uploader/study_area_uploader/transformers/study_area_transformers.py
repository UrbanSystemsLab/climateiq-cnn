import os
import pathlib
import typing
from typing import Optional

from osgeo import gdal

from study_area_uploader.transformers import elevation_transformers
from study_area_uploader.readers import polygon_readers as shape_readers
from usl_lib.readers import elevation_readers, polygon_readers
from usl_lib.shared import geo_data
from usl_lib.transformers import polygon_transformers
from usl_lib.writers import polygon_writers


def transform_shape_file(
    input_shape_file_path: pathlib.Path | str,
    output_polygon_file_path: pathlib.Path | str,
    sub_area_bbox: Optional[geo_data.BoundingBox],
    target_crs: str,
    mask_value_feature_property: Optional[str] = None,
):
    polygons = shape_readers.read_polygons_from_shape_file(
        input_shape_file_path, target_crs=target_crs,
        mask_value_feature_property=mask_value_feature_property,
    )
    filtered_polygons = polygons if sub_area_bbox is None else list(
        polygon_transformers.crop_polygons_to_sub_area(polygons, sub_area_bbox)
    )
    polygon_writers.write_polygons_to_text_file(
        filtered_polygons, output_polygon_file,
        support_mask_values=(mask_value_feature_property is not None),
    )


def transform_study_area_files(
    elevation_file_path: pathlib.Path | str,
    sub_area_boundaries_shape_file_path: Optional[pathlib.Path | str],
    buildings_shape_file_path: Optional[pathlib.Path | str],
    green_areas_shape_file_path: Optional[pathlib.Path | str],
    soil_classes_shape_file_path: Optional[pathlib.Path | str],
    soil_classes_shape_file_property: Optional[str],
    output_dir_path: pathlib.Path | str,
    elevation_output_file_name="elevation.tif",
    boundaries_output_file_name="boundaries.txt",
    buildings_output_file_name="buildings.txt",
    green_areas_output_file_name="green_areas.txt",
    soil_classes_output_file_name="soil_classes.txt",
) -> pathlib.Path:
    with pathlib.Path(elevation_file_path).open("rb") as input_file:
        elevation_header = elevation_readers.read_from_geotiff(
            input_file, header_only=True
        ).header
    crs = elevation_header.crs.to_string()
    print(f"Elevation data CRS: {crs}")

    output_dir = pathlib.Path(output_dir_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sub_area_bbox: Optional[geo_data.BoundingBox] = None
    if sub_area_boundaries_shape_file_path is not None:
        print("Preparing Boundaries...")
        boundaries_polygons = shape_readers.read_polygons_from_shape_file(
            sub_area_boundaries_shape_file_path, target_crs=crs
        )
        boundaries_file_path = output_dir / boundaries_output_file_name
        with boundaries_file_path.open("w") as output_file:
            polygon_writers.write_polygons_to_text_file(
                boundaries_polygons, output_file
            )
        sub_area_bbox = polygon_transformers.get_bounding_box_for_boundaries(
            p[0] for p in boundaries_polygons
        )

    if sub_area_bbox is not None:
        print("Preparing Elevation data...")
        output_elevation_file = output_dir / elevation_output_file_name
        elevation_transformers.crop_geotiff_to_sub_area(
            elevation_file_path, output_elevation_file, sub_area_bbox
        )
    else:
        output_elevation_file = pathlib.Path(elevation_file_path)

    if buildings_shape_file_path is not None:
        print("Preparing Buildings...")
        sub_area_buildings_file = output_dir / buildings_output_file_name
        transform_shape_file(
            buildings_shape_file_path, sub_area_buildings_file, sub_area_bbox,
            target_crs=crs
        )

    if green_areas_shape_file_path is not None:
        print("Preparing Green Areas...")
        sub_area_green_areas_file = output_dir / green_areas_output_file_name
        transform_shape_file(
            green_areas_shape_file_path, sub_area_green_areas_file, sub_area_bbox,
            target_crs=crs
        )

    if soil_classes_shape_file_path is not None:
        print("Preparing Soil Classes...")
        sub_area_soil_classes_file = output_dir / soil_classes_output_file_name
        transform_shape_file(
            soil_classes_shape_file_path, sub_area_soil_classes_file, sub_area_bbox,
            target_crs=crs,
            mask_value_feature_property=soil_classes_shape_file_property,
        )

    return output_elevation_file
