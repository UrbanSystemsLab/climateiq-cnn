import os
import pathlib
from typing import Optional

from study_area_uploader.transformers import elevation_transformers
from study_area_uploader.readers import polygon_readers as shape_readers
from usl_lib.readers import elevation_readers
from usl_lib.shared import geo_data
from usl_lib.storage import file_names
from usl_lib.transformers import polygon_transformers
from usl_lib.writers import polygon_writers


def _transform_shape_file(
    input_shape_file_path: pathlib.Path | str,
    output_polygon_file_path: pathlib.Path | str,
    sub_area_bbox: Optional[geo_data.BoundingBox],
    target_crs: str,
    mask_value_feature_property: Optional[str] = None,
) -> None:
    polygons = shape_readers.read_polygons_from_shape_file(
        input_shape_file_path,
        target_crs=target_crs,
        mask_value_feature_property=mask_value_feature_property,
    )
    filtered_polygons = (
        polygons
        if sub_area_bbox is None
        else list(
            polygon_transformers.crop_polygons_to_sub_area(polygons, sub_area_bbox)
        )
    )
    with pathlib.Path(output_polygon_file_path).open("w") as output_polygon_file:
        polygon_writers.write_polygons_to_text_file(
            filtered_polygons,
            output_polygon_file,
            support_mask_values=(mask_value_feature_property is not None),
        )


def transform_study_area_files(
    output_dir_path: pathlib.Path | str,
    elevation_file_path: pathlib.Path | str,
    sub_area_boundaries_shape_file_path: Optional[pathlib.Path | str] = None,
    buildings_shape_file_path: Optional[pathlib.Path | str] = None,
    green_areas_shape_file_path: Optional[pathlib.Path | str] = None,
    soil_classes_shape_file_path: Optional[pathlib.Path | str] = None,
    soil_classes_shape_file_property: Optional[str] = None,
    elevation_output_file_name: str = file_names.ELEVATION_TIF,
    boundaries_output_file_name: str = file_names.BOUNDARIES_TXT,
    buildings_output_file_name: str = file_names.BUILDINGS_TXT,
    green_areas_output_file_name: str = file_names.GREEN_AREAS_TXT,
    soil_classes_output_file_name: str = file_names.SOIL_CLASSES_TXT,
) -> pathlib.Path:
    """Function prepares study area input files in a format expected by the pipeline.

    Most of the files (except elevation one) are optional. If sub-area boundaries file
    is defined all input files are cropped to the bounding box calculated for the
    boundaries. Output files are created in the output directory (except for the case of
    elevation file with no need to crop it, in this case the original file should be
    used).

    Args:
        output_dir_path: Directory for output files.
        elevation_file_path: Path to input elevation data.
        sub_area_boundaries_shape_file_path: Optional shape-file with boundaries that
            is used to crop study area.
        buildings_shape_file_path: Optional shape-file with buildings.
        green_areas_shape_file_path: Optional shape-file with green areas.
        soil_classes_shape_file_path: Optional shape-file with soil classes
        soil_classes_shape_file_property: Optional property name in soil classes shape
            file that defines soil class values.
        elevation_output_file_name: Default elevation output file name.
        boundaries_output_file_name: Default boundaries output file name.
        buildings_output_file_name: Default buildings output file name.
        green_areas_output_file_name: Default green areas output file name.
        soil_classes_output_file_name: Default soil classes output file name.

    Returns:
        Output file path to elevation data (can point to input file path in case no
        changes are required for it).
    """
    with pathlib.Path(elevation_file_path).open("rb") as input_file:
        elevation_header = elevation_readers.read_from_geotiff(
            input_file, header_only=True
        ).header
    crs = elevation_header.crs.to_string()

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
        _transform_shape_file(
            buildings_shape_file_path,
            sub_area_buildings_file,
            sub_area_bbox,
            target_crs=crs,
        )

    if green_areas_shape_file_path is not None:
        print("Preparing Green Areas...")
        output_green_areas_file = output_dir / green_areas_output_file_name
        _transform_shape_file(
            green_areas_shape_file_path,
            output_green_areas_file,
            sub_area_bbox,
            target_crs=crs,
        )

    if soil_classes_shape_file_path is not None:
        print("Preparing Soil Classes...")
        output_soil_classes_file = output_dir / soil_classes_output_file_name
        _transform_shape_file(
            soil_classes_shape_file_path,
            output_soil_classes_file,
            sub_area_bbox,
            target_crs=crs,
            mask_value_feature_property=soil_classes_shape_file_property,
        )

    return output_elevation_file
