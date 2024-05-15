from dataclasses import dataclass
import pathlib
from typing import Iterable, Optional, Tuple

from google.cloud.storage import bucket
from shapely import geometry

from study_area_uploader.readers import polygon_readers as shape_readers
from study_area_uploader.transformers import elevation_transformers
from usl_lib.readers import elevation_readers
from usl_lib.shared import geo_data
from usl_lib.storage import file_names
from usl_lib.transformers import polygon_transformers
from usl_lib.writers import polygon_writers


@dataclass
class PreparedInputData:
    """Input data needed to run pipeline for flood scenarios."""

    elevation_file_path: pathlib.Path
    boundaries_polygons: Optional[list[Tuple[geometry.Polygon, int]]]
    buildings_polygons: Optional[list[Tuple[geometry.Polygon, int]]]
    green_areas_polygons: Optional[list[Tuple[geometry.Polygon, int]]]
    soil_classes_polygons: Optional[list[Tuple[geometry.Polygon, int]]]


def transform_shape_file(
    input_shape_file_path: pathlib.Path | str,
    sub_area_bounding_box: Optional[geo_data.BoundingBox],
    target_crs: str,
    mask_value_feature_property: Optional[str] = None,
    skip_zero_masks: bool = True,
) -> Iterable[Tuple[geometry.Polygon, int]]:
    """Reads, filters and transforms polygons from shape-file with optional cropping.

    Args:
        input_shape_file_path: Path to a shape file to read from.
        sub_area_bounding_box: Optional bounding box that is used to crop polygon area.
        target_crs: CRS that polygon coordinates should be translated to (typically
            comes from elevation data header).
        mask_value_feature_property: Optional shape feature property to load mask from.
            In case this property is not defined by the caller, mask value 1 is used.
        skip_zero_masks: Indicator that polygons associated with 0 masks should be
            filtered out (only has effect when mask_value_feature_property is defined).

    Returns:
        Iterator of polygons with associated masks.
    """
    polygons_iterator = (
        p
        for p in shape_readers.read_polygons_from_shape_file(
            input_shape_file_path,
            target_crs=target_crs,
            mask_value_feature_property=mask_value_feature_property,
        )
    )
    if mask_value_feature_property is not None and skip_zero_masks:
        polygons_iterator = (p for p in polygons_iterator if p[1] != 0)
    return (
        polygons_iterator
        if sub_area_bounding_box is None
        else polygon_transformers.crop_polygons_to_sub_area(
            polygons_iterator, sub_area_bounding_box
        )
    )


def prepare_and_upload_study_area_files(
    study_area_name: str,
    elevation_file_path: str | pathlib.Path,
    boundaries_shape_file_path: Optional[str | pathlib.Path],
    buildings_shape_file_path: Optional[str | pathlib.Path],
    green_areas_shape_file_path: Optional[str | pathlib.Path],
    soil_classes_shape_file_path: Optional[str | pathlib.Path],
    soil_class_mask_feature_property: Optional[str],
    work_dir: pathlib.Path,
    study_area_bucket: bucket.Bucket,
) -> PreparedInputData:
    """Prepares data needed to run pipeline for flood scenarios.

    Args:
        study_area_name: Name of study area that is used to form file paths in cloud
            storage bucket.
        elevation_file_path: Path to elevation GeoTIFF file.
        boundaries_shape_file_path: Optional path to shape file with boundaries to crop
            and carve out sub-area.
        buildings_shape_file_path: Optional path to shape file with building footprint.
        green_areas_shape_file_path: Optional path to shape file with green areas.
        soil_classes_shape_file_path: Optional path to shape file with regions
            describing soil class info.
        soil_class_mask_feature_property: Optional property name in soil classes shape
            file that defines soil class values.
        work_dir: A folder that can be used for transforming files.
        study_area_bucket: Target cloud storage bucket to export study area files to.

    Returns:
        Prepared input data that can be exported to CityCat or be passed to chunker.
    """
    with open(elevation_file_path, "rb") as input_file:
        elevation_header = elevation_readers.read_from_geotiff(
            input_file, header_only=True
        ).header
    crs = elevation_header.crs.to_string()

    boundaries_polygons: Optional[list[Tuple[geometry.Polygon, int]]] = None
    sub_area_bounding_box: Optional[geo_data.BoundingBox] = None
    buildings_polygons: Optional[list[Tuple[geometry.Polygon, int]]] = None
    green_areas_polygons: Optional[list[Tuple[geometry.Polygon, int]]] = None
    soil_classes_polygons: Optional[list[Tuple[geometry.Polygon, int]]] = None
    if boundaries_shape_file_path is None:
        output_elevation_file_path = pathlib.Path(elevation_file_path)
    else:
        boundaries_polygons = shape_readers.read_polygons_from_shape_file(
            boundaries_shape_file_path, target_crs=crs
        )
        # Write boundaries to study area bucket
        with study_area_bucket.blob(
            f"{study_area_name}/{file_names.BOUNDARIES_TXT}"
        ).open("w") as output_file:
            polygon_writers.write_polygons_to_text_file(
                boundaries_polygons, output_file
            )
        # Calculate bounding box rectangle for cropping sub-area
        sub_area_bounding_box = polygon_transformers.get_bounding_box_for_boundaries(
            p[0] for p in boundaries_polygons
        )
        # Crop elevation data
        output_elevation_file_path = work_dir / file_names.ELEVATION_TIF
        elevation_transformers.crop_geotiff_to_sub_area(
            elevation_file_path, str(output_elevation_file_path), sub_area_bounding_box
        )

    # Write elevation to study area bucket
    study_area_bucket.blob(
        f"{study_area_name}/{file_names.ELEVATION_TIF}"
    ).upload_from_filename(str(output_elevation_file_path))

    # Read polygon files
    if buildings_shape_file_path is not None:
        buildings_polygons = list(
            transform_shape_file(buildings_shape_file_path, sub_area_bounding_box, crs)
        )
        # Write buildings to study area bucket
        with study_area_bucket.blob(
            f"{study_area_name}/{file_names.BUILDINGS_TXT}"
        ).open("w") as output_file:
            polygon_writers.write_polygons_to_text_file(buildings_polygons, output_file)
    if green_areas_shape_file_path is not None:
        green_areas_polygons = list(
            transform_shape_file(
                green_areas_shape_file_path, sub_area_bounding_box, crs
            )
        )
        # Write green areas to study area bucket
        with study_area_bucket.blob(
            f"{study_area_name}/{file_names.GREEN_AREAS_TXT}"
        ).open("w") as output_file:
            polygon_writers.write_polygons_to_text_file(
                green_areas_polygons, output_file
            )
        # Soil information is only used when green area data is defined.
        if soil_classes_shape_file_path is not None:
            soil_classes_polygons = list(
                transform_shape_file(
                    soil_classes_shape_file_path,
                    sub_area_bounding_box,
                    crs,
                    mask_value_feature_property=soil_class_mask_feature_property,
                )
            )
            # Write soil classes to study area bucket
            with study_area_bucket.blob(
                f"{study_area_name}/{file_names.SOIL_CLASSES_TXT}"
            ).open("w") as output_file:
                polygon_writers.write_polygons_to_text_file(
                    soil_classes_polygons, output_file, support_mask_values=True
                )

    return PreparedInputData(
        output_elevation_file_path,
        boundaries_polygons,
        buildings_polygons,
        green_areas_polygons,
        soil_classes_polygons,
    )
