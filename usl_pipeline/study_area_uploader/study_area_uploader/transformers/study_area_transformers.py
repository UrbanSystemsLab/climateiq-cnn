import pathlib
from typing import Iterable, Optional, Tuple

from shapely import geometry

from study_area_uploader.readers import polygon_readers as shape_readers
from usl_lib.shared import geo_data
from usl_lib.transformers import polygon_transformers


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
