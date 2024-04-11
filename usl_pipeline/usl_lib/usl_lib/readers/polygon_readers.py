import typing

import fiona
import fiona.crs
import pyproj
from shapely.geometry import Polygon


SHAPE_FILE_POLYGON_FEATURE_TYPE = "Polygon"


def read_polygons_from_shape_file(
    file_path: str,
    target_crs: typing.Optional[str] = None,
    mask_value_feature_property: typing.Optional[str] = None,
    skip_zero_mask_values: bool = True,
) -> list[typing.Tuple[Polygon, int]]:
    """Reads polygon data from shape file.

    Function reads all the polygon shapes from shape file. There is optional support for
    transformation of coordinates to different CRS. Another option is to load masks
    associated to each polygon feature in caller-defined property. Third option is the
    ability to filter out polygons with 0 values in associated mask (switched on by
    default).

    Args:
        file_path: Path to a shape file to read from.
        target_crs: Optional CRS to transform coordinates to.
        mask_value_feature_property: Optional shape feature property to load mask from.
            In case this property is not defined by the caller, mask value 1 is used.
        skip_zero_mask_values: Indicates that polygons with mask 0 should be skipped.
            This mode is switched on by default.

    Returns:
         The list of tuples combining polygon with associated mask.
    """
    layer = fiona.open(file_path)
    source_crs = fiona.crs.to_string(layer.crs)
    transformer = None
    if target_crs is not None and target_crs != source_crs:
        transformer = pyproj.Transformer.from_crs(
            source_crs, target_crs, always_xy=True
        )
    polygons = []
    for feature in layer:
        feature_type = feature.geometry.type
        if feature_type != SHAPE_FILE_POLYGON_FEATURE_TYPE:
            continue
        fragments = feature.geometry.coordinates
        mask_value = (
            1
            if mask_value_feature_property is None
            else int(
                fiona.model.to_dict(feature.properties)[mask_value_feature_property]
            )
        )
        if mask_value == 0 and skip_zero_mask_values:
            continue
        for fragment in fragments:
            transformed_fragment = fragment
            if transformer is not None:
                xx, yy = transformer.transform(
                    [p[0] for p in fragment], [p[1] for p in fragment]
                )
                transformed_fragment = zip(xx, yy)
            polygons.append((Polygon(transformed_fragment), mask_value))
    return polygons
