from typing import Optional, Tuple

import fiona
import fiona.crs
import pyproj
from shapely import geometry

SHAPE_FILE_POLYGON_FEATURE_TYPE = "Polygon"


def read_polygons_from_shape_file(
    file_path: str,
    target_crs: Optional[str] = None,
    mask_value_feature_property: Optional[str] = None,
) -> list[Tuple[geometry.Polygon, int]]:
    """Reads polygon data from shape file.

    The function supports several optional transformations. target_crs transforms the
    coordinates to a different CRS. mask_value_feature_property loads masks
    associated to each polygon feature in caller-defined property.

    Args:
        file_path: Path to a shape file to read from.
        target_crs: Optional CRS to transform coordinates to.
        mask_value_feature_property: Optional shape feature property to load mask from.
            In case this property is not defined by the caller, mask value 1 is used.

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
        # Skipping non polygon features:
        if feature_type != SHAPE_FILE_POLYGON_FEATURE_TYPE:
            continue
        fragments = feature.geometry.coordinates
        mask_value = 1

        if mask_value_feature_property is not None:
            feature_properties = fiona.model.to_dict(feature.properties)
            if mask_value_feature_property not in feature_properties:
                raise ValueError(
                    "Mask value key '{}' not found in feature properties {}".format(
                        mask_value_feature_property, feature_properties
                    )
                )
            try:
                mask_value = int(feature_properties[mask_value_feature_property])
            except ValueError:
                raise ValueError(
                    "Mask value '{}' for key '{}' should be integer in {}".format(
                        feature_properties[mask_value_feature_property],
                        mask_value_feature_property,
                        feature_properties,
                    )
                )

        for fragment in fragments:
            transformed_fragment = fragment
            if transformer is not None:
                xx, yy = transformer.transform(
                    [p[0] for p in fragment], [p[1] for p in fragment]
                )
                transformed_fragment = zip(xx, yy)
            polygons.append((geometry.Polygon(transformed_fragment), mask_value))
    return polygons
