from enum import StrEnum
import pathlib
from typing import Iterable, Tuple
import fiona
import fiona.crs
import pyproj
from pyproj import transformer
from shapely import geometry


class _SupportedShapeFileFeatureType(StrEnum):
    POLYGON = "Polygon"
    MULTI_POLYGON = "MultiPolygon"


_Point = Tuple[float, float]
_PointFragment = Iterable[_Point]


def _transform_point_fragment(
    fragment: _PointFragment,
    point_transformer: transformer.Transformer | None,
) -> _PointFragment:
    if point_transformer is None:
        return fragment
    xx, yy = point_transformer.transform(
        [p[0] for p in fragment], [p[1] for p in fragment]
    )
    return zip(xx, yy)


def read_polygons_from_shape_file(
    file_path: str | pathlib.Path,
    target_crs: str | None = None,
    mask_value_feature_property: str | None = None,
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
    layer = fiona.open(str(file_path))
    source_crs = fiona.crs.to_string(layer.crs)
    point_transformer: transformer.Transformer | None = None
    if target_crs is not None and target_crs != source_crs:
        point_transformer = pyproj.Transformer.from_crs(
            source_crs, target_crs, always_xy=True
        )
    polygons: list[Tuple[geometry.Polygon, int]] = []
    for feature in layer:
        feature_type = feature.geometry.type
        # Skipping non polygon features:
        if feature_type not in (
            _SupportedShapeFileFeatureType.POLYGON,
            _SupportedShapeFileFeatureType.MULTI_POLYGON,
        ):
            continue
        is_multi_polygon = feature_type == _SupportedShapeFileFeatureType.MULTI_POLYGON
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
            if is_multi_polygon:
                polygons.extend(
                    (
                        geometry.Polygon(
                            _transform_point_fragment(f, point_transformer)
                        ),
                        mask_value,
                    )
                    for f in fragment
                )
            else:
                polygons.append(
                    (
                        geometry.Polygon(
                            _transform_point_fragment(fragment, point_transformer)
                        ),
                        mask_value,
                    )
                )
    return polygons
