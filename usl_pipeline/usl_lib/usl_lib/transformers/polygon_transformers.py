from typing import Iterable, Tuple

import numpy
import numpy.typing as npt
import rasterio
from rasterio import features
from shapely import geometry

from usl_lib.shared import geo_data


def rasterize_polygons(
    header: geo_data.ElevationHeader,
    polygons_with_masks: list[Tuple[geometry.Polygon, int]],
    background_value: int = 0,
) -> npt.NDArray[numpy.int_]:
    """Rasterize polygons to an integer raster matrix.

    Args:
        header: Coordinate system and raster grid information.
        polygons_with_masks: Polygons to rasterize with assigned mask values (mask
            values are used to draw and fill in polygon shape in the raster).
        background_value: Optional background mask used to fill in the raster before
            drawing polygons (default value is 0).

    Returns:
        Matrix with integer raster masks of the size corresponding to a number of rows
        and columns in the cell grid of the study area defined by the header.
    """
    for polygon_mask in polygons_with_masks:
        if polygon_mask[1] == background_value:
            raise ValueError(
                "Polygons with background mask are not allowed: {}".format(polygon_mask)
            )

    min_x = header.x_ll_corner
    min_y = header.y_ll_corner
    cell_size = header.cell_size
    raster_cols = header.col_count
    raster_rows = header.row_count

    if len(polygons_with_masks) == 0:
        return numpy.full((raster_rows, raster_cols), background_value, dtype=int)

    # This transformation is backward one mapping raster cells back to original
    # coordinates (original_x = min_x + raster_x * step, y is similar).
    transform = rasterio.Affine(cell_size, 0, min_x, 0, cell_size, min_y)

    raster_matrix = features.rasterize(
        polygons_with_masks,
        fill=background_value,
        out_shape=(raster_rows, raster_cols),
        transform=transform,
    )

    # Up-down flip is needed to correct the direction of Y-axis which should go from
    # the top to the bottom when raster is visualized.
    return numpy.flipud(raster_matrix)


def get_bounding_box_for_boundaries(
    boundary_polygons: Iterable[geometry.Polygon],
) -> geo_data.BoundingBox:
    """Calculates bounding box for polygons.

    Args:
        boundary_polygons: Sequence of polygons to iterate over.

    Returns:
        The smallest bounding box containing all the polygons.
    """
    bbox_list = [p.bounds for p in boundary_polygons]
    return geo_data.BoundingBox(
        min_x=min(bbox[0] for bbox in bbox_list),
        min_y=min(bbox[1] for bbox in bbox_list),
        max_x=max(bbox[2] for bbox in bbox_list),
        max_y=max(bbox[3] for bbox in bbox_list),
    )


def crop_polygons_to_sub_area(
    polygon_masks: Iterable[Tuple[geometry.Polygon, int]],
    sub_area_bounding_box: geo_data.BoundingBox,
) -> Iterable[Tuple[geometry.Polygon, int]]:
    """Crops the polygon data by sub-area bounding box.

    Args:
        polygon_masks: Source of polygons with associated mask values.
        sub_area_bounding_box: Sub-area bounding box defined in coordinate system of the
            source elevation data.

    Returns:
        Iterator of polygon/mask tuple overlapping with sub-area bounding box.
    """
    x1, y1, x2, y2 = sub_area_bounding_box.to_tuple()
    sub_area_polygon = geometry.Polygon(
        [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    )
    for polygon_mask in polygon_masks:
        pol_bbox = geo_data.BoundingBox.from_tuple(polygon_mask[0].bounds)

        # Let's ignore polygon if it's outside the sub-area.
        if not sub_area_bounding_box.intersects(pol_bbox):
            continue
        if sub_area_bounding_box.contains(pol_bbox):
            yield polygon_mask
        else:
            poly_or_multi = polygon_mask[0].intersection(sub_area_polygon)
            # Intersection of 2 polygons may be either a polygon or a multi-polygon
            if poly_or_multi.geom_type == "Polygon":
                if not poly_or_multi.is_empty:
                    yield poly_or_multi, polygon_mask[1]
            elif poly_or_multi.geom_type == "MultiPolygon":
                for sub_polygon in poly_or_multi.geoms:
                    if not sub_polygon.is_empty:
                        yield sub_polygon, polygon_mask[1]


def intersect(
    polygon_masks_to_process: Iterable[Tuple[geometry.Polygon, int]],
    multi_polygon_boundaries: geometry.MultiPolygon,
) -> Iterable[Tuple[geometry.Polygon, int]]:
    for polygon_mask in polygon_masks_to_process:
        poly_or_multi = polygon_mask[0].intersection(multi_polygon_boundaries)
        # Intersection of 2 polygons may be either a polygon or a multi-polygon
        if poly_or_multi.geom_type == "Polygon":
            if not poly_or_multi.is_empty:
                yield poly_or_multi, polygon_mask[1]
        elif poly_or_multi.geom_type == "MultiPolygon":
            for sub_polygon in poly_or_multi.geoms:
                if not sub_polygon.is_empty:
                    yield sub_polygon, polygon_mask[1]
