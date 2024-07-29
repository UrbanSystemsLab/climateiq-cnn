from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt
import rasterio


@dataclass
class ElevationHeader:
    """Elevation header data.

    Elevation header data class keeping information needed to convert
    geo-spacial coordinates to raster cells and back.

    Args:
        col_count: Number of columns in a raster cell matrix.
        row_count: Number of rows in a raster cell matrix.
        x_ll_corner: X-coordinate of lower-left corner of a raster region.
        y_ll_corner: Y-coordinate of lower-left corner of a raster region.
        cell_size: Size of each raster cell.
        nodata_value: Special data value that indicates that cells having this
            value correspond to missing data.
        crs: Optional value defining Coordinate Reference System.
    """

    col_count: int
    row_count: int
    x_ll_corner: float
    y_ll_corner: float
    cell_size: float
    nodata_value: float
    crs: rasterio.CRS | None = None

    def back_transform(self) -> rasterio.Affine:
        """Creates an affine converting raster indices to X/Y coordinates."""
        return rasterio.Affine(
            self.cell_size,
            0,
            self.x_ll_corner,
            0,
            -self.cell_size,
            self.y_ll_corner + self.row_count * self.cell_size,
        )

    def forward_transform(self) -> rasterio.Affine:
        """Creates an affine converting X/Y coordinates to raster indices."""
        return ~self.back_transform()


@dataclass(slots=True)
class Elevation:
    header: ElevationHeader
    data: npt.NDArray[np.float64] | None = None


@dataclass
class BoundingBox:
    """The bounding box represented by 4 bounds (min-x, min-y, max-x, max-y)."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def intersects(self, bbox2: "BoundingBox") -> bool:
        """Checks if this bounding box has intersection with another one.

        Args:
            bbox2: Another bounding box.

        Returns:
            Indicator of the intersection of bounding boxes.
        """
        # Intersection check is done separately over each of X- and Y-axis. Along each
        # dimension, we require that minimum bound of one box doesn't happen to be
        # greater than maximum bound of the other box.
        horizontal_overlap = not (self.min_x > bbox2.max_x or bbox2.min_x > self.max_x)
        vertical_overlap = not (self.min_y > bbox2.max_y or bbox2.min_y > self.max_y)
        return horizontal_overlap and vertical_overlap

    def contains(self, inner_bbox: "BoundingBox") -> bool:
        """Checks if inner bounding box is nested inside this one.

        Args:
            inner_bbox: Bounding box that is checked to be the inner one.

        Returns:
            Indicator of the nesting of the inner bounding box into this one.
        """
        # Nesting check is done separately over each of X- and Y-axis.
        hor_nesting = self.min_x <= inner_bbox.min_x and inner_bbox.max_x <= self.max_x
        vert_nesting = self.min_y <= inner_bbox.min_y and inner_bbox.max_y <= self.max_y
        return hor_nesting and vert_nesting

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Returns the data in form of a Tuple[min_x, min_y, max_x, max_y]."""
        return self.min_x, self.min_y, self.max_x, self.max_y

    @staticmethod
    def from_tuple(bbox: Tuple[float, float, float, float]) -> "BoundingBox":
        """Creates an instance from a Tuple[min_x, min_y, max_x, max_y]."""
        return BoundingBox(min_x=bbox[0], min_y=bbox[1], max_x=bbox[2], max_y=bbox[3])


@dataclass(slots=True, frozen=True)
class InfiltrationParams:
    """Infiltration properties of a soil class in flood simulation configuration."""

    soil_class: int
    hydraulic_conductivity: float
    wetting_front_suction_head: float
    effective_porosity: float
    effective_saturation: float


@dataclass(slots=True, frozen=True)
class InfiltrationConfiguration:
    """Infiltration configuration for soil classes in flood simulation configuration."""

    items: list[InfiltrationParams]

    def as_map(self) -> Dict[int, InfiltrationParams]:
        """Represents items as a dictionary indexed by soil class."""
        return {item.soil_class: item for item in self.items}


# Default infiltration configuration used to run flood simulations.
# TODO: Make these parameters configurable in case they vary between study areas.
DEFAULT_INFILTRATION_CONFIGURATION = InfiltrationConfiguration(
    items=[
        InfiltrationParams(
            soil_class=1,
            hydraulic_conductivity=1.09,
            wetting_front_suction_head=11.01,
            effective_porosity=0.412,
            effective_saturation=0.5,
        ),
        InfiltrationParams(
            soil_class=2,
            hydraulic_conductivity=0.65,
            wetting_front_suction_head=16.68,
            effective_porosity=0.486,
            effective_saturation=0.5,
        ),
        InfiltrationParams(
            soil_class=3,
            hydraulic_conductivity=0.1,
            wetting_front_suction_head=20.88,
            effective_porosity=0.309,
            effective_saturation=0.5,
        ),
        InfiltrationParams(
            soil_class=4,
            hydraulic_conductivity=0.34,
            wetting_front_suction_head=8.89,
            effective_porosity=0.434,
            effective_saturation=0.5,
        ),
        InfiltrationParams(
            soil_class=5,
            hydraulic_conductivity=2.99,
            wetting_front_suction_head=6.13,
            effective_porosity=0.401,
            effective_saturation=0.5,
        ),
        InfiltrationParams(
            soil_class=6,
            hydraulic_conductivity=11.78,
            wetting_front_suction_head=4.95,
            effective_porosity=0.417,
            effective_saturation=0.5,
        ),
        InfiltrationParams(
            soil_class=11,
            hydraulic_conductivity=1.09,
            wetting_front_suction_head=11.01,
            effective_porosity=0.412,
            effective_saturation=0.99,
        ),
        InfiltrationParams(
            soil_class=12,
            hydraulic_conductivity=0.65,
            wetting_front_suction_head=16.68,
            effective_porosity=0.486,
            effective_saturation=0.99,
        ),
        InfiltrationParams(
            soil_class=13,
            hydraulic_conductivity=0.1,
            wetting_front_suction_head=20.88,
            effective_porosity=0.309,
            effective_saturation=0.99,
        ),
        InfiltrationParams(
            soil_class=14,
            hydraulic_conductivity=0.34,
            wetting_front_suction_head=8.89,
            effective_porosity=0.434,
            effective_saturation=0.99,
        ),
        InfiltrationParams(
            soil_class=16,
            hydraulic_conductivity=11.78,
            wetting_front_suction_head=4.95,
            effective_porosity=0.417,
            effective_saturation=0.5,
        ),
    ]
)
