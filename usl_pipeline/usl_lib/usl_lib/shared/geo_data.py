import typing
from dataclasses import dataclass
from importlib import resources
from typing import Dict, Optional, Tuple
import xml.etree.ElementTree as ElementTree

import numpy as np
import numpy.typing as npt
import rasterio

from usl_lib.shared import resources as resource_data


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
    crs: Optional[rasterio.CRS] = None

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
    data: Optional[npt.NDArray[np.float64]] = None


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


def _find_xml_child_float_value(
    parent_node: ElementTree.Element,
    tag: str,
) -> float:
    items = [item for item in parent_node if item.tag == tag]
    if len(items) != 1:
        raise ValueError(f"Unexpected number of sub-items in xml for tag {tag}")
    if items[0].text is None:
        raise ValueError(f"Text value missing for xml sub-item {items[0]}")
    return float(items[0].text)


@dataclass
class InfiltrationParams:
    """Infiltration properties of a soil class in flood simulation configuration."""

    soil_class: int
    hydraulic_conductivity: float
    wetting_front_suction_head: float
    effective_porosity: float
    effective_saturation: float

    @staticmethod
    def parse_from_xml_node(node: ElementTree.Element) -> "InfiltrationParams":
        """Parses infiltration properties for a given soil class from XML."""
        soil_class_text = node.get("soilId")
        if soil_class_text is None:
            raise ValueError("Attribute 'soilId' missing for InfiltrationParams in xml")
        soil_class = int(soil_class_text)
        hydraulic_conductivity = _find_xml_child_float_value(node, "HydrConductivity")
        wetting_front_suction_head = _find_xml_child_float_value(
            node, "WettingFrontSuctionHead"
        )
        effective_porosity = _find_xml_child_float_value(node, "EffectivePorosity")
        effective_saturation = _find_xml_child_float_value(node, "EffectiveSaturation")
        return InfiltrationParams(
            soil_class=soil_class,
            hydraulic_conductivity=hydraulic_conductivity,
            wetting_front_suction_head=wetting_front_suction_head,
            effective_porosity=effective_porosity,
            effective_saturation=effective_saturation,
        )


@dataclass
class InfiltrationConfiguration:
    """Infiltration configuration for soil classes in flood simulation configuration."""

    items: list[InfiltrationParams]

    @staticmethod
    def parse_from_xml_file(fd: typing.IO[str]) -> "InfiltrationConfiguration":
        """Parses infiltration configuration for all soil classe from XML."""
        root = ElementTree.parse(fd).getroot()
        return InfiltrationConfiguration(
            items=[InfiltrationParams.parse_from_xml_node(item) for item in root]
        )

    @staticmethod
    def load_default() -> "InfiltrationConfiguration":
        """Loads default infiltration configuration from internal resource."""
        config_file_resource = resources.files(resource_data) / "infiltration.xml"
        with config_file_resource.open("r") as fd:
            return InfiltrationConfiguration.parse_from_xml_file(fd)

    def as_map(self) -> Dict[int, InfiltrationParams]:
        """Represents items as a dictionary indexed by soil class."""
        return {item.soil_class: item for item in self.items}
