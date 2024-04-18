import typing
from typing import Tuple

from shapely import geometry


def write_polygons_to_text_file(
    polygons_with_masks: list[Tuple[geometry.Polygon, int]],
    output_file: typing.TextIO,
    support_mask_values: bool = False,
) -> None:
    """Writes polygon information with optional mask values to text file.

    Args:
        polygons_with_masks: List of tuples combining polygon with associated mask.
        output_file: File text stream to write to.
        support_mask_values: Optional indicator that masks should be added into output
            as additional first column.
    """
    output_file.write(f"{len(polygons_with_masks)}\n")
    for polygon_mask in polygons_with_masks:
        xx, yy = polygon_mask[0].exterior.coords.xy
        mask_prefix = f"{polygon_mask[1]} " if support_mask_values else ""
        output_file.write(
            "{}{} {} {}\n".format(
                mask_prefix,
                len(xx),
                " ".join(str(v) for v in xx),
                " ".join(str(v) for v in yy),
            )
        )
