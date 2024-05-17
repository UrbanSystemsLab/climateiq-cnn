import typing
from typing import Optional, Tuple

import rasterio.features
from shapely import geometry

from usl_lib.readers import elevation_readers
from usl_lib.transformers import polygon_transformers
from usl_lib.writers import elevation_writers


def transform_geotiff_with_boundaries_to_esri_ascii(
    input_file: typing.BinaryIO,
    output_file: typing.TextIO,
    band: int = 1,
    no_data_value: Optional[float] = None,
    boundaries_polygons: Optional[list[Tuple[geometry.Polygon, int]]] = None,
) -> None:
    """Transforms elevation raster data from GeoTIFF file stream to Esri ASCII one.

    Transformation optionally includes a stage of zeroing-out values (setting No-data)
    for cells outside the polygon boundaries.

    Args:
        input_file: Binary stream to load from.
        output_file: Textual stream to write to.
        band: Index of a band that should be loaded from GeoTIFF.
        no_data_value: Optional value to set in the returned data to indicate absence of
            data. If not supplied, the no-data value defined in the TIFF file itself
            will be used.
        boundaries_polygons: Optional polygon boundaries that if defined would be used
            to set No-data for elevation raster outside of boundaries.
    """
    with rasterio.open(input_file, driver="GTiff") as src:
        input_nodata = src.nodata
        header = elevation_readers.read_header_from_rasterio_dataset_reader(
            src, no_data_value
        )
        elevation_writers.write_header_to_esri_ascii_raster_file(header, output_file)

        if boundaries_polygons is not None:
            boundaries_raster = polygon_transformers.rasterize_polygons(
                header, boundaries_polygons
            )

        data = src.read(band)
        row_index = 0
        for row in data:
            if no_data_value is not None:
                row[row == input_nodata] = no_data_value
            row_values = row.tolist()
            # Zero-out elevation cells where boundaries raster is 0 (meaning outside the
            # boundaries)
            if boundaries_raster is not None:
                boundaries_raster_row_values = boundaries_raster[row_index].tolist()
                for col_index in range(len(row_values)):
                    if boundaries_raster_row_values[col_index] == 0:
                        row_values[col_index] = no_data_value
            # Export row to output_file
            output_file.write(f"{' '.join(str(item) for item in row_values)}\n")
            row_index = row_index + 1
