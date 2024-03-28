import typing
from typing import Optional

import rasterio.features

from usl_pipeline.shared.geo_data import Elevation
from usl_pipeline.shared.geo_data import ElevationHeader


def read_from_geotiff(
    file: typing.BinaryIO,
    header_only: bool = False,
    band: int = 1,
    no_data_value: Optional[float] = None,
) -> Elevation:
    """Loading elevation raster data from GeoTIFF file.

    Args:
        file: Binary stream to load from.
        header_only: Indicates that only header should be loaded, whereas
            elevation data should be skipped.
        band: Index of a band that should be loaded from GeoTIFF.
        no_data_value: Optional value to set in the returned data to indicate
            absence of data. If not supplied, the no-data value defined in the
            TIFF file itself will be used.

    Returns:
        Elevation object.
    """

    with rasterio.open(file, driver="GTiff") as src:
        print(src.profile)
        transform = src.transform
        ll_corner = transform * (0, src.height)
        input_nodata = src.nodata
        final_nodata_value = (
            no_data_value if no_data_value is not None else input_nodata
        )
        elv_header = ElevationHeader(
            col_count=src.width,
            row_count=src.height,
            x_ll_corner=ll_corner[0],
            y_ll_corner=ll_corner[1],
            cell_size=transform[0],
            nodata_value=final_nodata_value,
            crs=src.crs,
        )
        elv_data = None

        if not header_only:
            data = src.read(band)
            if no_data_value is not None:
                data[data == input_nodata] = no_data_value
            elv_data = data

        return Elevation(header=elv_header, data=elv_data)
