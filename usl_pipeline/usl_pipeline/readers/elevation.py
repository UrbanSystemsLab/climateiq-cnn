import typing
from typing import Optional

import numpy
import rasterio.features

from usl_pipeline.shared.geo_data import Elevation
from usl_pipeline.shared.geo_data import ElevationHeader


def read_from_geotiff(
    file: typing.BinaryIO,
    header_only: bool = False,
    band: int = 1,
    no_data_value: Optional[float] = None,
) -> Elevation:
    """Loading elevation raster data from GeoTIFF file/stream.

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
        elv_header = ElevationHeader(
            col_count=src.width,
            row_count=src.height,
            x_ll_corner=ll_corner[0],
            y_ll_corner=ll_corner[1],
            cell_size=transform[0],
            nodata_value=(no_data_value if no_data_value is not None else input_nodata),
            crs=src.crs,
        )
        elv_data = None

        if not header_only:
            elv_data = src.read(band)
            if no_data_value is not None:
                elv_data[elv_data == input_nodata] = no_data_value

        return Elevation(header=elv_header, data=elv_data)


def read_from_esri_ascii(
    file: typing.TextIO,
    header_only: bool = False,
    no_data_value: Optional[float] = None,
) -> Elevation:
    """Loading elevation raster data from Esri ASCII file/stream.

    Args:
        file: Binary stream to load from.
        header_only: Indicates that only header should be loaded, whereas
            elevation data should be skipped.
        no_data_value: Optional value to set in the returned data to indicate
            absence of data. If not supplied, the no-data value defined in the
            TIFF file itself will be used.

    Returns:
        Elevation object.
    """
    print(type(file))
    # read 6 header lines
    header_map = {}
    for row_index in range(0, 6):
        line = file.readline()
        # split into key/value pair
        parts = line.split(" ", 1)
        header_map[parts[0].lower()] = parts[1]
    input_nodata = float(header_map["nodata_value"])
    header = ElevationHeader(
        col_count=int(header_map["ncols"]),
        row_count=int(header_map["nrows"]),
        x_ll_corner=float(header_map["xllcorner"]),
        y_ll_corner=float(header_map["yllcorner"]),
        cell_size=float(header_map["cellsize"]),
        nodata_value=(no_data_value if no_data_value is not None else input_nodata),
    )

    # read data matrix
    data = None
    if not header_only:
        data = numpy.loadtxt(file, dtype="float32")
        if no_data_value is not None:
            data[data == input_nodata] = no_data_value

    return Elevation(header=header, data=data)
