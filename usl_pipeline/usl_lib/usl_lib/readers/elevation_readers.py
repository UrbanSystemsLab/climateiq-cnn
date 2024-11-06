import json
import typing

import numpy
import rasterio.features
import rasterio.io

from usl_lib.shared import geo_data


_ESRI_HEADER_KEYWORDS = {
    "ncols",
    "nrows",
    "xllcorner",
    "yllcorner",
    "cellsize",
    "nodata_value",
}


def read_header_from_rasterio_dataset_reader(
    dataset_reader: rasterio.io.DatasetReader,
    no_data_value: float | None = None,
) -> geo_data.ElevationHeader:
    """Loading elevation header from GeoTIFF file/stream.

    Args:
        dataset_reader: RasterIO Dataset reader to load from.
        no_data_value: Optional value to set in the returned data to indicate absence of
            data. If not supplied, the no-data value defined in the TIFF file itself
            will be used.

    Returns:
        Elevation header object.
    """
    transform = dataset_reader.transform
    ll_corner = transform * (0, dataset_reader.height)
    input_nodata = dataset_reader.nodata
    return geo_data.ElevationHeader(
        col_count=dataset_reader.width,
        row_count=dataset_reader.height,
        x_ll_corner=ll_corner[0],
        y_ll_corner=ll_corner[1],
        cell_size=transform[0],
        nodata_value=(no_data_value if no_data_value is not None else input_nodata),
        crs=dataset_reader.crs,
    )


def read_from_geotiff(
    file: typing.BinaryIO,
    header_only: bool = False,
    band: int = 1,
    no_data_value: float | None = -9999.0,
) -> geo_data.Elevation:
    """Loading elevation raster data from GeoTIFF file/stream.

    Args:
        file: Binary stream to load from.
        header_only: Indicates that only header should be loaded, whereas elevation data
            should be skipped.
        band: Index of a band that should be loaded from GeoTIFF.
        no_data_value: Optional value to set in the returned data to indicate absence of
            data. If not supplied, the no-data value defined in the TIFF file itself
            will be used.

    Returns:
        Elevation object.
    """
    with rasterio.open(file, driver="GTiff") as src:
        input_nodata = src.nodata
        elv_header = read_header_from_rasterio_dataset_reader(src, no_data_value)
        elv_data = None

        if not header_only:
            elv_data = src.read(band)
            if no_data_value is not None:
                elv_data[elv_data == input_nodata] = no_data_value

        return geo_data.Elevation(header=elv_header, data=elv_data)


def read_from_esri_ascii(
    file: typing.TextIO,
    header_only: bool = False,
    no_data_value: float | None = None,
) -> geo_data.Elevation:
    """Loading elevation raster data from Esri ASCII file/stream.

    Args:
        file: Text stream to load from.
        header_only: Indicates that only header should be loaded, whereas elevation data
            should be skipped.
        no_data_value: Optional value to set in the returned data to indicate absence of
            data. If not supplied, the no-data value defined in the TIFF file itself
            will be used.

    Returns:
        Elevation object.
    """
    # Read initial header lines
    header_map = {}
    while True:
        read_start = file.tell()
        line = file.readline().rstrip()
        # split into key/value pair
        key, value = line.split(maxsplit=1)
        key = key.lower()

        # If the file line describes a header value, record it. Otherwise, seek back to
        # where the line began so it can be re-read in the numpy.loadtxt call below.
        if key in _ESRI_HEADER_KEYWORDS:
            header_map[key] = value
        else:
            file.seek(read_start)
            break

    # The nodata_value header is optional and defaults to -9999.
    # https://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/esri-ascii-raster-format.htm
    input_nodata = float(header_map.get("nodata_value", -9999))
    header = geo_data.ElevationHeader(
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
        data = numpy.loadtxt(file, dtype="float64")
        if no_data_value is not None:
            data[data == input_nodata] = no_data_value

    return geo_data.Elevation(header=header, data=data)


def read_header_from_json_file(
    input_stream: typing.TextIO,
) -> geo_data.ElevationHeader:
    """Loading elevation header from JSON file/stream.

    Args:
        input_stream: Text stream to load from.

    Returns:
        Elevation header.
    """
    header_map = json.load(input_stream)
    crs_text = header_map["crs"]
    return geo_data.ElevationHeader(
        col_count=int(header_map["col_count"]),
        row_count=int(header_map["row_count"]),
        x_ll_corner=float(header_map["x_ll_corner"]),
        y_ll_corner=float(header_map["y_ll_corner"]),
        cell_size=float(header_map["cell_size"]),
        nodata_value=float(header_map["nodata_value"]),
        crs=None if crs_text is None else rasterio.CRS({"init": crs_text}),
    )
