import rasterio.features


def load_elevation_from_geotiff(
    file_path: str, load_data: bool = False, result_nodata: float = None
):
    """
    Loading elevation raster data from GeoTIFF file.
    :param file_path: path to a file to load from
    :param load_data: indicates that in addition to header, data should also be
            loaded
    :param result_nodata: optional new value for special case of no data in
            raster cells
    :return: 3-size tuple (header, data, CRS of the raster)
    """

    with rasterio.open(file_path) as src:
        transform = src.transform
        ll_corner = transform * (0, src.height)
        input_nodata = src.nodata
        elv_header = {
            "ncols": int(src.width),
            "nrows": int(src.height),
            "xllcorner": float(ll_corner[0]),
            "yllcorner": float(ll_corner[1]),
            "cellsize": float(transform[0]),
            "nodata_value": float(
                result_nodata if result_nodata is not None else input_nodata
            ),
        }
        elv_data = None

        if load_data:
            data = src.read(1)
            if result_nodata is not None:
                data[data == input_nodata] = result_nodata
            elv_data = data.tolist()

        return elv_header, elv_data, src.crs
