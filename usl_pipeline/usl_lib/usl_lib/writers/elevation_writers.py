import json
import pathlib
import typing

import numpy
import rasterio

from usl_lib.shared import geo_data


def write_header_to_esri_ascii_raster_file(
    header: geo_data.ElevationHeader, file: typing.TextIO
) -> None:
    """Writes elevation header to a text file stream in Esri ASCII format.

    Args:
        header: Elevation header.
        file: Output file/stream to write data to.
    """
    file.write("ncols {}\n".format(header.col_count))
    file.write("nrows {}\n".format(header.row_count))
    file.write("xllcorner {}\n".format(header.x_ll_corner))
    file.write("yllcorner {}\n".format(header.y_ll_corner))
    file.write("cellsize {}\n".format(header.cell_size))
    file.write("NODATA_value {}\n".format(header.nodata_value))


def write_header_to_json_file(
    header: geo_data.ElevationHeader, output_stream: typing.TextIO
) -> None:
    """Writes elevation header to an output stream in JSON format.

    Args:
        header: Elevation header.
        output_stream: Output file/stream to write data to.
    """
    json.dump(
        {
            "col_count": header.col_count,
            "row_count": header.row_count,
            "x_ll_corner": header.x_ll_corner,
            "y_ll_corner": header.y_ll_corner,
            "cell_size": header.cell_size,
            "nodata_value": header.nodata_value,
            "crs": None if header.crs is None else header.crs.to_string(),
        },
        output_stream,
        indent=4,
    )


def write_to_esri_ascii_raster_file(
    elevation: geo_data.Elevation, file: typing.TextIO
) -> None:
    """Writes elevation data to a text file in Esri ASCII format.

    Args:
        elevation: Elevation data.
        file: Output file/stream to write data to.
    """
    if elevation.data is None:
        raise ValueError("Elevation data must be present")

    write_header_to_esri_ascii_raster_file(elevation.header, file)
    numpy.savetxt(file, elevation.data, delimiter=" ", fmt="%s")


def write_to_geotiff(
    elevation: geo_data.Elevation,
    target_file_path: pathlib.Path | str,
):
    """Writes elevation data to a GeoTIFF file using band number 1.

    Args:
        elevation: Elevation data.
        target_file_path: Path to output file.
    """
    height = elevation.header.row_count
    width = elevation.header.col_count
    cell_size = elevation.header.cell_size
    with rasterio.open(
        str(target_file_path),
        "w",
        driver="GTiff",
        dtype=rasterio.float32,
        nodata=elevation.header.nodata_value,
        width=width,
        height=height,
        count=1,
        crs=elevation.header.crs,
        transform=rasterio.Affine(
            cell_size,
            0.0,
            elevation.header.x_ll_corner,
            0.0,
            -cell_size,
            elevation.header.y_ll_corner + cell_size * height,
        ),
    ) as raster:
        raster.write(elevation.data, 1)
