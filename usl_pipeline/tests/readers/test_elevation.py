import os

from rasterio import CRS

from usl_pipeline.readers.elevation import read_from_geotiff
from usl_pipeline.shared.entities import ElevationHeader


def test_load_elevation_from_geotiff_default_no_data():
    test_file_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "test_data.tif"
    )
    with open(test_file_path, "rb") as fd:
        elevation = read_from_geotiff(fd)
        assert elevation.header == \
               ElevationHeader(
                       col_count=2,
                       row_count=2,
                       x_ll_corner=587069.5669010617,
                       y_ll_corner=4505948.133973204,
                       cell_size=2.0,
                       nodata_value=0.0,
                       crs=CRS({"init": "EPSG:32618"})
               )
        assert elevation.data == \
               [
                       [0.0, 5.72777795791626],
                       [5.713333606719971, 5.742222309112549],
               ]
        assert not elevation.header.crs.is_geographic

def test_load_elevation_from_geotiff_with_changed_no_data():
    test_file_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "test_data.tif"
    )
    with open(test_file_path, "rb") as fd:
        elevation = read_from_geotiff(fd, no_data_value=-9999.0)
        assert elevation.header == \
               ElevationHeader(
                       col_count=2,
                       row_count=2,
                       x_ll_corner=587069.5669010617,
                       y_ll_corner=4505948.133973204,
                       cell_size=2.0,
                       nodata_value=-9999.0,
                       crs=CRS({"init": "EPSG:32618"})
               )
        assert elevation.data == \
               [
                       [-9999.0, 5.72777795791626],
                       [5.713333606719971, 5.742222309112549],
               ]
