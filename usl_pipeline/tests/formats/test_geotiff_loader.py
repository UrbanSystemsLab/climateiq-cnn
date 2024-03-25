import unittest
from usl_pipeline.formats import geotiff_loader
from rasterio import CRS


class GeodiffLoaderTest(unittest.TestCase):

    def test_load_elevation_from_geotiff(self):
        (header, data, crs) = geotiff_loader.load_elevation_from_geotiff(
            "test_data.tif", load_data=True
        )
        self.assertDictEqual(
            header,
            {
                "ncols": 2,
                "nrows": 2,
                "xllcorner": 587069.5669010617,
                "yllcorner": 4505948.133973204,
                "cellsize": 2.0,
                "nodata_value": -9999.0,
            },
        )
        self.assertListEqual(
            data,
            [
                [5.648889064788818, 5.72777795791626],
                [5.713333606719971, 5.742222309112549],
            ],
        )
        self.assertEqual(crs, CRS({"init": "EPSG:32618"}))
        self.assertFalse(crs.is_geographic)


if __name__ == "__main__":
    unittest.main()
