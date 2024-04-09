import fiona


def test_read_polygons_from_shape_file():
    # Prepare temporary shape file
    with fiona.open(
        "test.shp", 'w', crs=CRS.from_epsg(3857), driver='ESRI Shapefile',
        schema=yourschema
    ) as output:
            shape = Polygon()
            prop = {'soil_type': 1}
            output.write({'geometry': [shape], 'properties': prop})
