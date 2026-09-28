import numpy
from numpy import testing
from shapely import geometry

from usl_lib.shared import geo_data
from usl_lib.storage import metastore
from usl_lib.transformers import feature_raster_transformers


def bbox_polygon(x1: float, y1: float, x2: float, y2: float) -> geometry.Polygon:
    return geometry.Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)])


def test_transform_to_feature_raster_layers():
    infiltration_configuration = geo_data.DEFAULT_INFILTRATION_CONFIGURATION
    max_hydraulic_conductivity = 11.78
    max_wetting_front_suction_head = 20.88
    max_effective_porosity = 0.486
    max_effective_saturation = 0.99

    header = geo_data.ElevationHeader(
        col_count=3,
        row_count=4,
        x_ll_corner=0.0,
        y_ll_corner=0.0,
        cell_size=1.0,
        nodata_value=-9999.0,
    )
    data = numpy.array(
        [
            [0.0, 1.0, 2.0],
            [3.0, 4.0, -9999.0],
            [6.0, 7.0, -9999.0],
            [9.0, 10.0, 11.0],
        ]
    )
    elevation = geo_data.Elevation(header=header, data=data)
    boundaries = [
        (bbox_polygon(1, 0, 3, 4), 1),  # exclude the first column (row:col=*:0)
    ]
    buildings = [
        (bbox_polygon(0, 3, 3, 4), 1),  # this is the first row (row:col=0:*)
    ]
    green_areas = [
        (bbox_polygon(0, 0, 3, 2), 1),  # these are last two rows (row:col=2,3:*)
    ]
    soil_classes = [
        (bbox_polygon(1, 1, 2, 2), 3),  # row:col=2:1
        (bbox_polygon(1, 0, 2, 1), 4),  # row:col=3:1
        (bbox_polygon(2, 0, 3, 1), 11),  # row:col=3:2
    ]
    feature_matrix = feature_raster_transformers.transform_to_feature_raster_layers(
        elevation,
        boundaries,
        buildings,
        green_areas,
        soil_classes,
        infiltration_configuration,
    )

    # Checking the first column (should be NODATA since outside the boundaries,
    # including the slope channel -- slope respects the same boundary mask as
    # every other channel, even though it's computed from the raw DEM)
    testing.assert_array_equal(
        feature_matrix[:, 0],
        numpy.array(
            [
                [-9999.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9999.0],
                [-9999.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9999.0],
                [-9999.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9999.0],
                [-9999.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9999.0],
            ],
            dtype=numpy.float32,
        ),
        strict=True,
    )

    # Checking the remaining first row (should be elevation/mask present + buildings=1)
    # Cell (0,1) has valid slope (flat corner, no NoData neighbours); (0,2) is
    # NaN-contaminated by the NoData cell at (1,2) and (2,2) via numpy.gradient,
    # so it's correctly folded into NoData by compute_slope_feature itself.
    testing.assert_array_almost_equal(
        feature_matrix[0, 1:],
        numpy.array(
            [
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 316.227783],
                [2.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9999.0],
            ],
            dtype=numpy.float32,
        ),
    )

    # Checking cells row:col={1:2},{2:2} (should be NODATA as in original elevation)
    testing.assert_array_equal(
        feature_matrix[1:3, 2],
        numpy.array(
            [
                [-9999.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9999.0],
                [-9999.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9999.0],
            ],
            dtype=numpy.float32,
        ),
        strict=True,
    )

    # Cell row:col=1:1 has just elevation (with mask=1), no buildings, no greens.
    # Its slope is NaN-contaminated (adjacent to NoData at (1,2)) -> NoData.
    testing.assert_array_equal(
        feature_matrix[1][1],
        numpy.array(
            [4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9999.0], dtype=numpy.float32
        ),
        strict=True,
    )

    # Checking remaining 3 soil-related cells (all adjacent to a NoData cell,
    # so slope is NoData for each):
    # Cell row:col=2:1, soil class=3
    testing.assert_array_almost_equal(
        feature_matrix[2][1],
        numpy.array(
            [
                7.0,
                1.0,
                0.0,
                1.0,
                0.1 / max_hydraulic_conductivity,
                20.88 / max_wetting_front_suction_head,
                0.309 / max_effective_porosity,
                0.5 / max_effective_saturation,
                -9999.0,
            ],
            dtype=numpy.float32,
        ),
    )
    # Cell row:col=3:1, soil class=4 -- not adjacent to any NoData cell, so
    # slope is a real computed value here.
    testing.assert_array_almost_equal(
        feature_matrix[3][1],
        numpy.array(
            [
                10.0,
                1.0,
                0.0,
                1.0,
                0.34 / max_hydraulic_conductivity,
                8.89 / max_wetting_front_suction_head,
                0.434 / max_effective_porosity,
                0.5 / max_effective_saturation,
                316.227783,
            ],
            dtype=numpy.float32,
        ),
    )
    # Cell row:col=3:2, soil class=11
    testing.assert_array_almost_equal(
        feature_matrix[3][2],
        numpy.array(
            [
                11.0,
                1.0,
                0.0,
                1.0,
                1.09 / max_hydraulic_conductivity,
                11.01 / max_wetting_front_suction_head,
                0.412 / max_effective_porosity,
                0.99 / max_effective_saturation,
                -9999.0,
            ],
            dtype=numpy.float32,
        ),
    )


def test_transform_to_feature_raster_layers_no_polygons():
    boundaries = None
    feature_matrix = feature_raster_transformers.transform_to_feature_raster_layers(
        geo_data.Elevation(
            header=geo_data.ElevationHeader(
                col_count=2,
                row_count=2,
                x_ll_corner=0.0,
                y_ll_corner=0.0,
                cell_size=1.0,
                nodata_value=-9999.0,
            ),
            data=numpy.array(
                [
                    [0.0, 1.0],
                    [2.0, -9999.0],
                ]
            ),
        ),
        boundaries,
        [],
        [],
        [],
        geo_data.DEFAULT_INFILTRATION_CONFIGURATION,
    )

    # Checking results. (0,0) has a real computed slope since none of its
    # neighbours are NoData; the other three cells are all adjacent to the
    # NoData cell at (1,1), so their slope folds into NoData too.
    testing.assert_array_almost_equal(
        feature_matrix,
        numpy.array(
            [
                [
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 223.606797],
                    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9999.0],
                ],
                [
                    [2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9999.0],
                    [-9999.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9999.0],
                ],
            ],
            dtype=numpy.float32,
        ),
    )


def test_rescale_feature_matrix():
    feature_matrix = numpy.array(
        [
            [
                [-1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [3.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [-9999.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ],
        dtype=numpy.float32,
    )
    feature_raster_transformers.rescale_feature_matrix(
        feature_matrix,
        metastore.StudyArea(
            name="1",
            col_count=2,
            row_count=2,
            x_ll_corner=0,
            y_ll_corner=0,
            cell_size=1,
            elevation_min=-5,
            elevation_max=5,
        ),
    )
    testing.assert_array_equal(
        feature_matrix,
        numpy.array(
            [
                [
                    [0.4, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.7, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.8, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ],
            dtype=numpy.float32,
        ),
        strict=True,
    )
