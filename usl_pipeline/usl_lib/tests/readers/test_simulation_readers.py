import io
import textwrap

import numpy

from usl_lib.readers import simulation_readers
from usl_lib.shared import geo_data


def test_read_city_cat_result_as_raster():
    result_file = io.StringIO(
        textwrap.dedent(
            """\
            XCen YCen Depth Vx Vy T_300.000_sec
            0 1 0.001 0.000 0.000
            1 2 0.002 0.000 0.000
            2 3 0.003 0.000 0.000
            2 1 0.004 0.000 0.000
            """
        )
    )
    header = geo_data.ElevationHeader(
        col_count=3,
        row_count=4,
        x_ll_corner=0,
        y_ll_corner=0,
        cell_size=1.0,
        nodata_value=-9999.0,
        crs=None,
    )

    raster = simulation_readers.read_city_cat_result_as_raster(result_file, header)
    numpy.testing.assert_array_equal(
        raster,
        numpy.array(
            [
                [0, 0, 0],
                [0, 0, 0.003],
                [0, 0.002, 0],
                [0.001, 0, 0.004],
            ],
            dtype=numpy.float32,
        ),
        strict=True,
    )
