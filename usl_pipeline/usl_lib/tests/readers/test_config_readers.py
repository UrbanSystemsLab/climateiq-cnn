import io
import textwrap

import pytest

from usl_lib.readers import config_readers


def test_read_rainfall_amounts():
    config = io.StringIO(
        textwrap.dedent(
            """\
            * * *
            * * * rainfall ***
            * * *
            13
            * * *
            0	0.0000000000
            3600	0.0000019756
            7200	0.0000019756
            10800	0.0000019756
            14400	0.0000039511
            """
        )
    )

    entries = config_readers.read_rainfall_amounts(config)
    assert entries == pytest.approx(
        [0, 0.0000019756, 0.0000019756, 0.0000019756, 0.0000039511]
    )
