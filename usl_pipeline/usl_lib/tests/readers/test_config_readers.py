import io
import textwrap

import pytest
import configparser

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


def test_read_key_value_pairs_valid_pairs():
    config_content = io.StringIO(
        textwrap.dedent(
            """\
            key1 = value1
            key2 = value2
            key3=no spaces around delimiter
            """
        )
    )
    result = config_readers.read_key_value_pairs(config_content)

    expected = {
        "key1": "value1",
        "key2": "value2",
        "key3": "no spaces around delimiter",
    }
    assert result == expected


def test_read_key_value_pairs_existing_header():
    config_content = io.StringIO(
        textwrap.dedent(
            """\
            [MY_HEADER]
            key1 = value1
            key2 = value2
            """
        )
    )
    result = config_readers.read_key_value_pairs(config_content)

    expected = {"key1": "value1", "key2": "value2"}
    assert result == expected


def test_read_key_value_pairs_empty_file():
    """Test with an empty file."""
    config = io.StringIO("")

    result = config_readers.read_key_value_pairs(config)
    assert result == {}


def test_read_key_value_pairs_comments_and_empty_lines():
    """Test with comments and empty lines."""
    config = io.StringIO(
        textwrap.dedent(
            """\
            # This is a comment

            key1 = value1

            # Another comment
            key2 = value2
            """
        )
    )

    result = config_readers.read_key_value_pairs(config)
    expected = {"key1": "value1", "key2": "value2"}
    assert result == expected


def test_read_key_value_pairs_invalid_lines():
    """Test with invalid line formats."""
    config = io.StringIO(
        textwrap.dedent(
            """\
            key1 = value1
            invalid_line
            key2= value2
            """
        )
    )

    with pytest.raises(configparser.ParsingError):
        config_readers.read_key_value_pairs(config)
