import numpy

from usl_lib.chunkers import raster_chunkers


def test_split_raster_into_chunks():
    # Produce a matrix like:
    # [[ 0,  1,  2,  3]
    #  [ 4,  5,  6,  7]
    #  [ 8,  9, 10, 11]]
    array = numpy.array(range(12)).reshape((3, 4))

    chunks = list(raster_chunkers.split_raster_into_chunks(2, array))

    expected_chunks = [
        (0, 0, numpy.array([[0, 1], [4, 5]])),
        (1, 0, numpy.array([[2, 3], [6, 7]])),
        (0, 1, numpy.array([[8, 9]])),
        (1, 1, numpy.array([[10, 11]])),
    ]
    # We need assert_array_equal, so we can't just assert chunks == expected_chunks.
    assert len(chunks) == len(expected_chunks)
    for chunk, expected_chunk in zip(chunks, expected_chunks):
        # Check the (x, y) index.
        assert chunk[:1] == expected_chunk[:1]
        # Check the chunk array.
        numpy.testing.assert_array_equal(chunk[2], expected_chunk[2])
