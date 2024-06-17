import io
import numpy
import numpy.testing

from usl_lib.readers import hdf5_readers
from usl_lib.writers import hdf5_writers


def test_read_write_hdf5():
    layer1 = numpy.array([[1, 2], [3, 4]], dtype=numpy.float32)
    layer2 = numpy.array([[10, 20], [30, 40]], dtype=numpy.float32)

    buffer = io.BytesIO()
    hdf5_writers.write_to_hdf5({"ft1": layer1, "ft2": layer2}, buffer)

    buffer.seek(0)

    result = hdf5_readers.read_from_hdf5(buffer)
    numpy.testing.assert_array_equal(layer1, result["ft1"], strict=True)
    numpy.testing.assert_array_equal(layer2, result["ft2"], strict=True)
