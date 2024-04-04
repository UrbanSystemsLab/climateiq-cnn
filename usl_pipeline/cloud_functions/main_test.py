import io
import tarfile
from unittest import mock

import functions_framework
import numpy
import rasterio

import main


@mock.patch.object(main.storage, "Client", autospec=True)
def test_build_feature_matrix(mock_client):
    # Get some random data to place in a tiff file.
    height = 5
    width = 3
    tiff_array = numpy.random.randint(
        low=0, high=16, size=(height, width), dtype=numpy.uint8
    )

    # Build an in-memory tiff file and grab its bytes.
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            width=width,
            height=height,
            count=1,
            dtype=rasterio.uint8,
        ) as raster:
            raster.write(tiff_array.astype(rasterio.uint8), 1)
        tiff_bytes = memfile.read()

    # Place the tiff file bytes into an archive.
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        tf = tarfile.TarInfo("elevation.tiff")
        tf.size = len(tiff_bytes)
        tar.addfile(tf, io.BytesIO(tiff_bytes))
    # Seek to the beginning so the file can be read.
    archive.seek(0)

    # Set the archive to be returned by the mock GCS client.
    mock_client().bucket("").blob("").open.return_value = archive
    mock_client.reset_mock()

    main._build_feature_matrix(
        functions_framework.CloudEvent(
            {"source": "test", "type": "event"},
            data={"bucket": "bucket", "name": "map/name.tar"},
        )
    )

    # Ensure we worked with the right GCP paths.
    mock_client.assert_has_calls(
        [
            mock.call().bucket("bucket"),
            mock.call().bucket().blob("map/name.tar"),
            mock.call().bucket().blob().open("rb"),
            mock.call().bucket("climateiq-map-feature-chunks"),
            mock.call().bucket().blob("map/name.npy"),
            mock.call().bucket().blob().upload_from_file(mock.ANY),
        ]
    )

    # Ensure we attempted to upload a serialized matrix of the tiff.
    mock_client().bucket("").blob("").upload_from_file.assert_called_once()
    uploaded_array = numpy.load(
        mock_client().bucket("").blob("").upload_from_file.call_args[0][0]
    )
    numpy.testing.assert_array_equal(uploaded_array, tiff_array)
