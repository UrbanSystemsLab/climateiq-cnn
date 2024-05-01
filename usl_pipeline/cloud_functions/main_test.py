import datetime
import io
import tarfile
from unittest import mock

from google.cloud import firestore
import functions_framework
import numpy
import rasterio

import main


@mock.patch.object(main.firestore, "Client", autospec=True)
@mock.patch.object(main.storage, "Client", autospec=True)
def test_build_feature_matrix(mock_storage_client, mock_firestore_client):
    # Get some random data to place in a tiff file.
    height = 2
    width = 3
    tiff_array = numpy.array([[1, 2, 3], [4, 5, 6]], dtype=numpy.uint8)

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
        tf = tarfile.TarInfo("elevation.tif")
        tf.size = len(tiff_bytes)
        tar.addfile(tf, io.BytesIO(tiff_bytes))
    # Seek to the beginning so the file can be read.
    archive.seek(0)

    # Create a mock blob for the archive which will return the above tiff when opened.
    mock_archive_blob = mock.MagicMock()
    mock_archive_blob.name = "study_area/name.tar"
    mock_archive_blob.bucket.name = "bucket"
    mock_archive_blob.open.return_value = archive

    # Create a mock blob for feature matrix we will upload.
    mock_feature_blob = mock.MagicMock()
    mock_feature_blob.name = "study_area/name.npy"
    mock_feature_blob.bucket.name = "climateiq-study-area-feature-chunks"

    # Return the mock blobs.
    mock_storage_client().bucket("").blob.side_effect = [
        mock_archive_blob,
        mock_feature_blob,
    ]
    mock_storage_client.reset_mock()

    # Simulate empty elevation min & max on the study area.
    mock_firestore_client().collection().document().get().get.side_effect = KeyError

    main.build_feature_matrix(
        functions_framework.CloudEvent(
            {"source": "test", "type": "event"},
            data={
                "bucket": "bucket",
                "name": "study_area/name.tar",
                "timeCreated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        )
    )

    # Ensure we worked with the right GCP paths.
    mock_storage_client.assert_has_calls(
        [
            mock.call().bucket("bucket"),
            mock.call().bucket().blob("study_area/name.tar"),
            mock.call().bucket("climateiq-study-area-feature-chunks"),
            mock.call().bucket().blob("study_area/name.npy"),
        ]
    )

    mock_archive_blob.open.assert_called_once_with("rb")

    # Ensure we attempted to upload a serialized matrix of the tiff.
    mock_feature_blob.upload_from_file.assert_called_once_with(mock.ANY)
    uploaded_array = numpy.load(mock_feature_blob.upload_from_file.call_args[0][0])
    numpy.testing.assert_array_equal(uploaded_array, tiff_array)

    # Ensure we wrote firestore entries for the chunk.
    mock_firestore_client.assert_has_calls(
        [
            mock.call(),
            mock.call().collection("study_areas"),
            mock.call().collection().document("study_area"),
            mock.call().collection().document().collection("chunks"),
            mock.call().collection().document().collection().document("name"),
            mock.call()
            .collection()
            .document()
            .collection()
            .document()
            .set(
                {
                    "archive_path": "gs://bucket/study_area/name.tar",
                    "feature_matrix_path": (
                        "gs://climateiq-study-area-feature-chunks/study_area/name.npy"
                    ),
                    "error": firestore.DELETE_FIELD,
                },
                merge=True,
            ),
        ]
    )
    # Ensure we set the elevation min & max
    mock_firestore_client().transaction().update.assert_called_once_with(
        mock_firestore_client().collection().document(),
        {"elevation_min": 1, "elevation_max": 6},
    )


@mock.patch.object(main.error_reporting, "Client", autospec=True)
@mock.patch.object(main.firestore, "Client", autospec=True)
@mock.patch.object(main.storage, "Client", autospec=True)
def test_build_feature_matrix_errors(
    mock_storage_client,
    mock_firestore_client,
    mock_error_reporting_client,
):
    """Ensure errors are reported appropriately."""
    # Cause the cloud function to fail.
    mock_storage_client.side_effect = RuntimeError("oh no!")

    # Grab the original datetime module before we mock it so we can use it below.
    orig_datetime = datetime.datetime
    # We can't mock datetime.now directly, we can only mock datetime
    # https://docs.python.org/3/library/unittest.mock-examples.html#partial-mocking
    with mock.patch.object(main.datetime, "datetime") as mock_datetime:
        # Use the same value for `now` as the cloud function states as its timeCreated.
        mock_datetime.now.return_value = orig_datetime(2012, 12, 21)
        # Use the original functions rather than mocks.
        mock_datetime.fromisoformat.side_effect = orig_datetime.fromisoformat
        mock_datetime.side_effect = orig_datetime

        main.build_feature_matrix(
            functions_framework.CloudEvent(
                {"source": "test", "type": "event"},
                data={
                    "bucket": "bucket",
                    "name": "study_area/name.tar",
                    "timeCreated": "2012-12-21T01:02:00",
                    "id": "function-id",
                },
            )
        )

    # Ensure we called the error reporter.
    mock_error_reporting_client().report_exception.assert_called_once()

    # Ensure we wrote the error to firestore.
    mock_firestore_client.assert_has_calls(
        [
            mock.call(),
            mock.call().collection("study_areas"),
            mock.call().collection().document("study_area"),
            mock.call().collection().document().collection("chunks"),
            mock.call().collection().document().collection().document("name"),
            mock.call()
            .collection()
            .document()
            .collection()
            .document()
            .set(
                {"error": "oh no!"},
                merge=True,
            ),
        ]
    )


@mock.patch.object(main.error_reporting, "Client", autospec=True)
@mock.patch.object(main.firestore, "Client", autospec=True)
@mock.patch.object(main.storage, "Client", autospec=True)
def test_build_feature_matrix_retries(
    mock_storage_client,
    mock_firestore_client,
    mock_error_reporting_client,
):
    """Ensure we halt on old retries."""
    # Grab the original datetime module before we mock it so we can use it below.
    orig_datetime = datetime.datetime
    # We can't mock datetime.now directly, we can only mock datetime
    # https://docs.python.org/3/library/unittest.mock-examples.html#partial-mocking
    with mock.patch.object(main.datetime, "datetime") as mock_datetime:
        # Use a value for `now` long after the cloud function states as its timeCreated.
        mock_datetime.now.return_value = orig_datetime(2030, 1, 1)
        # Use the original functions rather than mocks.
        mock_datetime.fromisoformat.side_effect = orig_datetime.fromisoformat
        mock_datetime.side_effect = orig_datetime

        main.build_feature_matrix(
            functions_framework.CloudEvent(
                {"source": "test", "type": "event"},
                data={
                    "bucket": "bucket",
                    "name": "study_area/name.tar",
                    "timeCreated": "2012-12-21T01:02:00",
                    "id": "function-id",
                },
            )
        )

    # Ensure we didn't try to do actual cloud function work.
    mock_storage_client.assert_not_called()


@mock.patch.object(main.firestore, "Client", autospec=True)
@mock.patch.object(main.storage, "Client", autospec=True)
def test_write_study_area_metadata(mock_storage_client, mock_firestore_client):
    # Get some random data to place in a tiff file.
    height = 2
    width = 3
    tiff_array = numpy.array([[1, 2, 3], [4, 5, 6]], dtype=numpy.uint8)

    # Build an in-memory tiff file and grab its bytes.
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            width=width,
            height=height,
            count=1,
            dtype=rasterio.uint8,
            crs=rasterio.CRS.from_epsg(32618),
        ) as raster:
            raster.write(tiff_array.astype(rasterio.uint8), 1)
        tiff_bytes = io.BytesIO(memfile.read())

    # Return the bytes above form the mock storage client.
    mock_storage_client().bucket("").blob("").open.return_value = tiff_bytes

    main.write_study_area_metadata(
        functions_framework.CloudEvent(
            {"source": "test", "type": "event"},
            data={
                "bucket": "bucket",
                "name": "study_area/elevation.tif",
                "timeCreated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        )
    )

    # Ensure we worked with the right GCP paths.
    mock_storage_client.assert_has_calls(
        [
            mock.call().bucket("bucket"),
            mock.call().bucket().blob("study_area/elevation.tif"),
        ]
    )
    mock_storage_client().bucket("").blob("").open.assert_called_once_with("rb")

    # Ensure we wrote the firestore entry for the study area.
    mock_firestore_client.assert_has_calls(
        [
            mock.call(),
            mock.call().collection("study_areas"),
            mock.call().collection().document("study_area"),
            mock.call()
            .collection()
            .document()
            .set(
                {
                    "col_count": 3,
                    "row_count": 2,
                    "x_ll_corner": 0.0,
                    "y_ll_corner": 2.0,
                    "cell_size": 1.0,
                    "crs": "EPSG:32618",
                }
            ),
        ]
    )
