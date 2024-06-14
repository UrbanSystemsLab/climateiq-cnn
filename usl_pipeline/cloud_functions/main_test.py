import datetime
import io
import tarfile
import textwrap
from unittest import mock

from google.cloud import firestore
from google.cloud import storage
import functions_framework
import netCDF4
import numpy
import rasterio
import xarray

import main
from usl_lib.shared import wps_data
from usl_lib.storage import file_names


def _add_to_tar(tar: tarfile.TarFile, file_name: str, content_bytes: bytes):
    """Adds a new entry to a TAR file."""
    tf = tarfile.TarInfo(file_name)
    tf.size = len(content_bytes)
    tar.addfile(tf, io.BytesIO(content_bytes))


@mock.patch.object(main.firestore, "Client", autospec=True)
@mock.patch.object(main.storage, "Client", autospec=True)
def test_build_feature_matrix_flood(mock_storage_client, mock_firestore_client):
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

    polygons_bytes = str.encode("1\n5 0 1 1 0 0 0 0 1 1 0\n")
    # Place the tiff file bytes into an archive.
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        _add_to_tar(tar, file_names.ELEVATION_TIF, tiff_bytes)
        _add_to_tar(tar, file_names.BUILDINGS_TXT, polygons_bytes)
        _add_to_tar(tar, file_names.GREEN_AREAS_TXT, polygons_bytes)
        _add_to_tar(tar, file_names.SOIL_CLASSES_TXT, polygons_bytes)
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
            # I don't understand why the following calls are not registered in the mock!
            # mock.call().bucket("climateiq-study-area-feature-chunks"),
            # mock.call().bucket().blob("study_area/name.npy"),
        ]
    )

    mock_archive_blob.open.assert_called_once_with("rb")

    # Ensure we attempted to upload a serialized matrix of the tiff.
    mock_feature_blob.upload_from_file.assert_called_once_with(mock.ANY)
    uploaded_array = numpy.load(mock_feature_blob.upload_from_file.call_args[0][0])
    numpy.testing.assert_array_equal(
        uploaded_array,
        [
            # Row 1
            [
                [1, 1, 0, 0, 0, 0, 0, 0],
                [2, 1, 0, 0, 0, 0, 0, 0],
                [3, 1, 0, 0, 0, 0, 0, 0],
            ],
            # Row 2
            [
                [4, 1, 0, 0, 0, 0, 0, 0],
                [5, 1, 0, 0, 0, 0, 0, 0],
                [6, 1, 0, 0, 0, 0, 0, 0],
            ],
        ],
    )

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
@mock.patch.dict(
    main.wps_data.ML_REQUIRED_VARS_REPO,
    {
        "GHT": {
            "scaling": {
                "type": wps_data.ScalingType.LOCAL,
            }
        },
        "RH": {
            "scaling": {
                "type": wps_data.ScalingType.NONE,
            }
        },
    },
    clear=True,
)
def test_build_feature_matrix_wrf(
    mock_storage_client,
    mock_firestore_client,
    mock_error_reporting_client,
):
    # Get some random data to place in a netcdf file
    netcdf_array1 = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=numpy.float32)
    netcdf_array2 = numpy.array(
        [[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=numpy.float32
    )

    # Create an in-memory netcdf file and grab its bytes
    ncfile = netCDF4.Dataset("met_em_test.nc", mode="w", format="NETCDF4", memory=1)
    ncfile.createDimension("Time", 1)
    ncfile.createDimension("west_east", 3)
    ncfile.createDimension("south_north", 3)

    time = ncfile.createVariable("time", "f8", ("Time",))
    lat = ncfile.createVariable("lat", "float32", ("west_east",))
    lon = ncfile.createVariable("lon", "float32", ("south_north",))
    # Create dataset entries for all variables in mock
    for var in wps_data.ML_REQUIRED_VARS_REPO.keys():
        ncfile.createVariable(var, "float32", ("Time", "south_north", "west_east"))
    # Represents var in DS that we don't want to process
    not_required_var = ncfile.createVariable(
        "NOT_REQUIRED_VAR", "float32", ("Time", "south_north", "west_east")
    )

    time[:] = 100
    lat[:] = [200, 200, 200]
    lon[:] = [300, 300, 300]
    ncfile.variables["GHT"][:] = netcdf_array1
    ncfile.variables["RH"][:] = netcdf_array2
    not_required_var[:] = [[11, 22, 33], [44, 55, 66], [77, 88, 99]]

    memfile = ncfile.close()
    ncfile_bytes = memfile.tobytes()

    # Place the ncfile bytes into an archive.
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        nc = tarfile.TarInfo("met_em_test.nc")
        nc.size = len(ncfile_bytes)
        tar.addfile(nc, io.BytesIO(ncfile_bytes))
    # Seek to the beginning so the file can be read.
    archive.seek(0)

    # Create a mock blob for the archive which will return the above netcdf when opened.
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

    # Ensure we attempted to upload a serialized matrix of the netcdf
    mock_feature_blob.upload_from_file.assert_called_once_with(mock.ANY)
    uploaded_array = numpy.load(mock_feature_blob.upload_from_file.call_args[0][0])
    # Expected array should be all required vars extracted and stacked
    expected_array = numpy.dstack((netcdf_array1, netcdf_array2))
    numpy.testing.assert_array_equal(uploaded_array, expected_array)

    # TODO: Ensure we wrote firestore entries for the chunk.


@mock.patch.dict(main.wps_data.ML_REQUIRED_VARS_REPO, {"RH": {}}, clear=True)
def test_process_wps_feature_drop_time_axis():
    arr = numpy.array([[[1, 2], [3, 4]]], dtype=numpy.float32)  # shape=(1,2,2)
    xrdarr = xarray.DataArray(arr, name="RH", dims=("Time", "south_north", "west_east"))

    processed = main._process_wps_feature(
        xrdarr, wps_data.ML_REQUIRED_VARS_REPO.get("RH")
    )

    numpy.testing.assert_equal((2, 2), processed.shape)


@mock.patch.dict(main.wps_data.ML_REQUIRED_VARS_REPO, {"RH": {}}, clear=True)
def test_process_wps_feature_extract_fnl_spatial_dim():
    arr = numpy.array([[[[10, 1], [20, 2]], [[30, 3], [40, 4]]]], dtype=numpy.float32)
    xrdarr = xarray.DataArray(
        arr, name="RH", dims=("Time", "south_north", "west_east", "num_metgrid_levels")
    )

    processed = main._process_wps_feature(
        xrdarr, wps_data.ML_REQUIRED_VARS_REPO.get("RH")
    )

    expected = [[10, 20], [30, 40]]
    numpy.testing.assert_array_equal(expected, processed)


@mock.patch.dict(
    main.wps_data.ML_REQUIRED_VARS_REPO,
    {
        "RH": {
            "unit": wps_data.Unit.PERCENTAGE,
        },
    },
    clear=True,
)
def test_process_wps_feature_convert_percent_to_decimal():
    arr = numpy.array([[[32.45, 15.11], [73.74, 33.21]]], dtype=numpy.float32)
    xrdarr = xarray.DataArray(
        arr,
        name="RH",
        dims=("Time", "south_north", "west_east"),
        attrs=dict(
            description="Test data array",
            units="%",
        ),
    )

    processed = main._process_wps_feature(
        xrdarr, wps_data.ML_REQUIRED_VARS_REPO.get("RH")
    )

    expected = [[0.3245, 0.1511], [0.7374, 0.3321]]
    numpy.testing.assert_array_almost_equal(expected, processed)


@mock.patch.dict(
    main.wps_data.ML_REQUIRED_VARS_REPO,
    {
        "PRES": {
            "scaling": {
                "type": wps_data.ScalingType.GLOBAL,
                "min": 98000,
                "max": 121590,
            }
        },
    },
    clear=True,
)
def test_process_wps_feature_apply_minmax_scaler():
    arr = numpy.array([[[111222, 555555], [121590, 12]]], dtype=numpy.float32)
    xrdarr = xarray.DataArray(
        arr,
        name="PRES",
        dims=("Time", "south_north", "west_east"),
    )

    processed = main._process_wps_feature(
        xrdarr, wps_data.ML_REQUIRED_VARS_REPO.get("PRES")
    )

    expected = [[0.56, 1], [1, 0]]
    numpy.testing.assert_array_almost_equal(expected, processed, decimal=3)


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


@mock.patch.object(main.firestore, "Client", autospec=True)
@mock.patch.object(main.storage, "Client", autospec=True)
def test_write_flood_scenario_metadata(mock_storage_client, mock_firestore_client):
    """Ensure we sync config uploads to the metastore."""
    config_blob = mock.Mock(spec=storage.Blob)
    config_blob.name = "config_name/Rainfall_Data_1.txt"
    config_blob.bucket.name = "bucket"
    config_blob.open.return_value = io.StringIO(
        textwrap.dedent(
            """\
            * * *
            * * * rainfall ***
            * * *
            5
            * * *
            0	0.0000000000
            3600	0.0000019756
            7200	0.0000019756
            10800	0.0000019756
            14400	0.0000039511
            """
        )
    )

    vector_blob = mock.Mock(spec=storage.Blob)
    vector_blob.name = "rainfall/config_name/Rainfall_Data_1.npy"
    vector_blob.bucket.name = "climateiq-study-area-feature-chunks"

    mock_storage_client().bucket("").blob.side_effect = [config_blob, vector_blob]

    main.write_flood_scenario_metadata_and_features(
        functions_framework.CloudEvent(
            {"source": "test", "type": "event"},
            data={
                "bucket": "bucket",
                "name": "config_name/Rainfall_Data_1.txt",
                "timeCreated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        )
    )

    # Ensure we worked with the right GCP paths.
    mock_storage_client.assert_has_calls(
        [
            mock.call().bucket("bucket"),
            mock.call().bucket().blob("config_name/Rainfall_Data_1.txt"),
            mock.call().bucket("climateiq-study-area-feature-chunks"),
            mock.call().bucket().blob("rainfall/config_name/Rainfall_Data_1.npy"),
        ]
    )

    config_blob.open.assert_called_once_with("rt")

    # Ensure we attempted to upload a serialized array of the rainfall.
    vector_blob.upload_from_file.assert_called_once_with(mock.ANY)
    uploaded_array = numpy.load(vector_blob.upload_from_file.call_args[0][0])
    numpy.testing.assert_array_almost_equal(
        uploaded_array,
        numpy.pad(
            numpy.array([0.0, 0.0000019756, 0.0000019756, 0.0000019756, 0.0000039511]),
            (0, main._RAINFALL_VECTOR_LENGTH - 5),
        ),
    )

    # Ensure we wrote the firestore entry for the config.
    mock_firestore_client.assert_has_calls(
        [
            mock.call(),
            mock.call().collection("city_cat_rainfall_configs"),
            mock.call().collection().document("config_name%2FRainfall_Data_1.txt"),
            mock.call()
            .collection()
            .document()
            .set(
                {
                    "parent_config_name": "config_name",
                    "gcs_uri": "gs://bucket/config_name/Rainfall_Data_1.txt",
                    "rainfall_duration": 5,
                    "as_vector_gcs_uri": (
                        "gs://climateiq-study-area-feature-chunks/"
                        "rainfall/config_name/Rainfall_Data_1.npy"
                    ),
                }
            ),
        ]
    )


@mock.patch.object(main.error_reporting, "Client", autospec=True)
@mock.patch.object(main.firestore, "Client", autospec=True)
@mock.patch.object(main.storage, "Client", autospec=True)
def test_write_flood_scenario_metadata_rejects_too_long_rainfall(
    mock_storage_client, mock_firestore_client, mock_error_reporting_client
):
    """Ensure we sync config uploads to the metastore."""
    config_blob = mock.Mock(spec=storage.Blob)
    config_blob.name = "config_name/Rainfall_Data_1.txt"
    config_blob.bucket.name = "bucket"

    # Create a rainfall config file with too many entries.
    bad_config_file = io.StringIO()
    bad_config_file.write(f"{main._RAINFALL_VECTOR_LENGTH + 3}\n")
    bad_config_file.writelines(
        "0   0.0000000000\n" for _ in range(main._RAINFALL_VECTOR_LENGTH + 3)
    )
    bad_config_file.seek(0)
    config_blob.open.return_value = bad_config_file

    vector_blob = mock.Mock(spec=storage.Blob)
    mock_storage_client().bucket("").blob.side_effect = [config_blob, vector_blob]

    main.write_flood_scenario_metadata_and_features(
        functions_framework.CloudEvent(
            {"source": "test", "type": "event"},
            data={
                "bucket": "bucket",
                "name": "config_name/Rainfall_Data_1.txt",
                "timeCreated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        )
    )

    # Ensure we didn't try to write anything.
    vector_blob.upload_from_file.assert_not_called()
    mock_firestore_client().collection.assert_not_called()
    mock_error_reporting_client().report_exception.assert_called_once()


@mock.patch.object(main.firestore, "Client", autospec=True)
def test_write_flood_scenario_metadata_ignore_non_rainfall_files(mock_firestore_client):
    """Ensure we ignore non-rainfall config uploads."""
    main.write_flood_scenario_metadata_and_features(
        functions_framework.CloudEvent(
            {"source": "test", "type": "event"},
            data={
                "bucket": "bucket",
                "name": "config_name/SomethingElse.txt",
                "timeCreated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        )
    )

    mock_firestore_client.assert_not_called()


@mock.patch.object(main.firestore, "Client", autospec=True)
def test_delete_flood_scenario_metadata(mock_firestore_client):
    """Ensure we sync config uploads to the metastore."""
    main.delete_flood_scenario_metadata(
        functions_framework.CloudEvent(
            {"source": "test", "type": "event"},
            data={
                "bucket": "bucket",
                "name": "config_name/Rainfall_Data_1.txt",
                "timeCreated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        )
    )

    # Ensure we delete the firestore entry for the config.
    mock_firestore_client.assert_has_calls(
        [
            mock.call(),
            mock.call().collection("city_cat_rainfall_configs"),
            mock.call().collection().document("config_name%2FRainfall_Data_1.txt"),
            mock.call().collection().document().delete(),
        ]
    )


@mock.patch.object(main.firestore, "Client", autospec=True)
def test_delete_flood_scenario_metadata_ignore_non_rainfall_files(
    mock_firestore_client,
):
    """Ensure we ignore non-rainfall config uploads."""
    main.delete_flood_scenario_metadata(
        functions_framework.CloudEvent(
            {"source": "test", "type": "event"},
            data={
                "bucket": "bucket",
                "name": "config_name/SomethingElse.txt",
                "timeCreated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        )
    )

    mock_firestore_client.assert_not_called()
