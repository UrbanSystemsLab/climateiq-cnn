import datetime
import io
import pathlib
import tarfile
import textwrap
from unittest import mock

from google.api_core import exceptions
from google.cloud import firestore
from google.cloud import storage
import functions_framework
import netCDF4
import numpy
import pytest
import rasterio
import xarray

import main
from usl_lib.shared import geo_data, wps_data
from usl_lib.storage import file_names


def _add_to_tar(tar: tarfile.TarFile, file_name: str, content_bytes: bytes):
    """Adds a new entry to a TAR file."""
    tf = tarfile.TarInfo(file_name)
    tf.size = len(content_bytes)
    tar.addfile(tf, io.BytesIO(content_bytes))


@mock.patch.object(main.error_reporting, "Client", autospec=True)
@mock.patch.object(main.firestore, "Client", autospec=True)
@mock.patch.object(main.storage, "Client", autospec=True)
def test_build_feature_matrix_flood(mock_storage_client, mock_firestore_client, _):
    # Get some random data to place in a tiff file.
    height = 2
    width = 2
    tiff_array = numpy.array([[1, 2], [5, 6]], dtype=numpy.uint8)

    # Build an in-memory tiff file and grab its bytes.
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            width=width,
            height=height,
            count=1,
            dtype=rasterio.uint8,
            nodata=1,
        ) as raster:
            raster.write(tiff_array.astype(rasterio.uint8), 1)
        tiff_bytes = memfile.read()

    polygons_bytes = b"1\n5 0 1 1 0 0 0 0 1 1 0\n"
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
    mock_archive_blob.name = "study_area/chunk_1_2.tar"
    mock_archive_blob.bucket.name = "bucket"
    mock_archive_blob.open.return_value = archive

    # Create a mock blob for feature matrix we will upload.
    mock_feature_blob = mock.MagicMock()
    mock_feature_blob.name = "study_area/chunk_1_2.npy"
    mock_feature_blob.bucket.name = "climateiq-study-area-feature-chunks"

    # Return the mock blobs.
    mock_storage_client().bucket("").blob.side_effect = [
        mock_archive_blob,
        mock_feature_blob,
    ]
    mock_storage_client.reset_mock()

    mock_firestore_client().collection().document().get().to_dict.return_value = {
        "col_count": 10,
        "row_count": 20,
        "x_ll_corner": 0.0,
        "y_ll_corner": 0.0,
        "cell_size": 1.0,
    }

    # Simulate empty elevation min & max on the study area.
    mock_firestore_client().collection().document().get().get.side_effect = KeyError

    main.build_feature_matrix(
        functions_framework.CloudEvent(
            {"source": "test", "type": "event"},
            data={
                "bucket": "bucket",
                "name": "study_area/chunk_1_2.tar",
                "timeCreated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        )
    )

    # Ensure we worked with the right GCP paths.
    mock_storage_client.assert_has_calls(
        [
            mock.call().bucket("bucket"),
            mock.call().bucket().blob("study_area/chunk_1_2.tar"),
            mock.call().bucket("climateiq-study-area-feature-chunks"),
            mock.call().bucket().blob("study_area/chunk_1_2.npy"),
        ]
    )

    mock_archive_blob.open.assert_called_once_with("rb")

    # Ensure we attempted to upload a serialized matrix of the tiff.
    mock_feature_blob.upload_from_file.assert_called_once_with(mock.ANY)
    uploaded_array = numpy.load(mock_feature_blob.upload_from_file.call_args[0][0])
    numpy.testing.assert_array_equal(
        uploaded_array,
        numpy.array(
            [
                # Row 1
                [
                    [1, 0, 0, 0, 0, 0, 0, 0],
                    [2, 1, 0, 0, 0, 0, 0, 0],
                ],
                # Row 2
                [
                    [5, 1, 0, 0, 0, 0, 0, 0],
                    [6, 1, 0, 0, 0, 0, 0, 0],
                ],
            ],
            dtype=numpy.float32,
        ),
        strict=True,
    )

    # Ensure we wrote firestore entries for the chunk.
    mock_firestore_client.assert_has_calls(
        [
            mock.call(),
            mock.call().collection("study_areas"),
            mock.call().collection().document("study_area"),
            mock.call().collection().document().collection("chunks"),
            mock.call().collection().document().collection().document("chunk_1_2"),
            mock.call()
            .collection()
            .document()
            .collection()
            .document()
            .set(
                {
                    "raw_path": "gs://bucket/study_area/chunk_1_2.tar",
                    "feature_matrix_path": (
                        "gs://climateiq-study-area-feature-chunks"
                        "/study_area/chunk_1_2.npy"
                    ),
                    "error": firestore.DELETE_FIELD,
                    "x_index": 1,
                    "y_index": 2,
                },
                merge=True,
            ),
        ]
    )
    # Ensure we set the elevation min & max
    mock_firestore_client().transaction().update.assert_called_once_with(
        mock_firestore_client().collection().document(),
        {"elevation_min": 2, "elevation_max": 6},
    )
    mock_firestore_client().collection().document().update.assert_called_once_with(
        {"chunk_size": 2, "chunk_x_count": 5, "chunk_y_count": 10}
    ),


def test_build_feature_matrix_from_archive_empty_polygons():
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
            nodata=1,
        ) as raster:
            raster.write(tiff_array.astype(rasterio.uint8), 1)
        tiff_bytes = memfile.read()

    polygons_bytes = b"0\n"
    # Place the tiff file bytes into an archive.
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        _add_to_tar(tar, file_names.ELEVATION_TIF, tiff_bytes)
        _add_to_tar(tar, file_names.BUILDINGS_TXT, polygons_bytes)
        _add_to_tar(tar, file_names.GREEN_AREAS_TXT, polygons_bytes)
        _add_to_tar(tar, file_names.SOIL_CLASSES_TXT, polygons_bytes)
    # Seek to the beginning so the file can be read.
    archive.seek(0)

    feature_matrix, metadata = main._build_flood_feature_matrix_from_archive(archive)

    numpy.testing.assert_array_equal(
        feature_matrix,
        numpy.array(
            [
                # Row 1
                [
                    [1, 0, 0, 0, 0, 0, 0, 0],
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
            dtype=numpy.float32,
        ),
        strict=True,
    )

    # Ensure we set the elevation min & max
    assert metadata == main.FeatureMetadata(
        elevation_min=2, elevation_max=6, chunk_size=2
    )


def test_build_feature_matrix_from_archive_elevation_file_missing():
    polygons_bytes = b"0\n"
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as tar:
        _add_to_tar(tar, file_names.BUILDINGS_TXT, polygons_bytes)
        _add_to_tar(tar, file_names.GREEN_AREAS_TXT, polygons_bytes)
        _add_to_tar(tar, file_names.SOIL_CLASSES_TXT, polygons_bytes)
    # Seek to the beginning so the file can be read.
    archive.seek(0)

    try:
        main._build_flood_feature_matrix_from_archive(archive)
    except Exception as e:
        # Check expected error:
        assert isinstance(e, ValueError)
        assert str(e) == (
            "Some flood simulation data missing (see tar list: ['buildings.txt', "
            + "'green_areas.txt', 'soil_classes.txt'])"
        )
    else:
        raise Exception("Expected error never happened")


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
def test_build_feature_matrix_wrf(mock_storage_client, mock_firestore_client, _):
    # Get some random data to place in a netcdf file
    netcdf_array1 = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=numpy.float32)
    netcdf_array2 = numpy.array(
        [[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=numpy.float32
    )

    # Create an in-memory netcdf file and grab its bytes
    ncfile = netCDF4.Dataset("met_em.d03_test.nc", mode="w", format="NETCDF4", memory=1)
    ncfile.createDimension("Time", 1)
    ncfile.createDimension("west_east", 3)
    ncfile.createDimension("south_north", 3)

    # In WPS/WRF files, 'Times'->dimension and 'Time'->variable
    time = ncfile.createVariable("Times", "str", ("Time",))
    lat = ncfile.createVariable("lat", "float32", ("south_north",))
    lon = ncfile.createVariable("lon", "float32", ("west_east",))
    # Create dataset entries for all variables in mock
    for var in wps_data.ML_REQUIRED_VARS_REPO.keys():
        ncfile.createVariable(var, "float32", ("Time", "south_north", "west_east"))
    # Represents var in DS that we don't want to process
    not_required_var = ncfile.createVariable(
        "NOT_REQUIRED_VAR", "float32", ("Time", "south_north", "west_east")
    )

    time[0] = "2010-02-02_18:00:00"
    lat[:] = [200, 200, 200]
    lon[:] = [300, 300, 300]
    ncfile.variables["GHT"][:] = netcdf_array1
    ncfile.variables["RH"][:] = netcdf_array2
    not_required_var[:] = [[11, 22, 33], [44, 55, 66], [77, 88, 99]]

    memfile = ncfile.close()
    ncfile_bytes = memfile.tobytes()

    # Create a mock blob for the input file which will return the above netcdf when
    # opened.
    mock_archive_blob = mock.MagicMock()
    mock_archive_blob.name = "study_area/met_em.d03_test.nc"
    mock_archive_blob.bucket.name = "bucket"
    mock_archive_blob.open.return_value = io.BytesIO(ncfile_bytes)

    # Create a mock blob for feature matrix we will upload.
    mock_feature_blob = mock.MagicMock()
    mock_feature_blob.name = "study_area/met_em.d03_test.npy"
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
                "name": "study_area/met_em.d03_test.nc",
                "timeCreated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        )
    )

    # Ensure we worked with the right GCP paths.
    mock_storage_client.assert_has_calls(
        [
            mock.call().bucket("bucket"),
            mock.call().bucket().blob("study_area/met_em.d03_test.nc"),
            mock.call().bucket("climateiq-study-area-feature-chunks"),
            mock.call().bucket().blob("study_area/met_em.d03_test.npy"),
        ]
    )

    mock_archive_blob.open.assert_called_once_with("rb")

    # Ensure we attempted to upload a serialized matrix of the netcdf
    mock_feature_blob.upload_from_file.assert_called_once_with(mock.ANY)
    uploaded_array = numpy.load(mock_feature_blob.upload_from_file.call_args[0][0])
    # Expected array should be all required vars extracted, lon/lat axis reordered, and
    # stacked
    expected_array = numpy.dstack(
        (numpy.swapaxes(netcdf_array1, 0, 1), numpy.swapaxes(netcdf_array2, 0, 1))
    )
    numpy.testing.assert_array_equal(uploaded_array, expected_array)

    # Ensure we wrote firestore entries for the chunk.
    mock_firestore_client.assert_has_calls(
        [
            mock.call(),
            mock.call().collection("study_areas"),
            mock.call().collection().document("study_area"),
            mock.call().collection().document().collection("chunks"),
            mock.call()
            .collection()
            .document()
            .collection()
            .document("met_em.d03_test"),
            mock.call()
            .collection()
            .document()
            .collection()
            .document()
            .set(
                {
                    # For heat, we process each WPS netcdf file as a chunk (un-tar'ed)
                    "raw_path": "gs://bucket/study_area/met_em.d03_test.nc",
                    "feature_matrix_path": (
                        "gs://climateiq-study-area-feature-chunks/study_area/"
                        "met_em.d03_test.npy"
                    ),
                    "time": datetime.datetime(2010, 2, 2, 18, 0, 0),
                    "error": firestore.DELETE_FIELD,
                },
                merge=True,
            ),
        ]
    )


@mock.patch.dict(main.wps_data.ML_REQUIRED_VARS_REPO, {"RH": {}}, clear=True)
def test_process_wps_feature_drop_time_axis():
    arr = numpy.array([[[1, 2], [3, 4]]], dtype=numpy.float32)  # shape=(1,2,2)
    xrdarr = xarray.DataArray(arr, name="RH", dims=("Time", "south_north", "west_east"))

    processed = main._process_wps_feature(
        xrdarr, wps_data.ML_REQUIRED_VARS_REPO.get("RH")
    )

    numpy.testing.assert_equal((2, 2), processed.shape)


@mock.patch.dict(main.wps_data.ML_REQUIRED_VARS_REPO, {"RH": {}}, clear=True)
def test_process_wps_feature_reorder_spatial_dims():
    # Dims ordered here are representative of how they will be ordered in actual netcdf
    # file
    arr = numpy.array([[[[10, 20], [30, 40]], [[1, 2], [3, 4]]]], dtype=numpy.float32)
    xrdarr = xarray.DataArray(
        arr, name="RH", dims=("Time", "num_metgrid_levels", "south_north", "west_east")
    )

    processed = main._process_wps_feature(
        xrdarr, wps_data.ML_REQUIRED_VARS_REPO.get("RH")
    )

    expected = [[10, 30], [20, 40]]
    numpy.testing.assert_array_equal(expected, processed)


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
    # Expected arr will also have lat/lon axis swapped
    numpy.testing.assert_array_equal(numpy.swapaxes(expected, 0, 1), processed)


@mock.patch.dict(main.wps_data.ML_REQUIRED_VARS_REPO, {"GREENFRAC": {}}, clear=True)
def test_process_wps_feature_extract_monthly_climate_map_dim():
    arr = numpy.array([[[[10, 1], [20, 2]], [[30, 3], [40, 4]]]], dtype=numpy.float32)
    xrdarr = xarray.DataArray(
        arr,
        name="GREENFRAC",
        dims=("Time", "south_north", "west_east", "z-dimension0012"),
        coords={"Time": [b"2010-02-02_18:00:00"]},
    )

    processed = main._process_wps_feature(
        xrdarr, wps_data.ML_REQUIRED_VARS_REPO.get("GREENFRAC")
    )

    # Since dataset datetime is February (2), then the corresponding index to select
    # from z-dimension0012 will be 1
    expected = [[1, 2], [3, 4]]
    # Expected arr will also have lat/lon axis swapped
    numpy.testing.assert_array_equal(numpy.swapaxes(expected, 0, 1), processed)


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
    # Expected arr will also have lat/lon axis swapped
    numpy.testing.assert_array_almost_equal(numpy.swapaxes(expected, 0, 1), processed)


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
    # Expected arr will also have lat/lon axis swapped
    numpy.testing.assert_array_almost_equal(
        numpy.swapaxes(expected, 0, 1), processed, decimal=3
    )


def test_compute_custom_wps_variables_wind():
    ncfile = netCDF4.Dataset("met_em_test.nc", mode="w", format="NETCDF4", memory=1)
    ncfile.createDimension("Time", 1)
    ncfile.createDimension("west_east", 3)
    ncfile.createDimension("south_north", 3)
    ncfile.createDimension("west_east_stag", 4)
    ncfile.createDimension("south_north_stag", 4)
    ncfile.createDimension("num_metgrid_levels", 3)

    # In WPS/WRF files, 'Times'->dimension and 'Time'->variable
    ncfile.createVariable("Times", "f8", ("Time",))
    uu = ncfile.createVariable(
        "UU", "float32", ("Time", "num_metgrid_levels", "south_north", "west_east_stag")
    )
    vv = ncfile.createVariable(
        "VV", "float32", ("Time", "num_metgrid_levels", "south_north_stag", "west_east")
    )

    uu[:] = numpy.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=numpy.float32
    )
    vv[:] = numpy.array(
        [[22, 33, 44], [55, 66, 77], [88, 99, 111], [222, 333, 444]],
        dtype=numpy.float32,
    )

    memfile = ncfile.close()
    ncfile_bytes = memfile.tobytes()
    ds = xarray.open_dataset(io.BytesIO(ncfile_bytes))

    processed_ds = main._compute_custom_wps_variables(ds)

    # Check that processed dataset contains newly computed variables
    assert all(var in processed_ds.keys() for var in ["WSPD10", "WDIR10"])
    # Check expected shape of computed variable
    assert processed_ds.data_vars["WSPD10"].values.shape == (1, 3, 3, 3)


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

    with pytest.raises(RuntimeError):
        main.build_feature_matrix(
            functions_framework.CloudEvent(
                {"source": "test", "type": "event"},
                data={
                    "bucket": "bucket",
                    "name": "study_area/chunk_1_2.tar",
                    "timeCreated": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
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
            mock.call().collection().document().collection().document("chunk_1_2"),
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


@mock.patch.object(main.firestore, "Client", autospec=True)
@mock.patch.object(main.storage, "Client", autospec=True)
def test_build_and_upload_study_area_chunk(mock_storage_client, mock_firestore_client):
    # Create an in-memory netcdf file and grab its bytes
    ncfile = netCDF4.Dataset("met_em.d03_test.nc", mode="w", format="NETCDF4", memory=1)
    ncfile.createDimension("Time", 1)
    ncfile.createVariable("time", "f8", ("Time",))

    memfile = ncfile.close()

    # Create a mock blob for the input file uploaded
    mock_input_blob = mock.MagicMock()
    mock_input_blob.name = "study_area/met_em.d03_test.nc"
    mock_input_blob.bucket.name = "input-bucket"
    mock_input_blob.open.return_value = memfile

    # Create a mock blob for the archive we will upload.
    mock_chunk_blob = mock.MagicMock()
    mock_chunk_blob.name = "study_area/met_em.d03_test.nc"
    mock_chunk_blob.bucket.name = "climateiq-study-area-chunks"

    # Return the mock blobs.
    mock_storage_client().bucket("").blob.side_effect = [
        mock_input_blob,
        mock_chunk_blob,
    ]
    mock_storage_client.reset_mock()

    main.build_and_upload_study_area_chunk(
        functions_framework.CloudEvent(
            {"source": "test", "type": "event"},
            data={
                "bucket": "input-bucket",
                "name": "study_area/met_em.d03_test.nc",
                "timeCreated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        )
    )

    # Ensure we worked with the right GCP paths.
    mock_storage_client.assert_has_calls(
        [
            mock.call().bucket("input-bucket"),
            mock.call().bucket().blob("study_area/met_em.d03_test.nc"),
            mock.call().bucket("climateiq-study-area-chunks"),
            mock.call().bucket().blob("study_area/met_em.d03_test.nc"),
        ]
    )

    # Ensure we attempted to upload the chunked file
    mock_chunk_blob.upload_from_file.assert_called_once_with(mock.ANY)


@mock.patch.object(main.error_reporting, "Client", autospec=True)
@mock.patch.object(main.firestore, "Client", autospec=True)
@mock.patch.object(main.storage, "Client", autospec=True)
def test_write_study_area_metadata(mock_storage_client, mock_firestore_client, _):
    # Return the elevation header from the mock storage client.
    mock_storage_client().bucket("").blob("").open.return_value = io.StringIO(
        textwrap.dedent(
            """\
            {
            "col_count": 3,
            "row_count": 2,
            "x_ll_corner": 0.0,
            "y_ll_corner": 2.0,
            "cell_size": 1.0,
            "nodata_value": 0.0,
            "crs": "EPSG:32618"
            }
            """
        )
    )

    main.write_study_area_metadata(
        functions_framework.CloudEvent(
            {"source": "test", "type": "event"},
            data={
                "bucket": "bucket",
                "name": "study_area/header.json",
                "timeCreated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        )
    )

    # Ensure we worked with the right GCP paths.
    mock_storage_client.assert_has_calls(
        [
            mock.call().bucket("bucket"),
            mock.call().bucket().blob("study_area/header.json"),
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


@mock.patch.object(main.error_reporting, "Client", autospec=True)
@mock.patch.object(main.firestore, "Client", autospec=True)
@mock.patch.object(main.storage, "Client", autospec=True)
@mock.patch.object(main.wrf, "getvar", autospec=True)
def test_build_label_matrix_wrf(
    mock_wrf_getvar, mock_storage_client, mock_firestore_client, _
):
    # Create an in-memory mock netcdf file and grab its bytes
    ncfile = netCDF4.Dataset("met_em.d03_test.nc", mode="w", format="NETCDF4", memory=1)
    ncfile.createDimension("Time", 1)
    time = ncfile.createVariable("Times", "str", ("Time",))
    time[0] = "2010-02-02_18:00:00"

    memfile = ncfile.close()
    ncfile_bytes = memfile.tobytes()

    # Create a mock blob for the input file which will return the above netcdf when
    # opened.
    mock_input_blob = mock.MagicMock()
    mock_input_blob.name = "study_area/wrfout.d03_test"
    mock_input_blob.bucket.name = "bucket"
    mock_input_blob.open.return_value = io.BytesIO(ncfile_bytes)

    # Create a mock blob for label matrix we will upload.
    mock_label_blob = mock.MagicMock()
    mock_label_blob.name = "study_area/wrfout.d03_test.npy"
    mock_label_blob.bucket.name = "climateiq-study-area-label-chunks"

    # Return the mock blobs.
    mock_storage_client().bucket("").blob.side_effect = [
        mock_input_blob,
        mock_label_blob,
    ]
    mock_storage_client.reset_mock()

    # Mock out the values that wrf-python will derive and return
    rh2 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    t2 = [[10, 20, 30], [40, 50, 60], [70, 80, 90]]
    wspd_wdir10 = [
        [[11, 11, 11], [11, 11, 11], [11, 11, 11]],
        [[22, 22, 22], [22, 22, 22], [22, 22, 22]],
    ]

    mock_wrf_getvar.side_effect = [
        rh2,
        t2,
        wspd_wdir10,
    ]
    mock_wrf_getvar.reset_mock()

    main.build_label_matrix(
        functions_framework.CloudEvent(
            {"source": "test", "type": "event"},
            data={
                "bucket": "bucket",
                "name": "study_area/wrfout.d03_test",
                "timeCreated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        )
    )

    # Ensure we worked with the right GCP paths.
    mock_storage_client.assert_has_calls(
        [
            mock.call().bucket("bucket"),
            mock.call().bucket().blob("study_area/wrfout.d03_test"),
            mock.call().bucket("climateiq-study-area-label-chunks"),
            mock.call().bucket().blob("study_area/wrfout.d03_test.npy"),
        ]
    )

    mock_input_blob.open.assert_called_once_with("rb")

    # Ensure we attempted to upload a serialized label matrix
    mock_label_blob.upload_from_file.assert_called_once_with(mock.ANY)
    uploaded_array = numpy.load(mock_label_blob.upload_from_file.call_args[0][0])
    # Expected array should be all required vars extracted, lon/lat axis reordered, and
    # stacked
    expected_array = numpy.dstack(
        [
            numpy.swapaxes(rh2, 0, 1),
            numpy.swapaxes(t2, 0, 1),
            numpy.swapaxes(wspd_wdir10[0], 0, 1),
            numpy.swapaxes(wspd_wdir10[1], 0, 1),
        ]
    )
    numpy.testing.assert_array_equal(uploaded_array, expected_array)

    mock_firestore_client.assert_has_calls(
        [
            mock.call().collection("simulations"),
            mock.call().collection().document("study_area-None"),
            mock.call().collection().document().collection("label_chunks"),
            mock.call()
            .collection()
            .document()
            .collection()
            .document("2010-02-02 18:00:00"),
            mock.call()
            .collection()
            .document()
            .collection()
            .document()
            .set(
                {
                    "gcs_uri": (
                        "gs://climateiq-study-area-label-chunks/study_area/"
                        "wrfout.d03_test.npy"
                    ),
                    "time": datetime.datetime(2010, 2, 2, 18, 0, 0),
                }
            ),
        ]
    )


@mock.patch.object(main.error_reporting, "Client", autospec=True)
@mock.patch.object(main.firestore, "Client", autospec=True)
@mock.patch.object(main.storage, "Client", autospec=True)
def test_write_flood_scenario_metadata(mock_storage_client, mock_firestore_client, _):
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

    with pytest.raises(ValueError):
        main.write_flood_scenario_metadata_and_features(
            functions_framework.CloudEvent(
                {"source": "test", "type": "event"},
                data={
                    "bucket": "bucket",
                    "name": "config_name/Rainfall_Data_1.txt",
                    "timeCreated": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                },
            )
        )

    # Ensure we didn't try to write anything.
    vector_blob.upload_from_file.assert_not_called()
    mock_firestore_client().collection.assert_not_called()
    mock_error_reporting_client().report_exception.assert_called_once()


@mock.patch.object(main.error_reporting, "Client", autospec=True)
@mock.patch.object(main.firestore, "Client", autospec=True)
def test_write_flood_scenario_metadata_ignore_non_rainfall_files(
    mock_firestore_client, _
):
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


@mock.patch.object(main.error_reporting, "Client", autospec=True)
@mock.patch.object(main.firestore, "Client", autospec=True)
def test_delete_flood_scenario_metadata(mock_firestore_client, _):
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


@mock.patch.object(main.error_reporting, "Client", autospec=True)
@mock.patch.object(main.firestore, "Client", autospec=True)
def test_delete_flood_scenario_metadata_ignore_non_rainfall_files(
    mock_firestore_client, _
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


@mock.patch.object(main.error_reporting, "Client", autospec=True)
def test_retry_decorator_runs_decorated_func(mock_error_reporter):
    """Ensures our retry decorator passes the event to the decorated function."""
    mock_func = mock.MagicMock()
    decorated_func = main._retry_and_report_errors()(mock_func)

    event = functions_framework.CloudEvent(
        {"source": "test", "type": "event"},
        data={
            "bucket": "bucket",
            "name": "some_file",
            "timeCreated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )

    decorated_func(event)

    mock_func.assert_called_with(event)
    mock_error_reporter.assert_not_called()


@mock.patch.object(main.error_reporting, "Client", autospec=True)
def test_retry_decorator_skips_gcloud_tmp_files(_):
    """Ensures our retry decorator skips GCS events for temp gcloud files."""
    mock_func = mock.MagicMock()
    decorated_func = main._retry_and_report_errors()(mock_func)

    decorated_func(
        functions_framework.CloudEvent(
            {"source": "test", "type": "event"},
            data={
                "bucket": "bucket",
                "name": "gcloud/tmp/parallel_composite_uploads/some_file",
                "timeCreated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
        )
    )
    mock_func.assert_not_called()


@mock.patch.object(main.error_reporting, "Client", autospec=True)
def test_retry_decorator_error_reporting(mock_error_reporter):
    """Ensures our retry decorator logs and raises errors."""
    mock_func = mock.MagicMock(side_effect=RuntimeError("oh no!"))
    decorated_func = main._retry_and_report_errors()(mock_func)

    with pytest.raises(RuntimeError):
        decorated_func(
            functions_framework.CloudEvent(
                {"source": "test", "type": "event"},
                data={
                    "bucket": "bucket",
                    "name": "some_file",
                    "timeCreated": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                },
            )
        )

    mock_error_reporter.assert_called_with()


@mock.patch.object(main.error_reporting, "Client", autospec=True)
def test_retry_decorator_custom_error_reporting(mock_error_reporter):
    """Ensures our retry decorator passes the event to custom error handlers."""
    error = RuntimeError("oh no!")
    mock_func = mock.MagicMock(side_effect=error)
    error_handler = mock.MagicMock()
    decorated_func = main._retry_and_report_errors(error_handler)(mock_func)

    event = functions_framework.CloudEvent(
        {"source": "test", "type": "event"},
        data={
            "bucket": "bucket",
            "name": "some_file",
            "timeCreated": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )

    with pytest.raises(RuntimeError):
        decorated_func(event)

    error_handler.assert_called_once_with(event, error)
    mock_error_reporter.assert_called_once_with()


@mock.patch.object(main.error_reporting, "Client", autospec=True)
def test_retry_decorator_error_timeout(mock_error_reporter):
    """Ensures our retry decorator halts on old invocations."""
    mock_func = mock.MagicMock(side_effect=RuntimeError("oh no!"))
    decorated_func = main._retry_and_report_errors()(mock_func)

    decorated_func(
        functions_framework.CloudEvent(
            {"source": "test", "type": "event"},
            data={
                "bucket": "bucket",
                "name": "some_file",
                # Pick a time from long ago so the function will not be called.
                "timeCreated": "2012-12-21T01:02:00+00:00",
            },
        )
    )

    mock_func.assert_not_called()
    mock_error_reporter.assert_not_called()


@mock.patch.object(main.firestore, "Client", autospec=True)
def test_write_city_cat_output_chunks(mock_firestore_client):
    """Ensures we write the correct chunks to GCS."""
    # Create a fake .rsl file simulation result to read.
    blob_path = pathlib.PurePosixPath(
        "study_area_name/config_group/config_name.txt/R11_C1_T2_10min.rsl"
    )
    mock_blob = mock.MagicMock(spec=storage.Blob)
    mock_blob.name = str(blob_path)
    mock_blob.bucket.name = "sim-bucket"
    mock_blob.open.return_value = io.StringIO(
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
    # The above will rasterize to
    # array([[0.   , 0.   , 0.   ],
    #        [0.   , 0.   , 0.003],
    #        [0.   , 0.002, 0.   ],
    #        [0.001, 0.   , 0.004]])

    # Return a study area geography from the mock firestore client.
    mock_firestore_client().collection().document().get().to_dict.return_value = {
        "col_count": 3,
        "row_count": 4,
        "x_ll_corner": 0,
        "y_ll_corner": 0,
        "cell_size": 1.0,
    }

    # Have the blob return IO objects so we can capture what was written to them.
    # Empty their close method so we can read them later.
    buffers = (io.BytesIO(), io.BytesIO())
    for buf in buffers:
        buf.close = lambda: None
    mock_blob.bucket.blob().open.side_effect = buffers

    mock_blob.reset_mock()
    mock_firestore_client.reset_mock()
    main._write_city_cat_output_chunks(mock_blob, blob_path, chunk_size=3)

    # Ensure we wrote the two expected chunks.
    for buf in buffers:
        buf.seek(0)
    numpy.testing.assert_array_almost_equal(
        numpy.load(buffers[0]),
        numpy.array([[0, 0, 0], [0, 0, 0.003], [0, 0.002, 0]], dtype=numpy.float32),
    )
    numpy.testing.assert_array_almost_equal(
        numpy.load(buffers[1]),
        numpy.array([[0.001, 0, 0.004], [0, 0, 0], [0, 0, 0]], dtype=numpy.float32),
    )

    # Ensure we wrote the chunks to the expected paths.
    mock_blob.assert_has_calls(
        [
            mock.call.bucket.blob(
                "timestep_parts/study_area_name/config_group/config_name.txt/0_0/2.npy"
            ),
            mock.call.bucket.blob().open("wb"),
            mock.call.bucket.blob(
                "timestep_parts/study_area_name/config_group/config_name.txt/0_1/2.npy"
            ),
            mock.call.bucket.blob().open("wb"),
        ]
    )

    # Ensure we created a metastore entry for the simulation run.
    mock_firestore_client.assert_has_calls(
        [
            mock.call().collection("study_areas"),
            mock.call().collection().document("study_area_name"),
            mock.call().collection("city_cat_rainfall_configs"),
            mock.call().collection().document("config_group%2Fconfig_name.txt"),
        ]
    )
    mock_firestore_client().collection().document().set.assert_called_once_with(
        {
            "gcs_prefix_uri": (
                "gs://sim-bucket/study_area_name/config_group/config_name.txt"
            ),
            "simulation_type": "CityCAT",
            "study_area": mock_firestore_client().collection().document(),
            "configuration": mock_firestore_client().collection().document(),
        }
    )


@mock.patch.object(main.firestore, "Client", autospec=True)
def test_collapse_city_cat_output_chunks(mock_firestore_client):
    """Ensures we write the correct chunks to GCS."""
    # Have firestore return a three-timestep configuration.
    mock_firestore_client().collection().document().get().to_dict.return_value = {
        "gcs_uri": "gs://config-bucket/config/file.txt",
        "as_vector_gcs_uri": "gs://label-bucket/config/file.npy",
        "parent_config_name": "config",
        "rainfall_duration": 3,
    }

    # Create a blob for the file triggering the cloud function run.
    blob_path = pathlib.PurePosixPath(
        "timestep_parts/study_area_name/config_group/config_name.txt/0_1/2.npy"
    )
    mock_blob = mock.MagicMock(spec=storage.Blob)
    mock_blob.name = str(blob_path)
    mock_blob.bucket.name = "sim-bucket"

    # Have list blobs return paths for the three expected timesteps
    mock_blob.bucket.list_blobs.return_value = [
        mock.MagicMock(spec=storage.Blob),
        mock.MagicMock(spec=storage.Blob),
        mock.MagicMock(spec=storage.Blob),
    ]

    # Set the names and read values of the above blobs.
    buf_t1 = io.BytesIO()
    numpy.save(buf_t1, numpy.array([[1, 1], [1, 1]]))
    buf_t1.seek(0)
    mock_blob.bucket.list_blobs.return_value[0].name = (
        "timestep_parts/study_area_name/config_group/config_name.txt/0_1/1.npy"
    )
    mock_blob.bucket.list_blobs.return_value[0].open.return_value = buf_t1

    buf_t0 = io.BytesIO()
    numpy.save(buf_t0, numpy.array([[0, 0], [0, 0]]))
    buf_t0.seek(0)
    mock_blob.bucket.list_blobs.return_value[1].name = (
        "timestep_parts/study_area_name/config_group/config_name.txt/0_1/0.npy"
    )
    mock_blob.bucket.list_blobs.return_value[1].open.return_value = buf_t0

    buf_t2 = io.BytesIO()
    numpy.save(buf_t2, numpy.array([[2, 2], [2, 2]]))
    buf_t2.seek(0)
    mock_blob.bucket.list_blobs.return_value[2].name = (
        "timestep_parts/study_area_name/config_group/config_name.txt/0_1/2.npy"
    )
    mock_blob.bucket.list_blobs.return_value[2].open.return_value = buf_t2

    # Create a mock labels bucket.
    # Have the blob return IO objects so we can capture what was written to them.
    # Empty their close method so we can read them later.
    mock_labels_bucket = mock.MagicMock(spec=storage.Bucket)
    mock_labels_bucket.name = "labels"
    labels_buf = io.BytesIO()
    labels_buf.close = lambda: None
    mock_labels_bucket.blob().open.return_value = labels_buf

    mock_blob.reset_mock()
    mock_labels_bucket.reset_mock()
    main._collapse_city_cat_output_chunks(mock_blob, blob_path, mock_labels_bucket)

    # Ensure we checked the right prefix.
    mock_blob.bucket.list_blobs.assert_called_once_with(
        prefix="timestep_parts/study_area_name/config_group/config_name.txt/0_1/"
    )
    # Ensure we wrote to the correct labels path.
    mock_labels_bucket.blob.assert_called_once_with(
        "study_area_name/config_group/config_name.txt/0_1.npy"
    )
    # Ensure we wrote the correct array to the labels bucket.
    labels_buf.seek(0)
    numpy.testing.assert_array_equal(
        numpy.load(labels_buf),
        numpy.array([[[0, 1, 2], [0, 1, 2]], [[0, 1, 2], [0, 1, 2]]]),
    )
    # Ensure we created a metastore entry for the chunk.
    mock_firestore_client.assert_has_calls(
        [
            mock.call().collection("simulations"),
            mock.call()
            .collection()
            .document("study_area_name-config_group%2Fconfig_name.txt"),
            mock.call().collection().document().collection("label_chunks"),
            mock.call().collection().document().collection().document("0_1"),
            mock.call()
            .collection()
            .document()
            .collection()
            .document()
            .set(
                {
                    "gcs_uri": (
                        "gs://labels/study_area_name/"
                        "config_group/config_name.txt/0_1.npy"
                    ),
                    "x_index": 0,
                    "y_index": 1,
                }
            ),
        ]
    )
    # Ensure we deleted all the intermediary chunks.
    for blob in mock_blob.bucket.list_blobs.return_value:
        blob.delete.assert_called_once()


@mock.patch.object(main.firestore, "Client", autospec=True)
def test_collapse_city_cat_output_chunks_missing_timesteps(mock_firestore_client):
    """Ensures halt if files for all timesteps are not present."""
    # Have firestore return a three-timestep configuration.
    mock_firestore_client().collection().document().get().to_dict.return_value = {
        "gcs_uri": "gs://config-bucket/config/file.txt",
        "as_vector_gcs_uri": "gs://label-bucket/config/file.npy",
        "parent_config_name": "config",
        "rainfall_duration": 3,
    }

    # Create a blob for the file triggering the cloud function run.
    blob_path = pathlib.PurePosixPath(
        "timestep_parts/study_area_name/config_group/config_name.txt/0_1/2.npy"
    )
    mock_blob = mock.MagicMock(spec=storage.Blob)
    mock_blob.name = str(blob_path)
    mock_blob.bucket.name = "sim-bucket"

    # Have list blobs return paths for the two of the three expected timesteps
    mock_blob.bucket.list_blobs.return_value = [
        mock.MagicMock(spec=storage.Blob),
        mock.MagicMock(spec=storage.Blob),
    ]
    mock_blob.bucket.list_blobs.return_value[0].name = (
        "timestep_parts/study_area_name/config_group/config_name.txt/0_1/1.npy"
    )
    mock_blob.bucket.list_blobs.return_value[1].name = (
        "timestep_parts/study_area_name/config_group/config_name.txt/0_1/2.npy"
    )

    # Create a mock labels bucket.
    mock_labels_bucket = mock.MagicMock(spec=storage.Bucket)
    mock_labels_bucket.name = "labels"

    mock_blob.reset_mock()
    mock_labels_bucket.reset_mock()
    main._collapse_city_cat_output_chunks(mock_blob, blob_path, mock_labels_bucket)

    # Ensure we didn't try to write to the mock labels.
    mock_labels_bucket.blob.assert_not_called()

    # Ensure we didn't try to write a metastore entry.
    (
        mock_firestore_client()
        .collection()
        .document()
        .collection()
        .document()
        .set.assert_not_called()
    )

    # Ensure we didn't try to delete the intermediary chunks.
    for blob in mock_blob.bucket.list_blobs.return_value:
        blob.delete.assert_not_called()


@mock.patch.object(main.firestore, "Client", autospec=True)
def test_collapse_city_cat_output_chunks_deleted_chunks(mock_firestore_client):
    """Ensures halt if files for timesteps are deleted when read."""
    # Have firestore return a three-timestep configuration.
    mock_firestore_client().collection().document().get().to_dict.return_value = {
        "gcs_uri": "gs://config-bucket/config/file.txt",
        "as_vector_gcs_uri": "gs://label-bucket/config/file.npy",
        "parent_config_name": "config",
        "rainfall_duration": 3,
    }

    # Create a blob for the file triggering the cloud function run.
    blob_path = pathlib.PurePosixPath(
        "timestep_parts/study_area_name/config_group/config_name.txt/0_1/2.npy"
    )
    mock_blob = mock.MagicMock(spec=storage.Blob)
    mock_blob.name = str(blob_path)
    mock_blob.bucket.name = "sim-bucket"

    # Have list blobs return paths for the three expected timesteps, but have their
    # reads fail.
    mock_blob.bucket.list_blobs.return_value = [
        mock.MagicMock(spec=storage.Blob),
        mock.MagicMock(spec=storage.Blob),
        mock.MagicMock(spec=storage.Blob),
    ]
    mock_blob.bucket.list_blobs.return_value[0].name = (
        "timestep_parts/study_area_name/config_group/config_name.txt/0_1/0.npy"
    )
    mock_blob.bucket.list_blobs.return_value[0].open.side_effect = exceptions.NotFound(
        "oh no!"
    )
    mock_blob.bucket.list_blobs.return_value[1].name = (
        "timestep_parts/study_area_name/config_group/config_name.txt/0_1/1.npy"
    )
    mock_blob.bucket.list_blobs.return_value[2].name = (
        "timestep_parts/study_area_name/config_group/config_name.txt/0_1/2.npy"
    )

    # Create a mock labels bucket.
    mock_labels_bucket = mock.MagicMock(spec=storage.Bucket)
    mock_labels_bucket.name = "labels"

    mock_blob.reset_mock()
    mock_labels_bucket.reset_mock()
    main._collapse_city_cat_output_chunks(mock_blob, blob_path, mock_labels_bucket)

    # Ensure we tried to read the deleted blob.
    mock_blob.bucket.list_blobs.return_value[0].open.assert_called_once()

    # Ensure we didn't try to write to the mock labels.
    mock_labels_bucket.blob.assert_not_called()

    # Ensure we didn't try to write a metastore entry.
    (
        mock_firestore_client()
        .collection()
        .document()
        .collection()
        .document()
        .set.assert_not_called()
    )

    # Ensure we didn't try to delete the intermediary chunks.
    for blob in mock_blob.bucket.list_blobs.return_value:
        blob.delete.assert_not_called()


def test_calculate_metadata_for_elevation_happy_path():
    header = geo_data.ElevationHeader(
        col_count=2,
        row_count=2,
        x_ll_corner=0.0,
        y_ll_corner=2.0,
        cell_size=1.0,
        nodata_value=0.0,
    )
    data = numpy.array([[0.0, 1.0], [4.0, 5.0]])
    metadata = main._calculate_metadata_for_elevation(
        geo_data.Elevation(header=header, data=data)
    )
    assert metadata == main.FeatureMetadata(
        elevation_min=1.0, elevation_max=5.0, chunk_size=2
    )


def test_calculate_metadata_for_elevation_empty_area():
    header = geo_data.ElevationHeader(
        col_count=3,
        row_count=2,
        x_ll_corner=0.0,
        y_ll_corner=2.0,
        cell_size=1.0,
        nodata_value=0.0,
    )
    data = numpy.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    metadata = main._calculate_metadata_for_elevation(
        geo_data.Elevation(header=header, data=data)
    )
    assert metadata == main.FeatureMetadata(elevation_min=None, elevation_max=None)
