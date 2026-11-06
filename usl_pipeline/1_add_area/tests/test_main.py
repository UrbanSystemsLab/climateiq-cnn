import pathlib
from unittest import mock

from study_area_uploader import main
from study_area_uploader.transformers import study_area_transformers
from usl_lib.storage import cloud_storage


@mock.patch.object(main.study_area_chunkers, "build_and_upload_chunks")
@mock.patch.object(main.study_area_chunkers, "build_and_upload_chunks_citycat")
@mock.patch.object(
    main.study_area_transformers, "prepare_and_upload_citycat_input_files"
)
@mock.patch.object(main.study_area_transformers, "prepare_and_upload_study_area_files")
@mock.patch.object(main, "_parse_args")
@mock.patch.object(main.firestore, "Client", autospec=True)
@mock.patch.object(main.storage, "Client", autospec=True)
def test_happy_path(
    mock_storage_client,
    mock_firestore_client,
    mock_parse_args_func,
    mock_prepare_and_upload_study_area_files,
    mock_prepare_and_upload_citycat_input_files,
    mock_build_and_upload_chunks_citycat,
    mock_build_and_upload_chunks,
):
    """Ensure study area files are uploaded, metadata is stored and chunks are made."""
    study_area_name = "TestStudyArea"
    elevation_file_path = "path/to/elevation.tiff"
    boundaries_file_path = "path/to/boundaries.shp"
    buildings_file_path = "path/to/buildings.shp"
    green_areas_file_path = "path/to/green_areas.shp"
    soil_classes_file_path = "path/to/soil_classes.shp"
    parser = main._get_args_parser()
    mock_parse_args_func.return_value = parser.parse_args(
        [
            f"--name={study_area_name}",
            f"--elevation-file={elevation_file_path}",
            f"--boundaries-file={boundaries_file_path}",
            f"--building-footprint-file={buildings_file_path}",
            f"--green-areas-file={green_areas_file_path}",
            f"--soil-type-file={soil_classes_file_path}",
            "--export-to-citycat",
            "--overwrite",
        ]
    )

    study_area_bucket_mock = mock.MagicMock()
    study_area_blob_mock = mock.MagicMock()
    study_area_bucket_mock.list_blobs.return_value = [study_area_blob_mock]
    citycat_bucket_mock = mock.MagicMock()
    chunk_bucket_mock = mock.MagicMock()
    feature_bucket_mock = mock.MagicMock()
    citycat_chunked_bucket_mock = mock.MagicMock()
    mock_storage_client().bucket.side_effect = [
        study_area_bucket_mock,
        citycat_bucket_mock,
        citycat_chunked_bucket_mock,
        chunk_bucket_mock,
        feature_bucket_mock,
    ]
    mock_storage_client.reset_mock()

    prepared_inputs = study_area_transformers.PreparedInputData(
        elevation_file_path=pathlib.Path(elevation_file_path),
        boundaries_polygons=None,
        buildings_polygons=None,
        green_areas_polygons=None,
        soil_classes_polygons=None,
    )
    mock_prepare_and_upload_study_area_files.return_value = prepared_inputs

    mock_db = mock_firestore_client()
    mock_db.collection().document().get().to_dict.return_value = {
        "col_count": 10,
        "row_count": 20,
        "x_ll_corner": 0.0,
        "y_ll_corner": 0.0,
        "cell_size": 1.0,
    }

    main.main()

    mock_storage_client.assert_has_calls(
        [
            mock.call().bucket(cloud_storage.STUDY_AREA_BUCKET),
            mock.call().bucket(cloud_storage.FLOOD_SIMULATION_INPUT_BUCKET),
            mock.call().bucket(cloud_storage.FLOOD_SIMULATION_INPUT_BUCKET_CHUNKED),
            mock.call().bucket(cloud_storage.STUDY_AREA_CHUNKS_BUCKET),
            mock.call().bucket(cloud_storage.FEATURE_CHUNKS_BUCKET),
        ]
    )

    study_area_bucket_mock.list_blobs.assert_called_once_with(
        prefix=f"{study_area_name}/"
    )
    study_area_blob_mock.delete.assert_called_once()

    mock_prepare_and_upload_study_area_files.assert_called_once_with(
        study_area_name,
        elevation_file_path,
        boundaries_file_path,
        buildings_file_path,
        green_areas_file_path,
        soil_classes_file_path,
        "soil_class",
        mock.ANY,
        study_area_bucket_mock,
        input_non_green_area_soil_classes=set(),
    )

    mock_prepare_and_upload_citycat_input_files.assert_called_once_with(
        study_area_name,
        prepared_inputs,
        mock.ANY,
        citycat_bucket_mock,
        elevation_geotiff_band=1,
    )

    mock_build_and_upload_chunks_citycat.assert_called_once_with(
        study_area_name,
        prepared_inputs,
        mock.ANY,
        citycat_chunked_bucket_mock,
        1000,
        input_elevation_band=1,
    )

    mock_build_and_upload_chunks.assert_called_once_with(
        study_area_name,
        prepared_inputs,
        mock.ANY,
        chunk_bucket_mock,
        1000,
        input_elevation_band=1,
    )
