from unittest import mock

from study_area_uploader import main


@mock.patch.object(main, "study_area_chunkers")
@mock.patch.object(main, "study_area_transformers")
@mock.patch.object(main, "parse_args")
@mock.patch.object(main.firestore, "Client", autospec=True)
@mock.patch.object(main.storage, "Client", autospec=True)
def test_happy_path(
    mock_storage_client,
    mock_firestore_client,
    mock_parse_args_func,
    mock_transformers,
    mock_chunkers,
):
    """Ensure study area files are uploaded, metadata is stored and chunks are made."""
    main.main()
