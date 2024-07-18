import io
from unittest import mock

from google.cloud import firestore  # type:ignore[attr-defined]
from google.cloud import storage  # type:ignore[attr-defined]
import numpy

from usl_models.flood_ml import dataset


@mock.patch.object(dataset, "metastore")
def test_load_dataset_full(mock_metastore) -> None:
    """Ensures we create the expected dataset from GCS objects."""
    mock_firestore_client = mock.MagicMock(spec=firestore.Client)
    mock_storage_client = mock.MagicMock(spec=storage.Client)

    batch_size = 1

    # Set some URLs to return from our metastore functions.
    mock_metastore.get_temporal_feature_metadata.return_value = {
        "as_vector_gcs_uri": "gs://temporal-features/temporal-feature.npy"
    }
    mock_metastore.get_train_spatial_feature_and_label_chunk_metadata.return_value = [
        (
            {"feature_matrix_path": "gs://spatial-features/train-spatial-feature.npy"},
            {"gcs_uri": "gs://labels/train-labels.npy"},
        )
    ] * batch_size
    mock_metastore.get_test_spatial_feature_and_label_chunk_metadata.return_value = [
        (
            {"feature_matrix_path": "gs://spatial-features/test-spatial-feature.npy"},
            {"gcs_uri": "gs://labels/test-labels.npy"},
        )
    ] * batch_size

    # Set up some fake data to retrieve from our mock GCS.
    mock_rainfall = numpy.pad(numpy.array([1, 2, 3, 4]), (0, 860))

    mock_train_spatial_features = numpy.random.rand(1000, 1000, 8)
    mock_train_labels = numpy.random.rand(1000, 1000, 19)

    mock_test_spatial_features = numpy.random.rand(1000, 1000, 8)
    mock_test_labels = numpy.random.rand(1000, 1000, 19)

    # Set the storage client to return the right mock data for each GCS path.
    def mock_blob_func(blob_name):
        """Returns the mock data depending for blob requested."""
        if blob_name == "temporal-feature.npy":
            mock_data = mock_rainfall
        elif blob_name == "train-spatial-feature.npy":
            mock_data = mock_train_spatial_features
        elif blob_name == "train-labels.npy":
            mock_data = mock_train_labels
        elif blob_name == "test-spatial-feature.npy":
            mock_data = mock_test_spatial_features
        elif blob_name == "test-labels.npy":
            mock_data = mock_test_labels
        else:
            raise ValueError(f"Unexpected name {blob_name} passed to mock.")

        buf = io.BytesIO()
        numpy.save(buf, mock_data)
        buf.seek(0)

        mock_blob = mock.MagicMock(spec=storage.Blob)
        mock_blob.open.return_value = buf
        return mock_blob

    mock_storage_client.bucket().blob.side_effect = mock_blob_func
    mock_storage_client.reset_mock()

    m_rainfall = 6
    train_ds, test_ds = dataset.load_dataset(
        sim_names=["sim_name"],
        batch_size=batch_size,
        n_flood_maps=5,
        m_rainfall=m_rainfall,
        firestore_client=mock_firestore_client,
        storage_client=mock_storage_client,
    )

    train_batch = train_ds.take(1)
    train_element = train_batch.get_single_element()[0]
    assert train_element["spatiotemporal"].shape == (batch_size, 5, 1000, 1000, 1)

    test_batch = test_ds.take(1)
    test_element = test_batch.get_single_element()[0]
    assert train_element["spatiotemporal"].shape == (batch_size, 5, 1000, 1000, 1)

    # Ensure we requested the right blobs from GCS.
    mock_storage_client.assert_has_calls(
        [
            mock.call.bucket("temporal-features"),
            mock.call.bucket().blob("temporal-feature.npy"),
            mock.call.bucket("spatial-features"),
            mock.call.bucket().blob("train-spatial-feature.npy"),
            mock.call.bucket("labels"),
            mock.call.bucket().blob("train-labels.npy"),
        ]
    )
    mock_storage_client.assert_has_calls(
        [
            mock.call.bucket("temporal-features"),
            mock.call.bucket().blob("temporal-feature.npy"),
            mock.call.bucket("spatial-features"),
            mock.call.bucket().blob("test-spatial-feature.npy"),
            mock.call.bucket("labels"),
            mock.call.bucket().blob("test-labels.npy"),
        ]
    )

    # geospatial will have the same spatial features.
    numpy.testing.assert_array_almost_equal(
        train_element["geospatial"].numpy(),
        numpy.array([mock_train_spatial_features] * batch_size),
    )
    numpy.testing.assert_array_almost_equal(
        test_element["geospatial"].numpy(),
        numpy.array([mock_test_spatial_features] * batch_size),
    )

    # spatiotemporal will have the labels creeping into a sequence of zeros.
    numpy.testing.assert_array_almost_equal(
        train_element["spatiotemporal"].numpy(),
        numpy.array(
            [
                numpy.array(
                    [
                        numpy.zeros((1000, 1000, 1)),
                        numpy.zeros((1000, 1000, 1)),
                        numpy.zeros((1000, 1000, 1)),
                        numpy.zeros((1000, 1000, 1)),
                        numpy.zeros((1000, 1000, 1)),
                    ]
                ),
            ]
            * batch_size
        ),
    )

    numpy.testing.assert_array_almost_equal(
        test_element["spatiotemporal"].numpy(),
        numpy.array(
            [
                numpy.array(
                    [
                        numpy.zeros((1000, 1000, 1)),
                        numpy.zeros((1000, 1000, 1)),
                        numpy.zeros((1000, 1000, 1)),
                        numpy.zeros((1000, 1000, 1)),
                        numpy.zeros((1000, 1000, 1)),
                    ]
                ),
            ]
            * batch_size
        ),
    )

    # temporal will have the rainfall creeping into a sequence of zeros.
    expected_temporal = numpy.array(
        [
            numpy.array(
                [
                    numpy.full(m_rainfall, mock_rainfall[0]),
                    numpy.full(m_rainfall, mock_rainfall[1]),
                    numpy.full(m_rainfall, mock_rainfall[2]),
                    numpy.full(m_rainfall, mock_rainfall[3]),
                ]
                + [numpy.zeros(m_rainfall)] * 860
            )
        ]
    )
    numpy.testing.assert_array_almost_equal(
        train_element["temporal"].numpy(), expected_temporal
    )
    numpy.testing.assert_array_almost_equal(
        test_element["temporal"].numpy(), expected_temporal
    )


@mock.patch.object(dataset, "metastore")
def test_load_dataset_windowed(mock_metastore) -> None:
    """Ensures we create the expected dataset from GCS objects."""
    mock_firestore_client = mock.MagicMock(spec=firestore.Client)
    mock_storage_client = mock.MagicMock(spec=storage.Client)

    # Set some URLs to return from our metastore functions.
    mock_metastore.get_temporal_feature_metadata.return_value = {
        "as_vector_gcs_uri": "gs://temporal-features/temporal-feature.npy"
    }
    mock_metastore.get_train_spatial_feature_and_label_chunk_metadata.return_value = [
        (
            {"feature_matrix_path": "gs://spatial-features/train-spatial-feature.npy"},
            {"gcs_uri": "gs://labels/train-labels.npy"},
        )
    ]

    mock_metastore.get_test_spatial_feature_and_label_chunk_metadata.return_value = [
        (
            {"feature_matrix_path": "gs://spatial-features/test-spatial-feature.npy"},
            {"gcs_uri": "gs://labels/test-labels.npy"},
        )
    ]

    # Set up some fake data to retrieve from our mock GCS.
    mock_rainfall = numpy.pad(numpy.array([1, 2, 3, 4]), (0, 860))

    mock_train_spatial_features = numpy.random.rand(1000, 1000, 8)
    mock_train_labels = numpy.random.rand(1000, 1000, 19)

    mock_test_spatial_features = numpy.random.rand(1000, 1000, 8)
    mock_test_labels = numpy.random.rand(1000, 1000, 19)

    # Set the storage client to return the right mock data for each GCS path.
    def mock_blob_func(blob_name):
        """Returns the mock data depending for blob requested."""
        if blob_name == "temporal-feature.npy":
            mock_data = mock_rainfall
        elif blob_name == "train-spatial-feature.npy":
            mock_data = mock_train_spatial_features
        elif blob_name == "train-labels.npy":
            mock_data = mock_train_labels
        elif blob_name == "test-spatial-feature.npy":
            mock_data = mock_test_spatial_features
        elif blob_name == "test-labels.npy":
            mock_data = mock_test_labels
        else:
            raise ValueError(f"Unexpected name {blob_name} passed to mock.")

        buf = io.BytesIO()
        numpy.save(buf, mock_data)
        buf.seek(0)

        mock_blob = mock.MagicMock(spec=storage.Blob)
        mock_blob.open.return_value = buf
        return mock_blob

    mock_storage_client.bucket().blob.side_effect = mock_blob_func

    batch_size = 4
    m_rainfall = 6
    mock_storage_client.reset_mock()
    train_ds, test_ds = dataset.load_dataset_windowed(
        sim_names=["sim_name"],
        batch_size=batch_size,
        n_flood_maps=5,
        m_rainfall=m_rainfall,
        firestore_client=mock_firestore_client,
        storage_client=mock_storage_client,
    )
    train_batch = train_ds.take(1)
    train_element = train_batch.get_single_element()[0]

    test_batch = test_ds.take(1)
    test_element = test_batch.get_single_element()[0]

    # Ensure we requested the right blobs from GCS.
    mock_storage_client.assert_has_calls(
        [
            mock.call.bucket("temporal-features"),
            mock.call.bucket().blob("temporal-feature.npy"),
            mock.call.bucket("spatial-features"),
            mock.call.bucket().blob("train-spatial-feature.npy"),
            mock.call.bucket("labels"),
            mock.call.bucket().blob("train-labels.npy"),
        ]
    )
    mock_storage_client.assert_has_calls(
        [
            mock.call.bucket("temporal-features"),
            mock.call.bucket().blob("temporal-feature.npy"),
            mock.call.bucket("spatial-features"),
            mock.call.bucket().blob("test-spatial-feature.npy"),
            mock.call.bucket("labels"),
            mock.call.bucket().blob("test-labels.npy"),
        ]
    )

    # geospatial will have the same spatial features stacked batch_size times.
    numpy.testing.assert_array_almost_equal(
        train_element["geospatial"].numpy(),
        numpy.array([mock_train_spatial_features] * batch_size),
    )
    numpy.testing.assert_array_almost_equal(
        test_element["geospatial"].numpy(),
        numpy.array([mock_test_spatial_features] * batch_size),
    )

    # spatiotemporal will have the labels creeping into a sequence of zeros.
    numpy.testing.assert_array_almost_equal(
        train_element["spatiotemporal"].numpy(),
        numpy.array(
            [
                numpy.array(
                    [
                        numpy.zeros((1000, 1000, 1)),
                        numpy.zeros((1000, 1000, 1)),
                        numpy.zeros((1000, 1000, 1)),
                        numpy.zeros((1000, 1000, 1)),
                        numpy.zeros((1000, 1000, 1)),
                    ]
                ),
                numpy.array(
                    [
                        numpy.zeros((1000, 1000, 1)),
                        numpy.zeros((1000, 1000, 1)),
                        numpy.zeros((1000, 1000, 1)),
                        numpy.zeros((1000, 1000, 1)),
                        numpy.expand_dims(mock_train_labels[:, :, 0], axis=-1),
                    ]
                ),
                numpy.array(
                    [
                        numpy.zeros((1000, 1000, 1)),
                        numpy.zeros((1000, 1000, 1)),
                        numpy.zeros((1000, 1000, 1)),
                        numpy.expand_dims(mock_train_labels[:, :, 0], axis=-1),
                        numpy.expand_dims(mock_train_labels[:, :, 1], axis=-1),
                    ]
                ),
                numpy.array(
                    [
                        numpy.zeros((1000, 1000, 1)),
                        numpy.zeros((1000, 1000, 1)),
                        numpy.expand_dims(mock_train_labels[:, :, 0], axis=-1),
                        numpy.expand_dims(mock_train_labels[:, :, 1], axis=-1),
                        numpy.expand_dims(mock_train_labels[:, :, 2], axis=-1),
                    ]
                ),
            ]
        ),
    )
    numpy.testing.assert_array_almost_equal(
        test_element["spatiotemporal"].numpy(),
        numpy.array(
            [
                numpy.array(
                    [
                        numpy.zeros((1000, 1000, 1)),
                        numpy.zeros((1000, 1000, 1)),
                        numpy.zeros((1000, 1000, 1)),
                        numpy.zeros((1000, 1000, 1)),
                        numpy.zeros((1000, 1000, 1)),
                    ]
                ),
                numpy.array(
                    [
                        numpy.zeros((1000, 1000, 1)),
                        numpy.zeros((1000, 1000, 1)),
                        numpy.zeros((1000, 1000, 1)),
                        numpy.zeros((1000, 1000, 1)),
                        numpy.expand_dims(mock_test_labels[:, :, 0], axis=-1),
                    ]
                ),
                numpy.array(
                    [
                        numpy.zeros((1000, 1000, 1)),
                        numpy.zeros((1000, 1000, 1)),
                        numpy.zeros((1000, 1000, 1)),
                        numpy.expand_dims(mock_test_labels[:, :, 0], axis=-1),
                        numpy.expand_dims(mock_test_labels[:, :, 1], axis=-1),
                    ]
                ),
                numpy.array(
                    [
                        numpy.zeros((1000, 1000, 1)),
                        numpy.zeros((1000, 1000, 1)),
                        numpy.expand_dims(mock_test_labels[:, :, 0], axis=-1),
                        numpy.expand_dims(mock_test_labels[:, :, 1], axis=-1),
                        numpy.expand_dims(mock_test_labels[:, :, 2], axis=-1),
                    ]
                ),
            ]
        ),
    )

    # temporal will have the rainfall creeping into a sequence of zeros.
    expected_temporal = numpy.array(
        [
            numpy.array(
                [
                    numpy.zeros(m_rainfall),
                    numpy.zeros(m_rainfall),
                    numpy.zeros(m_rainfall),
                    numpy.zeros(m_rainfall),
                    numpy.zeros(m_rainfall),
                ]
            ),
            numpy.array(
                [
                    numpy.zeros(m_rainfall),
                    numpy.zeros(m_rainfall),
                    numpy.zeros(m_rainfall),
                    numpy.zeros(m_rainfall),
                    numpy.full(m_rainfall, mock_rainfall[0]),
                ]
            ),
            numpy.array(
                [
                    numpy.zeros(m_rainfall),
                    numpy.zeros(m_rainfall),
                    numpy.zeros(m_rainfall),
                    numpy.full(m_rainfall, mock_rainfall[0]),
                    numpy.full(m_rainfall, mock_rainfall[1]),
                ]
            ),
            numpy.array(
                [
                    numpy.zeros(m_rainfall),
                    numpy.zeros(m_rainfall),
                    numpy.full(m_rainfall, mock_rainfall[0]),
                    numpy.full(m_rainfall, mock_rainfall[1]),
                    numpy.full(m_rainfall, mock_rainfall[2]),
                ]
            ),
        ]
    )
    # The test and train sets will have the same temporal aspect.
    numpy.testing.assert_array_almost_equal(
        train_element["temporal"].numpy(), expected_temporal
    )
    numpy.testing.assert_array_almost_equal(
        test_element["temporal"].numpy(), expected_temporal
    )
