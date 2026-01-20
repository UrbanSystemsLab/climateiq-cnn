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
        "as_vector_gcs_uri": "gs://temporal-features/temporal-feature.npy",
        "rainfall_duration": 5,
    }
    mock_metastore.get_spatial_feature_and_label_chunk_metadata.return_value = [
        (
            {"feature_matrix_path": "gs://spatial-features/spatial-feature.npy"},
            {"gcs_uri": "gs://labels/labels.npy"},
        )
    ] * batch_size

    # Set up some fake data to retrieve from our mock GCS.
    mock_rainfall = numpy.pad(numpy.array([1, 2, 3, 4]), (0, 860))
    mock_spatial_features = numpy.random.rand(1000, 1000, 9)
    mock_labels = numpy.random.rand(1000, 1000, 19)

    # Set the storage client to return the right mock data for each GCS path.
    def mock_blob_func(blob_name):
        """Returns the mock data depending for blob requested."""
        if blob_name == "temporal-feature.npy":
            mock_data = mock_rainfall
        elif blob_name == "spatial-feature.npy":
            mock_data = mock_spatial_features
        elif blob_name == "labels.npy":
            mock_data = mock_labels
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
    ds = dataset.load_dataset(
        sim_names=["sim_name"],
        dataset_split="train",
        batch_size=batch_size,
        n_flood_maps=5,
        m_rainfall=m_rainfall,
        firestore_client=mock_firestore_client,
        storage_client=mock_storage_client,
    )
    batch = ds.take(1)
    element = batch.get_single_element()[0]
    assert element["spatiotemporal"].shape == (batch_size, 5, 1000, 1000, 1)

    numpy.testing.assert_array_almost_equal(
        element["geospatial"].numpy(),
        numpy.array([mock_spatial_features] * batch_size),
    )


@mock.patch.object(dataset, "metastore")
def test_load_dataset_windowed(mock_metastore) -> None:
    """Ensures windowed dataset starts at full temporal window."""
    mock_firestore_client = mock.MagicMock(spec=firestore.Client)
    mock_storage_client = mock.MagicMock(spec=storage.Client)

    mock_metastore.get_temporal_feature_metadata.return_value = {
        "as_vector_gcs_uri": "gs://temporal-features/temporal-feature.npy",
        "rainfall_duration": 5,
    }
    mock_metastore.get_spatial_feature_and_label_chunk_metadata.return_value = [
        (
            {"feature_matrix_path": "gs://spatial-features/spatial-feature.npy"},
            {"gcs_uri": "gs://labels/labels.npy"},
        )
    ]

    # Set up some fake data to retrieve from our mock GCS.
    mock_rainfall = numpy.pad(numpy.array([1, 2, 3, 4]), (0, 860))
    mock_spatial_features = numpy.random.rand(1000, 1000, 9)
    mock_labels = numpy.random.rand(1000, 1000, 19)

    # Set the storage client to return the right mock data for each GCS path.
    def mock_blob_func(blob_name):
        """Returns the mock data depending for blob requested."""
        if blob_name == "temporal-feature.npy":
            mock_data = mock_rainfall
        elif blob_name == "spatial-feature.npy":
            mock_data = mock_spatial_features
        elif blob_name == "labels.npy":
            mock_data = mock_labels
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
    ds = dataset.load_dataset_windowed(
        sim_names=["sim_name"],
        dataset_split="train",
        batch_size=batch_size,
        n_flood_maps=5,
        m_rainfall=m_rainfall,
        firestore_client=mock_firestore_client,
        storage_client=mock_storage_client,
    )
    batch = ds.take(1)
    element = batch.get_single_element()[0]

    numpy.testing.assert_array_almost_equal(
        element["geospatial"].numpy(),
        numpy.array([mock_spatial_features] * batch_size),
    )

    # Spatiotemporal windows start at t = n_flood_maps
    numpy.testing.assert_array_almost_equal(
        element["spatiotemporal"].numpy(),
        numpy.array(
            [
                numpy.stack(
                    [
                        numpy.expand_dims(mock_labels[:, :, i], axis=-1)
                        for i in range(0, 5)
                    ]
                ),
                numpy.stack(
                    [
                        numpy.expand_dims(mock_labels[:, :, i], axis=-1)
                        for i in range(1, 6)
                    ]
                ),
                numpy.stack(
                    [
                        numpy.expand_dims(mock_labels[:, :, i], axis=-1)
                        for i in range(2, 7)
                    ]
                ),
                numpy.stack(
                    [
                        numpy.expand_dims(mock_labels[:, :, i], axis=-1)
                        for i in range(3, 8)
                    ]
                ),
            ]
        ),
    )

    # ✅ temporal windows aligned with spatiotemporal
    numpy.testing.assert_array_almost_equal(
        element["temporal"].numpy(),
        numpy.array(
            [
                numpy.stack(
                    [numpy.full(m_rainfall, mock_rainfall[i]) for i in range(0, 5)]
                ),
                numpy.stack(
                    [numpy.full(m_rainfall, mock_rainfall[i]) for i in range(1, 6)]
                ),
                numpy.stack(
                    [numpy.full(m_rainfall, mock_rainfall[i]) for i in range(2, 7)]
                ),
                numpy.stack(
                    [numpy.full(m_rainfall, mock_rainfall[i]) for i in range(3, 8)]
                ),
            ]
        ),
    )
