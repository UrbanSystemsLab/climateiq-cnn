import io
from unittest import mock

from google.cloud import firestore  # type:ignore[attr-defined]
from google.cloud import storage  # type:ignore[attr-defined]
import numpy

from usl_models.flood_ml import dataset
from usl_models.flood_ml import prediction_dataset


@mock.patch.object(dataset, "metastore")
def test_load_prediction_dataset(mock_metastore):
    mock_firestore_client = mock.MagicMock(spec=firestore.Client)
    mock_storage_client = mock.MagicMock(spec=storage.Client)

    batch_size = 1
    m_rainfall = 6

    # Set some URLs to return from our metastore functions.
    mock_metastore.get_temporal_feature_metadata_for_prediction.return_value = {
        "as_vector_gcs_uri": "gs://temporal-features/temporal-feature.npy",
        "rainfall_duration": m_rainfall,
    }
    mock_metastore.get_spatial_feature_chunk_metadata_for_prediction.return_value = [
        {"feature_matrix_path": "gs://spatial-features/study_area/chunk_0_0"},
    ] * batch_size

    # Set up some fake data to retrieve from our mock GCS.
    mock_rainfall = numpy.pad(numpy.array([1, 2, 3, 4]), (0, 860))
    mock_spatial_features = numpy.random.rand(1000, 1000, 8)
    mock_labels = numpy.random.rand(1000, 1000, 19)

    # Set the storage client to return the right mock data for each GCS path.
    def mock_blob_func(blob_name):
        """Returns the mock data depending for blob requested."""
        if blob_name == "temporal-feature.npy":
            mock_data = mock_rainfall
        elif blob_name == "study_area/chunk_0_0":
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

    ds = prediction_dataset.load_prediction_dataset(
        study_area="study_area_name",
        city_cat_config="config_name",
        batch_size=batch_size,
        n_flood_maps=5,
        m_rainfall=m_rainfall,
        firestore_client=mock_firestore_client,
        storage_client=mock_storage_client,
    )
    batch = ds.take(1)
    tensors, metadata = batch.get_single_element()

    numpy.testing.assert_array_equal(
        metadata["feature_chunk"].numpy(), numpy.array([b"chunk_0_0"])
    )
    numpy.testing.assert_array_equal(
        metadata["rainfall"].numpy(), numpy.array([m_rainfall])
    )

    # geospatial will have the same spatial features.
    numpy.testing.assert_array_almost_equal(
        tensors["geospatial"].numpy(),
        numpy.array([mock_spatial_features] * batch_size),
    )

    # spatiotemporal will have the labels creeping into a sequence of zeros.
    numpy.testing.assert_array_almost_equal(
        tensors["spatiotemporal"].numpy(),
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
    numpy.testing.assert_array_almost_equal(
        tensors["temporal"].numpy(),
        numpy.array(
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
        ),
    )
