import pytest
from unittest.mock import patch
from unittest.mock import MagicMock
from unittest.mock import Mock
import json
from google.cloud import firestore

from usl_models.flood_ml.metastore import FirestoreDataHandler


class TestFirestoreDataHandler:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.mock_firestore_client = Mock()
        self.mock_settings = Mock()
        self.mock_settings.DEBUG = 2
        self.data_handler = FirestoreDataHandler(self.mock_firestore_client, self.mock_settings)

    def test__print_document_content(self):
        collection_info = {
            "collection1": {
                "document1": {"key1": "value1"},
                "document2": {"key2": "value2"},
            }
        }
        document_id = "document1"
        with patch("builtins.print") as mock_print:
            self.data_handler._print_document_content(collection_info, document_id)
            mock_print.assert_any_call(f"Document ID: {document_id}")
            mock_print.assert_any_call(json.dumps({"key1": "value1"}, indent=4))

    def test__is_document_in_collection(self):
        collection_info = {
            "collection1": {"document1": {"key1": "value1"}},
        }
        assert self.data_handler._is_document_in_collection(collection_info, "document1") is True
        assert self.data_handler._is_document_in_collection(collection_info, "document2") is False

    def test__find_simulation_document(self):
        collections_info = {
            "simulations": {
                "test-1-15-min-test%2FRainfall_Data_15.txt": {"label_chunks": {}},
            }
        }
        study_area = "15-min-test"
        result = self.data_handler._find_simulation_document(collections_info, study_area)
        assert result == {"document_id": "test-1-15-min-test%2FRainfall_Data_15.txt", "subcollections": {"label_chunks": {}}}

    @patch('google.cloud.firestore.Client')
    def test__list_collections_and_print_documents(self, mock_firestore_client):
        # Mock Firestore collections and documents
        mock_city_cat_rainfall_configs = MagicMock()
        mock_city_cat_rainfall_configs.id = "city_cat_rainfall_configs"
        mock_city_doc = MagicMock()
        mock_city_doc.id = "15-min-test%2FRainfall_Data_15.txt"
        mock_city_cat_rainfall_configs.stream.return_value = [mock_city_doc]
        mock_city_doc.reference.collections.return_value = []

        mock_simulations = MagicMock()
        mock_simulations.id = "simulations"
        mock_sim_doc = MagicMock()
        mock_sim_doc.id = "test-1-15-min-test%2FRainfall_Data_15.txt"
        mock_simulations.stream.return_value = [mock_sim_doc]

        mock_study_areas = MagicMock()
        mock_study_areas.id = "study_areas"
        mock_study_doc = MagicMock()
        mock_study_doc.id = "Manhattan"
        mock_study_areas.stream.return_value = [mock_study_doc]

        mock_label_chunks = MagicMock()
        mock_label_chunks.id = "label_chunks"
        
        mock_chunks = MagicMock()
        mock_chunks.id = "chunks"

        # Mock the collections() method for each document
        mock_sim_doc.reference.collections.return_value = [mock_label_chunks]
        mock_study_doc.reference.collections.return_value = [mock_chunks]

        # Mock the collections() method of the Firestore client
        mock_firestore_client.return_value.collections.return_value = [
            mock_city_cat_rainfall_configs,
            mock_simulations,
            mock_study_areas
        ]

        # Inject the mocked Firestore client into the FirestoreDataHandler instance
        self.data_handler.firestore_client = mock_firestore_client.return_value

        collections_info = self.data_handler._list_collections_and_print_documents()
        expected_info = {
            "city_cat_rainfall_configs": {"15-min-test%2FRainfall_Data_15.txt": []},
            "simulations": {"test-1-15-min-test%2FRainfall_Data_15.txt": ["label_chunks"]},
            "study_areas": {"Manhattan": ["chunks"]}
        }
        
        assert collections_info == expected_info

    @patch('google.cloud.firestore.DocumentSnapshot')
    def test__extract_rainfall_info(self, mock_document_snapshot):
        # Create a mock document snapshot with the expected dictionary
        mock_doc = MagicMock()
        mock_doc.to_dict.return_value = {
            "rainfall_duration": 60,
            "as_vector_gcs_uri": "gs://bucket/vector1.npy",
            "parent_config_name": "config1"
        }
        
        # Ensure the stream method returns a list with the mock document
        self.mock_firestore_client.collection.return_value.stream.return_value = [mock_doc]

        # Call the method to test
        rainfall_info = self.data_handler._extract_rainfall_info()
        
        # Define the expected result
        expected_info = [{
            "rainfall_duration": 60,
            "as_vector_gcs_uri": "gs://bucket/vector1.npy",
            "parent_config_name": "config1"
        }]
        
        # Assert that the result matches the expected result
        assert rainfall_info == expected_info

    @patch('google.cloud.firestore.DocumentSnapshot')
    def test__get_rainfall_durations(self, mock_document_snapshot):
        # Create mock document snapshots with the expected dictionaries
        mock_doc1 = MagicMock()
        mock_doc1.to_dict.return_value = {"rainfall_duration": 60}
        mock_doc2 = MagicMock()
        mock_doc2.to_dict.return_value = {"rainfall_duration": 30}
        
        # Ensure the stream method returns a list with the mock documents
        self.mock_firestore_client.collection.return_value.stream.return_value = [mock_doc1, mock_doc2]

        # Call the method to test
        durations = self.data_handler._get_rainfall_durations()
        
        # Assert that the result matches the expected result
        assert durations == [60, 30]

    @patch('google.cloud.firestore.DocumentSnapshot')
    def test__get_label_chunks_urls(self, mock_document_snapshot):
        # Create mock document snapshots with the expected dictionaries
        mock_doc1 = MagicMock()
        mock_doc1.to_dict.return_value = {"gcs_uri": "gs://bucket/chunk1.npy"}
        mock_doc2 = MagicMock()
        mock_doc2.to_dict.return_value = {"gcs_uri": "gs://bucket/chunk2.npy"}
        
        # Ensure the stream method returns a list with the mock documents
        self.mock_firestore_client.collection.return_value.stream.return_value = [mock_doc1, mock_doc2]

        # Call the method to test
        gcs_urls = self.data_handler._get_label_chunks_urls("document1")
        
        # Assert that the result matches the expected result
        assert gcs_urls == ["gs://bucket/chunk1.npy", "gs://bucket/chunk2.npy"]

    @patch('google.cloud.firestore.DocumentSnapshot')
    def test__get_label_chunks_urls_with_nonexistent_document(self, mock_document_snapshot):
        # Ensure the stream method returns an empty list for nonexistent document ID
        self.mock_firestore_client.collection.return_value.stream.return_value = []
        
        # Call the method to test
        gcs_urls = self.data_handler._get_label_chunks_urls("nonexistent_document")
        
        # Assert that the result is an empty list
        assert gcs_urls == []

    def test__get_label_chunks_urls_with_error(self):
        # Simulate an exception when querying Firestore
        self.mock_firestore_client.collection.side_effect = Exception("Firestore query failed")
        
        # Call the method to test
        gcs_urls = self.data_handler._get_label_chunks_urls("document1")
        
        # Assert that the result is an empty list
        assert gcs_urls == []






