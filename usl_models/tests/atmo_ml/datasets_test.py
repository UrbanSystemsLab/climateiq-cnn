import unittest
from unittest.mock import patch
from unittest.mock import MagicMock
from usl_models.atmo_ml import datasets  # Adjust import as necessary

class TestCreateAtmoDataset(unittest.TestCase):
    
    @patch("usl_models.atmo_ml.datasets.firestore.Client")
    @patch("usl_models.atmo_ml.datasets.storage.Client")
    def setUp(self, mock_firestore_client, mock_storage_client):
        # Initialize mocked Firestore and Storage clients
        self.sim_names = ["sim1"]
        self.batch_size = 4
        self.max_chunks = 2
        self.split_ratios = (0.7, 0.2, 0.1)  # Define the dataset split ratios
        self.firestore_client = MagicMock()
        self.storage_client = MagicMock()

    def test_create_atmo_dataset(self):
        """Test create_atmo_dataset function for generating train, val, and test splits."""
        
        # Call create_atmo_dataset with the required parameters
        train_dataset, val_dataset, test_dataset = datasets.create_atmo_dataset(
            sim_names=self.sim_names,
            batch_size=self.batch_size,
            split_ratios=self.split_ratios,
            max_chunks=self.max_chunks,
            firestore_client=self.firestore_client,
            storage_client=self.storage_client,
        )

        # Assertions to validate the results
        self.assertIsNotNone(train_dataset)
        self.assertIsNotNone(val_dataset)
        self.assertIsNotNone(test_dataset)
        self.assertEqual(len(train_dataset), int(self.batch_size * self.split_ratios[0]))
        self.assertEqual(len(val_dataset), int(self.batch_size * self.split_ratios[1]))
        self.assertEqual(len(test_dataset), int(self.batch_size * self.split_ratios[2]))