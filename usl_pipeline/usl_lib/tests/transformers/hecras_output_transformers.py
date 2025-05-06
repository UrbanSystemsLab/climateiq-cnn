import unittest
import os
import tempfile
import h5py
import numpy as np
import pandas as pd
from usl_lib.transformers.hecras_output_transformers import (
    create_rsl_files,
    get_cell_centers,
    get_depth_dataframe,
    get_geometry_data,
    DOMAIN_AREA,
    CELL_CENTERS_TABLE,
    TABLE_WATER_SURFACE,
    GEOMETRY_GROUP,
    TSERIES_GROUP,
    OUTPUT_FILE_PREFIX,
    XCEN_COLUMN,
    YCEN_COLUMN,
    DEPTH_COLUMN_PREFIX,
)


class TestRSLCreation(unittest.TestCase):

    def setUp(self):
        """Set up test data and environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.hdf_file_path = os.path.join(self.temp_dir, "test_data.hdf")
        self.output_dir = os.path.join(self.temp_dir, "output")
        self.domain = DOMAIN_AREA

        # Create a sample HDF5 file
        with h5py.File(self.hdf_file_path, "w") as f:
            # Create geometry data
            geometry_group = f.create_group(GEOMETRY_GROUP)
            domain_group = geometry_group.create_group(self.domain)
            domain_group.create_dataset(
                CELL_CENTERS_TABLE, data=np.array([[1, 2], [3, 4], [5, 6]])
            )

            # Create time series data
            ts_group = f.create_group(TSERIES_GROUP)
            domain_ts_group = ts_group.create_group(self.domain)
            domain_ts_group.create_dataset(
                TABLE_WATER_SURFACE,
                data=np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]]).T,
            )

    def tearDown(self):
        """Clean up the temporary directory."""
        if os.path.exists(self.temp_dir):
            import shutil

            shutil.rmtree(self.temp_dir)

    def test_get_geometry_data_success(self):
        """Test that geometry data is read correctly."""
        data = get_geometry_data(self.hdf_file_path, CELL_CENTERS_TABLE, self.domain)
        self.assertTrue(np.array_equal(data, np.array([[1, 2], [3, 4], [5, 6]])))

    def test_get_geometry_data_missing_data(self):
        """Test that get_geometry_data returns None when data is not present."""
        data = get_geometry_data(self.hdf_file_path, "invalid_table", self.domain)
        self.assertIsNone(data)

    def test_get_depth_dataframe_success(self):
        """Test that depth dataframe is created correctly."""
        df = get_depth_dataframe(self.hdf_file_path, self.domain)
        expected_df = pd.DataFrame(
            {
                f"{DEPTH_COLUMN_PREFIX}0": [10, 40, 70],
                f"{DEPTH_COLUMN_PREFIX}1": [20, 50, 80],
                f"{DEPTH_COLUMN_PREFIX}2": [30, 60, 90],
            }
        )
        pd.testing.assert_frame_equal(df, expected_df)

    def test_get_depth_dataframe_missing_data(self):
        """Test that get_depth_dataframe returns None when data is not present."""
        df = get_depth_dataframe(self.hdf_file_path, "invalid_domain")
        self.assertIsNone(df)

    def test_get_cell_centers_success(self):
        """Test that cell centers are read correctly."""
        cell_centers = get_cell_centers(self.hdf_file_path, self.domain)
        self.assertTrue(
            np.array_equal(cell_centers, np.array([[1, 2], [3, 4], [5, 6]]))
        )

    def test_get_cell_centers_missing_data(self):
        """Test that get_cell_centers returns None when data is not present."""
        cell_centers = get_cell_centers(self.hdf_file_path, "invalid_domain")
        self.assertIsNone(cell_centers)

    def test_create_rsl_files_success(self):
        """Test that RSL files are created correctly."""
        create_rsl_files(self.hdf_file_path, self.output_dir, self.domain)

        expected_files = [
            f"{OUTPUT_FILE_PREFIX}_value0.rsl",
            f"{OUTPUT_FILE_PREFIX}_value1.rsl",
            f"{OUTPUT_FILE_PREFIX}_value2.rsl",
        ]

        for file_name in expected_files:
            output_file_path = os.path.join(self.output_dir, file_name)
            self.assertTrue(
                os.path.exists(output_file_path), f"File {file_name} not found"
            )

            df = pd.read_csv(output_file_path)
            if "value0" in file_name:
                expected_df = pd.DataFrame(
                    {
                        XCEN_COLUMN: [1, 3, 5],
                        YCEN_COLUMN: [2, 4, 6],
                        f"{DEPTH_COLUMN_PREFIX}0": [10, 40, 70],
                    }
                )
            elif "value1" in file_name:
                expected_df = pd.DataFrame(
                    {
                        XCEN_COLUMN: [1, 3, 5],
                        YCEN_COLUMN: [2, 4, 6],
                        f"{DEPTH_COLUMN_PREFIX}1": [20, 50, 80],
                    }
                )
            elif "value2" in file_name:
                expected_df = pd.DataFrame(
                    {
                        XCEN_COLUMN: [1, 3, 5],
                        YCEN_COLUMN: [2, 4, 6],
                        f"{DEPTH_COLUMN_PREFIX}2": [30, 60, 90],
                    }
                )
            pd.testing.assert_frame_equal(df, expected_df, check_dtype=False)

    def test_create_rsl_files_missing_data(self):
        """Create_rsl_files does not create a file when data is not present."""
        create_rsl_files(self.hdf_file_path, self.output_dir, "invalid_domain")
        output_files = [
            file for file in os.listdir(self.output_dir) if file.endswith(".rsl")
        ]
        self.assertEqual(len(output_files), 0)

    def test_create_rsl_files_empty_data(self):
        """Test that create_rsl does not create a file when data is empty."""
        # Create a sample HDF5 file with empty data
        empty_hdf_file_path = os.path.join(self.temp_dir, "empty_test_data.hdf")
        with h5py.File(empty_hdf_file_path, "w") as f:
            # Create empty geometry data
            geometry_group = f.create_group(GEOMETRY_GROUP)
            domain_group = geometry_group.create_group(self.domain)
            domain_group.create_dataset(CELL_CENTERS_TABLE, data=np.array([]))

            # Create empty time series data
            ts_group = f.create_group(TSERIES_GROUP)
            domain_ts_group = ts_group.create_group(self.domain)
            domain_ts_group.create_dataset(TABLE_WATER_SURFACE, data=np.array([]).T)

        create_rsl_files(empty_hdf_file_path, self.output_dir, self.domain)
        output_files = [
            file for file in os.listdir(self.output_dir) if file.endswith(".rsl")
        ]
        self.assertEqual(len(output_files), 0)


if __name__ == "__main__":
    unittest.main()
