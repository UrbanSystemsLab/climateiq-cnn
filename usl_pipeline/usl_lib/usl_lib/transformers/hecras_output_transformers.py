import h5py
import numpy as np
import pandas as pd
import os

# Constants
GEOMETRY_GROUP = "/Geometry/2D Flow Areas"
RESULTS_GROUP = "/Results/Unsteady/Output/Output Blocks/Base Output"
TSERIES_GROUP = f"""{RESULTS_GROUP}/Unsteady Time Series/2D Flow Areas"""
TABLE_WATER_SURFACE = "Water Surface"
DOMAIN_AREA = "2D Interior Area"
CELL_CENTERS_TABLE = "Cells Center Coordinate"
OUTPUT_FILE_NAME = "cell_centers_depth.rsl"
OUTPUT_FILE_PREFIX = "cell_centers_depth"
XCEN_COLUMN = "XCen"
YCEN_COLUMN = "YCen"
DEPTH_COLUMN_PREFIX = "value"


def get_geometry_data(hdf_file, table, domain):
    """Reads geometry data from the HDF5 file."""
    try:
        with h5py.File(hdf_file, "r") as f:
            data_path = f"{GEOMETRY_GROUP}/{domain}/{table}"
            return np.array(f[data_path])
    except Exception as e:
        print(f"{data_path} is missing from the HDF!")
        print(e)
        return None


def get_depth_dataframe(hdf_file, domain):
    """Reads depth data from the HDF5 file and returns a pandas DataFrame."""
    try:
        with h5py.File(hdf_file, "r") as f:
            data_path = f"{TSERIES_GROUP}/{domain}/{TABLE_WATER_SURFACE}"
            depth_array = np.array(f[data_path]).T
            num_cols = depth_array.shape[1]
            cols = [f"{DEPTH_COLUMN_PREFIX}{i}" for i in range(num_cols)]
            print("Time Series data shape:" + str(depth_array.shape))
            return pd.DataFrame(depth_array, columns=cols)
    except Exception as e:
        print(f"{data_path} is missing from the HDF!")
        print(e)
        return None


def get_cell_centers(hdf_file, domain):
    """Gets cell center coordinates from the HDF5 file."""
    cell_centers = get_geometry_data(hdf_file, CELL_CENTERS_TABLE, domain)
    if cell_centers is not None:
        print(cell_centers.shape)
    return cell_centers


def create_rsl_files(hdf_file, output_dir, domain):
    """Creates and saves multiple RSL files."""
    os.makedirs(output_dir, exist_ok=True)
    cell_centers = get_cell_centers(hdf_file, domain)
    if cell_centers is None:
        print("Could not get cell centers data, exiting.")
        return
    depth_df = get_depth_dataframe(hdf_file, domain)
    if depth_df is None:
        print("Could not get depth dataframe, exiting.")
        return
    print(cell_centers.shape)
    cell_centers_df = pd.DataFrame(cell_centers, columns=[XCEN_COLUMN, YCEN_COLUMN])
    if len(cell_centers_df) != len(depth_df):
        print(
            f"""Length of cell centers ({len(cell_centers_df)})
            does not match the length of depth dataframe ({len(depth_df)})."""
        )
        return

    for depth_col in depth_df.columns:
        output_file_name = f"{OUTPUT_FILE_PREFIX}_{depth_col}.rsl"
        output_file_path = os.path.join(output_dir, output_file_name)

        selected_depth_df = depth_df[[depth_col]]
        cell_centers_depth_df = cell_centers_df.join(selected_depth_df, how="inner")
        cell_centers_depth_df.to_csv(output_file_path, index=False)
        print(f"Created: {output_file_path}")
