"""Download flood simulation data (features + labels) from GCS into a local filecache.

This populates the directory that run_quick_train.py reads from (FILECACHE_DIR).

Run with:
    cd usl_models && python ../download_filecache.py
"""

import pathlib

from google.cloud import firestore
from google.cloud import storage

from usl_models.flood_ml.dataset import download_dataset

# Must match FILECACHE_DIR in run_quick_train.py.
FILECACHE_DIR = pathlib.Path("/scratch/hw4402/climateiq_filecache_us")
PROJECT = "climateiq-test"

# ── Old cities (already trained in ep29 checkpoint) ──────────────────
old_sims = [
    # Atlanta (3 scenarios)
    "Atlanta-Atlanta_config/Rainfall_Data_1.txt",
    "Atlanta-Atlanta_config/Rainfall_Data_2.txt",
    "Atlanta-Atlanta_config/Rainfall_Data_5.txt",
    # Phoenix SM (4 scenarios)
    "Phoenix_SM-PHX_SM/Rainfall_Data_1.txt",
    "Phoenix_SM-PHX_SM/Rainfall_Data_5.txt",
    "Phoenix_SM-PHX_SM/Rainfall_Data_15.txt",
    "Phoenix_SM-PHX_SM/Rainfall_Data_16.txt",
    # Manhattan (8 scenarios; 7 and 15 reserved for test)
    "Manhattan-Manhattan_config/Rainfall_Data_1.txt",
    "Manhattan-Manhattan_config/Rainfall_Data_2.txt",
    "Manhattan-Manhattan_config/Rainfall_Data_5.txt",
    "Manhattan-Manhattan_config/Rainfall_Data_6.txt",
    "Manhattan-Manhattan_config/Rainfall_Data_10.txt",
    "Manhattan-Manhattan_config/Rainfall_Data_13.txt",
    "Manhattan-Manhattan_config/Rainfall_Data_16.txt",
    "Manhattan-Manhattan_config/Rainfall_Data_19.txt",
    # Phoenix PV (2 scenarios)
    "Phoenix_PV-PHX_PV/Rainfall_Data_1.txt",
    "Phoenix_PV-PHX_PV/Rainfall_Data_5.txt",
]

# ── New cities (never seen by model) ─────────────────────────────────
new_sims = [
    # Denton TX (13 timesteps, 1 chunk each)
    "Denton_TX-Denton_config/Rainfall_Data_1.txt",
    "Denton_TX-Denton_config/Rainfall_Data_2.txt",
    "Denton_TX-Denton_config/Rainfall_Data_5.txt",
    # Lafayette LA (13 timesteps, 1 chunk each)
    "Lafayette_LA-Lafayette_config/Rainfall_Data_1.txt",
    "Lafayette_LA-Lafayette_config/Rainfall_Data_2.txt",
    "Lafayette_LA-Lafayette_config/Rainfall_Data_5.txt",
    # Boise ID (13 timesteps, 1 chunk each)
    "Boise_ID-Boise_config/Rainfall_Data_1.txt",
    "Boise_ID-Boise_config/Rainfall_Data_2.txt",
    "Boise_ID-Boise_config/Rainfall_Data_5.txt",
    # Navarre FL (13 timesteps, 1-2 chunks each)
    "Navarre_FL-Navarre_config/Rainfall_Data_1.txt",
    "Navarre_FL-Navarre_config/Rainfall_Data_2.txt",
    "Navarre_FL-Navarre_config/Rainfall_Data_5.txt",
    # New Orleans
    "New_Orleans-New_Orleans_config/Rainfall_Data_1.txt",
    "New_Orleans-New_Orleans_config/Rainfall_Data_2.txt",
    "New_Orleans-New_Orleans_config/Rainfall_Data_5.txt",

    # # Copenhagen
    # "Copenhagen-Copenhagen_config/Rainfall_Data_1.txt",
    # "Copenhagen-Copenhagen_config/Rainfall_Data_5.txt",
    # "Copenhagen-Copenhagen_config/Rainfall_Data_10.txt",
    # "Copenhagen-Copenhagen_config/Rainfall_Data_15.txt",
    # "Copenhagen-Copenhagen_config/Rainfall_Data_20.txt",
    # # CapeTown
    # "CapeTown_Watershed_1-CapeTown_config/Rainfall_Data_1.txt",
    # "CapeTown_Watershed_1-CapeTown_config/Rainfall_Data_5.txt",
    # "CapeTown_Watershed_1-CapeTown_config/Rainfall_Data_10.txt",
    # "CapeTown_Watershed_1-CapeTown_config/Rainfall_Data_15.txt",
    # "CapeTown_Watershed_1-CapeTown_config/Rainfall_Data_20.txt",
    # "CapeTown_Watershed_2-CapeTown_config/Rainfall_Data_1.txt",
    # "CapeTown_Watershed_2-CapeTown_config/Rainfall_Data_5.txt",
    # "CapeTown_Watershed_2-CapeTown_config/Rainfall_Data_10.txt",
    # "CapeTown_Watershed_2-CapeTown_config/Rainfall_Data_15.txt",
    # "CapeTown_Watershed_2-CapeTown_config/Rainfall_Data_20.txt",
]

sim_names = old_sims + new_sims

def main():
    FILECACHE_DIR.mkdir(parents=True, exist_ok=True)
    firestore_client = firestore.Client(project=PROJECT)
    storage_client = storage.Client(project=PROJECT)

    print(f"Downloading {len(sim_names)} simulations into {FILECACHE_DIR}")
    for i, sim_name in enumerate(sim_names, 1):
        sim_path = FILECACHE_DIR / sim_name
        if sim_path.exists():
            print(f"[{i}/{len(sim_names)}] SKIP (already present): {sim_name}")
            continue
        print(f"[{i}/{len(sim_names)}] Downloading: {sim_name}")
        download_dataset(
            sim_names=[sim_name],
            output_path=FILECACHE_DIR,
            firestore_client=firestore_client,
            storage_client=storage_client,
            dataset_splits=["train", "val", "test"],
            allow_missing_sim=True,
        )

    print("Done.")


if __name__ == "__main__":
    main()
