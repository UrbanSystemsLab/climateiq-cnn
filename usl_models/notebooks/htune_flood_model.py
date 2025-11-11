import sys
import tensorflow as tf
import keras_tuner
import time
from datetime import datetime
import keras
import logging
from usl_models.flood_ml.model import FloodModel
import pathlib
from usl_models.flood_ml.dataset import load_dataset_windowed_cached
from keras.callbacks import ModelCheckpoint, EarlyStopping

sys.path.insert(0, "/home/se2890/climateiq-cnn-9/usl_models")


# === CONFIG ===
# Set random seeds and GPU memory growth
logging.getLogger().setLevel(logging.WARNING)
keras.utils.set_random_seed(812)

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

timestamp = time.strftime("%Y%m%d-%H%M%S")
log_dir = f"logs/training_{timestamp}"

filecache_dir = pathlib.Path("/home/shared/climateiq/filecache")
city_config_mapping = {
    "Manhattan": "Manhattan_config",
    # "Atlanta": "Atlanta_config",
    # "Phoenix_SM": "PHX_SM",
    # "Phoenix_PV": "PHX_PV",
    # "Phoenix_central": "PHX_CCC"
    # "Atlanta_Prediction": "Atlanta_config",
}
# Rainfall files you want
rainfall_files = [7, 5, 13, 11, 9, 16, 15, 10, 12, 2, 3]  # Only 5 and 6
# rainfall_files = [5]  # Only 5 and 6
dataset_splits = ["test", "train", "val"]
n_flood_maps = 5
m_rainfall = 6
batch_size = 4
epochs = 2
# Generate sim_names
sim_names = []
for city, config in city_config_mapping.items():
    for rain_id in rainfall_files:
        sim_name = f"{city}-{config}/Rainfall_Data_{rain_id}.txt"
        sim_names.append(sim_name)

# === STEP 1: DOWNLOAD DATASET TO FILECACHE ===
# print("Downloading simulations into local cache")
# download_dataset(
#     sim_names=sim_names,
#     output_path=filecache_dir,
#     dataset_splits=dataset_splits,
#    include_labels=True
# )


# print(":white_check_mark: Download complete.")
# # === STEP 2: LOAD CACHED WINDOWED DATASETS ===
# print("open_file_folder: Loading datasets from cache")
train_dataset = load_dataset_windowed_cached(
    filecache_dir=filecache_dir,
    sim_names=sim_names,
    dataset_split="train",
    batch_size=batch_size,
    n_flood_maps=n_flood_maps,
    m_rainfall=m_rainfall,
    shuffle=True,
).prefetch(tf.data.AUTOTUNE)
validation_dataset = load_dataset_windowed_cached(
    filecache_dir=filecache_dir,
    sim_names=sim_names,
    dataset_split="val",
    batch_size=batch_size,
    n_flood_maps=n_flood_maps,
    m_rainfall=m_rainfall,
    shuffle=True,
).prefetch(tf.data.AUTOTUNE)
test_dataset = load_dataset_windowed_cached(
    filecache_dir=filecache_dir,
    sim_names=sim_names,
    dataset_split="test",
    batch_size=batch_size,
    n_flood_maps=n_flood_maps,
    m_rainfall=m_rainfall,
    shuffle=True,
).prefetch(tf.data.AUTOTUNE)


tuner = keras_tuner.BayesianOptimization(
    FloodModel.get_hypermodel(
        lstm_units=[32, 64, 128],
        lstm_kernel_size=[3, 5],
        lstm_dropout=[0.2, 0.3],
        lstm_recurrent_dropout=[0.2, 0.3],
        n_flood_maps=[5],
        m_rainfall=[6],
    ),
    objective="val_loss",
    max_trials=1,
    project_name=f"logs/htune_project_{timestamp}",
)

tuner.search_space_summary()


timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"logs/htune_project_{timestamp}"

tuner = keras_tuner.BayesianOptimization(
    FloodModel.get_hypermodel(
        lstm_units=[32, 64, 128],
        lstm_kernel_size=[3, 5],
        lstm_dropout=[0.2, 0.3],
        lstm_recurrent_dropout=[0.2, 0.3],
        n_flood_maps=[5],
        m_rainfall=[6],
    ),
    objective="val_loss",
    max_trials=2,
    project_name=log_dir,  # use same path for both tuner and logs
    overwrite=True,  # ensures no leftover checkpoint confusion
    directory=None,
)

print("Log dir:", log_dir)

tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Run tuning
tuner.search(
    train_dataset,
    epochs=2,
    validation_data=validation_dataset,
    callbacks=[tb_callback],
    verbose=2,
)

# If successful, get best model
results = tuner.oracle.get_best_trials(num_trials=1)
print("Trials completed:", len(results))

if results:
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("✅ Best hyperparameters:", best_hp.values)
else:
    print("⚠️ No trials completed; check earlier logs for failed trial messages.")


# Define final parameters and model
final_params_dict = best_hp.values.copy()
final_params = FloodModel.Params(**final_params_dict)
model = FloodModel(params=final_params)
# Define callbacks
callbacks = [
    keras.callbacks.TensorBoard(log_dir=log_dir),
    ModelCheckpoint(
        filepath=log_dir + "/checkpoint",
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        save_format="tf",
    ),
    EarlyStopping(  # <--- ADD THIS
        monitor="val_loss",  # What to monitor
        patience=100,  # Number of epochs with no improvement to wait
        restore_best_weights=True,  # Restore model weights from best epoch
        mode="min",  # "min" because lower val_loss is better
    ),
]

# Train
model.fit(train_dataset, validation_dataset, epochs=2, callbacks=callbacks)

# Save final model
model.save_model(log_dir + "/model")
