#!/usr/bin/env python
# coding: utf-8

# # Flood Model Training Notebook
# 
# Train a Flood ConvLSTM Model using `usl_models` lib.

# In[1]:

import sys
sys.path.insert(0, "/home/se2890/climateiq-cnn-main/usl_models")
from usl_models.flood_ml.dataset import  load_dataset_windowed_cached

import tensorflow as tf
import keras_tuner
import time
from datetime import datetime
import keras
import logging
from usl_models.flood_ml.model import FloodModel
from usl_models.flood_ml.dataset import load_dataset_windowed
import pathlib

# === CONFIG ===
# Set random seeds and GPU memory growth
logging.getLogger().setLevel(logging.WARNING)
keras.utils.set_random_seed(812)

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

timestamp = time.strftime("%Y%m%d-%H%M%S")
log_dir = f"logs/training_{timestamp}"


# In[2]:


# for fatser loading during hyperparameter tuning use this function
def get_datasets(batch_size=2):
    filecache_dir = pathlib.Path("/home/shared/climateiq/filecache")
    city_config_mapping = {"Manhattan": "Manhattan_config"}
    # rainfall_files = [7, 5, 13, 11, 9, 16, 15, 10, 12, 2, 3]
    rainfall_files = [7,5,16,15]  # Only 5 and 6
    m_rainfall = 6
    n_flood_maps = 5

    sim_names = []
    for city, config in city_config_mapping.items():
        for rain_id in rainfall_files:
            sim_names.append(f"{city}-{config}/Rainfall_Data_{rain_id}.txt")
    print("Sim names in use:")
    for s in sim_names:
        print("  ", s, (filecache_dir / s).exists())

    train_ds = load_dataset_windowed_cached(
        filecache_dir=filecache_dir,
        sim_names=sim_names,
        dataset_split="train",
        batch_size=batch_size,
        n_flood_maps=n_flood_maps,
        m_rainfall=m_rainfall,
        shuffle=True,
    )

    val_ds = load_dataset_windowed_cached(
        filecache_dir=filecache_dir,
        sim_names=sim_names,
        dataset_split="val",
        batch_size=batch_size,
        n_flood_maps=n_flood_maps,
        m_rainfall=m_rainfall,
        shuffle=True,
    )

    return train_ds, val_ds



def make_k_step_gt_batch_safe(ds, batch_size, k_steps=4):
    """
    Converts:
        (x_t, y_t) → (x_t, [y_t, y_{t+1}, ..., y_{t+k-1}])

    Output label shape:
        (B, k_steps, H, W)
    """

    # 1. Unbatch to single examples
    ds = ds.unbatch()

    # 2. Window consecutive timesteps of length k_steps
    ds = ds.window(k_steps, shift=1, drop_remainder=True)

    # 3. Batch each tensor field inside each window
    def batch_window(x_win, y_win):
        x_seq = {key: tensor.batch(k_steps)
                 for key, tensor in x_win.items()}
        y_seq = y_win.batch(k_steps)
        return tf.data.Dataset.zip((x_seq, y_seq))

    ds = ds.flat_map(batch_window)

    # 4. Pack input = first step, GT = full sequence
    def pack_k_steps(x_seq, y_seq):
        # Input stays at time t
        x_t = {key: tensor[0] for key, tensor in x_seq.items()}

        # Label sequence is length k_steps
        gt_seq = y_seq

        return x_t, gt_seq

    ds = ds.map(pack_k_steps, num_parallel_calls=tf.data.AUTOTUNE)

    # 5. Rebatch for training
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds




train_ds, val_ds = get_datasets(batch_size=2)
train_2 = make_k_step_gt_batch_safe(train_ds, batch_size=2, k_steps=2)
val_2   = make_k_step_gt_batch_safe(val_ds, batch_size=2, k_steps=2)


# In[5]:


train_ds_final = train_2
val_ds_final   = val_2


# In[6]:


print(train_ds_final.element_spec)


# In[7]:


import gc

tuner = keras_tuner.BayesianOptimization(
    FloodModel.get_hypermodel(
        lstm_units=[64],
        lstm_kernel_size=[5],
        lstm_dropout=[0.3],
        lstm_recurrent_dropout=[0.3],
        n_flood_maps=[5],
        m_rainfall=[6],
    ),
    objective="val_loss",
    max_trials=1,  # increase if you want more search
    project_name=log_dir,
)

tb_callback = keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=0,
    profile_batch=0,
)

def tuner_search(batch_size=2, num_train_samples=200, num_val_samples=100):
    """
    Run Bayesian optimization tuner on a limited number of *samples*.
    Automatically computes how many batches are needed based on batch_size.
    """
    # Clear memory and TensorFlow graph
    gc.collect()
    tf.keras.backend.clear_session()

    # Get datasets
    train_ds, val_ds = get_datasets(batch_size=batch_size)

    # Convert sample counts → batch counts
    num_train_batches = max(1, num_train_samples // batch_size)
    num_val_batches = max(1, num_val_samples // batch_size)

    print(f"Using {num_train_batches} train batches "
          f"({num_train_batches * batch_size} samples)")
    print(f"Using {num_val_batches} validation batches "
          f"({num_val_batches * batch_size} samples)")

    # Run tuner
    tuner.search(
        train_ds_final.take(num_train_batches),
        validation_data=val_ds_final.take(num_val_batches),
        epochs=1,
        callbacks=[tb_callback],
        verbose=1,
    )

# Enable GPU operation logging (optional)
tf.debugging.set_log_device_placement(True)

# Run tuner
tuner_search(batch_size=2, num_train_samples=200, num_val_samples=50)

# Retrieve best model and hyperparameters
best_hp = tuner.get_best_hyperparameters()[0]
best_model = tuner.hypermodel.build(best_hp)
print("Best hyperparameters:", best_hp.values)


# In[9]:


from keras.callbacks import ModelCheckpoint, EarlyStopping

# Build model from best hyperparameters
final_params_dict = best_hp.values.copy()
final_params = FloodModel.Params(**final_params_dict)
model = FloodModel(params=final_params)

callbacks = [
    keras.callbacks.TensorBoard(log_dir=log_dir),
    ModelCheckpoint(
        filepath=log_dir + "/checkpoint",
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        save_format="tf",
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=100,
        restore_best_weights=True,
        mode="min",
    ),
]

# IMPORTANT: use multi-step datasets
model.fit(
    train_ds_final,
    epochs=50,
    val_dataset = val_ds_final,
    callbacks=callbacks,
)

model.save_model(log_dir + "/model")

