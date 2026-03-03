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
    # city_config_mapping = {"Manhattan": "Manhattan_config", "Atlanta": "Atlanta_config", "Phoenix_SM": "PHX_SM"}
    # rainfall_files = [7, 5, 13, 11, 9, 16, 15, 10, 12, 2, 3]
    rainfall_files = [1, 2, 7,5,16,15]  # Only 5 and 6
    # rainfall_files = [7,5,16,15]  # Only 5 and 6
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



# def make_k_step_gt_batch_safe(ds, batch_size, k_steps=4):
#     """
#     Converts:
#         (x_t, y_t) → (x_t, [y_t, y_{t+1}, ..., y_{t+k-1}])

#     Output label shape:
#         (B, k_steps, H, W)
#     """

#     # 1. Unbatch to single examples
#     ds = ds.unbatch()

#     # 2. Window consecutive timesteps of length k_steps
#     ds = ds.window(k_steps, shift=1, drop_remainder=True)

#     # 3. Batch each tensor field inside each window
#     def batch_window(x_win, y_win):
#         x_seq = {key: tensor.batch(k_steps)
#                  for key, tensor in x_win.items()}
#         y_seq = y_win.batch(k_steps)
#         return tf.data.Dataset.zip((x_seq, y_seq))

#     ds = ds.flat_map(batch_window)

#     # 4. Pack input = first step, GT = full sequence
#     def pack_k_steps(x_seq, y_seq):
#         # Input stays at time t
#         x_t = {key: tensor[0] for key, tensor in x_seq.items()}

#         # Label sequence is length k_steps
#         gt_seq = y_seq

#         return x_t, gt_seq

#     ds = ds.map(pack_k_steps, num_parallel_calls=tf.data.AUTOTUNE)

#     # 5. Rebatch for training
#     ds = ds.batch(batch_size, drop_remainder=True)
#     ds = ds.prefetch(tf.data.AUTOTUNE)

#     return ds


# if we need to filter part iof data:
PHYS_MAX = 1.5  # physical upper bound


def make_k_step_gt_batch_safe_filter(ds, batch_size, k_steps=4):
    """
    Converts:
        (x_t, y_t) → (x_t, [y_t, ..., y_{t+k-1}])

    Clips flood values to [0, 1.5] (physical range)
    Then normalizes to [0, 1] for training stability.

    Output label shape:
        (B, k_steps, H/2, W/2)
    """

    # -------------------------------------------------
    # 1. Unbatch
    # -------------------------------------------------
    ds = ds.unbatch()

    # -------------------------------------------------
    # 2. Window consecutive timesteps
    # -------------------------------------------------
    ds = ds.window(k_steps, shift=1, drop_remainder=True)

    # -------------------------------------------------
    # 3. Batch inside window
    # -------------------------------------------------
    def batch_window(x_win, y_win):
        x_seq = {k: v.batch(k_steps) for k, v in x_win.items()}
        y_seq = y_win.batch(k_steps)
        return tf.data.Dataset.zip((x_seq, y_seq))

    ds = ds.flat_map(batch_window)

    # -------------------------------------------------
    # 4. Pack input and GT sequence
    # -------------------------------------------------
    def pack_k_steps(x_seq, y_seq):
        x_t = {k: v[0] for k, v in x_seq.items()}
        return x_t, y_seq

    ds = ds.map(pack_k_steps, num_parallel_calls=tf.data.AUTOTUNE)

    # -------------------------------------------------
    # 4.5 CLIP + NORMALIZE
    # -------------------------------------------------
    def clip_and_normalize(x, y):

        # Clip physical range
        x["spatiotemporal"] = tf.clip_by_value(
            x["spatiotemporal"], 0.0, PHYS_MAX
        )
        y = tf.clip_by_value(y, 0.0, PHYS_MAX)

        # Normalize to [0,1]
        x["spatiotemporal"] = x["spatiotemporal"] / PHYS_MAX
        y = y / PHYS_MAX

        return x, y

    ds = ds.map(clip_and_normalize, num_parallel_calls=tf.data.AUTOTUNE)

    # -------------------------------------------------
    # 5. Select most flooded quadrant
    # -------------------------------------------------
    def select_most_flooded_quadrant(x, y):

        if y.shape.rank == 4:
            y = tf.squeeze(y, axis=-1)

        H = tf.shape(y)[-2]
        W = tf.shape(y)[-1]

        H2 = H // 2
        W2 = W // 2

        q0 = y[:, :H2, :W2]
        q1 = y[:, :H2, W2:]
        q2 = y[:, H2:, :W2]
        q3 = y[:, H2:, W2:]

        masses = tf.stack([
            tf.reduce_sum(q0),
            tf.reduce_sum(q1),
            tf.reduce_sum(q2),
            tf.reduce_sum(q3),
        ])

        best_idx = tf.cast(tf.argmax(masses), tf.int32)

        def crop_tensor(tensor):

            H = tf.shape(tensor)[-3]
            W = tf.shape(tensor)[-2]

            H2 = H // 2
            W2 = W // 2

            def case0(): return tensor[..., :H2, :W2, :]
            def case1(): return tensor[..., :H2, W2:, :]
            def case2(): return tensor[..., H2:, :W2, :]
            def case3(): return tensor[..., H2:, W2:, :]

            cropped = tf.switch_case(
                best_idx,
                branch_fns=[case0, case1, case2, case3]
            )

            static_shape = tensor.shape
            if static_shape.rank is not None:
                new_shape = list(static_shape)
                if static_shape[-3] is not None:
                    new_shape[-3] = static_shape[-3] // 2
                if static_shape[-2] is not None:
                    new_shape[-2] = static_shape[-2] // 2
                cropped.set_shape(new_shape)

            return cropped

        x["spatiotemporal"] = crop_tensor(x["spatiotemporal"])
        x["geospatial"] = crop_tensor(x["geospatial"])

        y = y[..., tf.newaxis]
        y = crop_tensor(y)
        y = tf.squeeze(y, axis=-1)

        return x, y

    ds = ds.map(select_most_flooded_quadrant,
                num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds




train_ds, val_ds = get_datasets(batch_size=2)
train_2 = make_k_step_gt_batch_safe_filter(train_ds, batch_size=2, k_steps=2)
val_2   = make_k_step_gt_batch_safe_filter(val_ds, batch_size=2, k_steps=2)


# In[5]:


train_ds_final = train_2
val_ds_final   = val_2


# In[6]:


print(train_ds_final.element_spec)


# In[7]:


import gc
params = FloodModel.Params(
    lstm_units=128,
    lstm_kernel_size=5,
    lstm_dropout=0.3,
    lstm_recurrent_dropout=0.3,
    n_flood_maps=5,
    m_rainfall=6,
)

model = FloodModel(
    params=params,
    spatial_dims=(500, 500)
)



# In[9]:


from keras.callbacks import ModelCheckpoint, EarlyStopping

# Build model from best hyperparameters
# final_params_dict = best_hp.values.copy()
# final_params = FloodModel.Params(**final_params_dict)
# model = FloodModel(params=final_params)

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
    epochs=1500,
    val_dataset = val_ds_final,
    callbacks=callbacks,
)

model.save_model(log_dir + "/model")

