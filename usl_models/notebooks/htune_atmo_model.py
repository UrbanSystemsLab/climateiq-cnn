#!/usr/bin/env python
# coding: utf-8

# # AtmoML Hyperparameter Tuning

# In[ ]:


import os
import random
import pathlib
import logging
import keras
import keras_tuner
import tensorflow as tf
import time
from collections import defaultdict

from usl_models.atmo_ml.model import AtmoModel
from usl_models.atmo_ml import dataset, visualizer, vars

# === GPU Setup ===
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

# === Logging and Reproducibility ===
logging.getLogger().setLevel(logging.WARNING)
keras.utils.set_random_seed(812)
visualizer.init_plt()

# === Constants ===
batch_size = 4
timestamp = time.strftime("%Y%m%d-%H%M%S")
filecache_dir = pathlib.Path("/home/shared/climateiq/filecache")

# === Simulation folders ===
sim_folders = [
    "NYC_Heat_Test/NYC_summer_2000_01p",
    "NYC_Heat_Test/NYC_summer_2010_99p",
    "NYC_Heat_Test/NYC_summer_2015_50p",
    "NYC_Heat_Test/NYC_summer_2017_25p",
    "NYC_Heat_Test/NYC_summer_2018_75p",
    "PHX_Heat_Test/PHX_summer_2008_25p",
    "PHX_Heat_Test/PHX_summer_2009_50p",
    "PHX_Heat_Test/PHX_summer_2011_99p",
    "PHX_Heat_Test/PHX_summer_2015_75p",
    "PHX_Heat_Test/PHX_summer_2020_01p",
    "CPN_Heat/CPN_summer_2000_01p",
    "CPN_Heat/CPN_summer_2005_25p",
    "CPN_Heat/CPN_summer_2007_50p",
    "CPN_Heat/CPN_summer_2014_75p",
    "CPN_Heat/CPN_summer_2018_99p"
]

# === Helper: Extract 30 valid days with all 4 time steps ===
def extract_valid_dates(region_prefix, max_samples=30):
    daily_files = defaultdict(set)
    for folder in sim_folders:
        if folder.startswith(region_prefix):
            spatio_dir = filecache_dir / folder / "spatiotemporal"
            if not spatio_dir.exists():
                continue
            for fname in os.listdir(spatio_dir):
                if fname.startswith("met_em.d03.") and fname.endswith(".npz"):
                    try:
                        time_str = fname.split(".")[2]  # '2000-05-24_06:00:00'
                        date_str, hour_str = time_str.split("_")
                        hour = hour_str.split(":")[0]  # '06'
                        daily_files[(folder, date_str)].add(hour)
                    except Exception as e:
                        print(f"⚠️ Failed to parse {fname}: {e}")
    valid_keys = [
        (folder, date)
        for (folder, date), hours in daily_files.items()
        if {"00", "06", "12", "18"}.issubset(hours)
    ]
    random.shuffle(valid_keys)
    return valid_keys[:max_samples]

# === Collect keys ===
example_keys = (
    extract_valid_dates("NYC_Heat_Test", 30) +
    extract_valid_dates("PHX_Heat_Test", 30) +
    extract_valid_dates("CPN_Heat", 30)
)

# === Print diagnostics ===
print(f"✅ Loaded {len(example_keys)} example keys")
print("🔹 First 5:", example_keys[:5])

city_counts = defaultdict(int)
for sim, _ in example_keys:
    city = sim.split("_")[0]  # e.g., 'NYC'
    city_counts[city] += 1
print("📊 Region breakdown:", dict(city_counts))

# === Dataset loading ===
ds_config = dataset.Config(output_timesteps=2)

train_ds = dataset.load_dataset_cached(
    filecache_dir,
    example_keys=example_keys,
    config=ds_config,
).batch(batch_size=batch_size)

val_ds = dataset.load_dataset_cached(
    filecache_dir,
    example_keys=example_keys,
    config=ds_config,
    shuffle=False,
).batch(batch_size=batch_size)

# === GPU strategy ===
# strategy = tf.distribute.MirroredStrategy()
# print("✅ Number of devices:", strategy.num_replicas_in_sync)


# In[ ]:


for inputs, labels in train_ds.take(1):
    print("📦 Inputs:")
    for k, v in inputs.items():
        print(f"  - {k}: shape={v.shape}, dtype={v.dtype}")
    
    print("📦 Labels:")
    print(f"  - shape={labels.shape}, dtype={labels.dtype}")


# In[ ]:


# with strategy.scope():
tuner = keras_tuner.BayesianOptimization(
    AtmoModel.get_hypermodel(
        input_cnn_kernel_size=[1, 2, 5],
        lstm_kernel_size=[5, 3],
        spatial_activation=["relu"],
        st_activation=["relu"],
        lstm_activation=["relu"],
        output_activation=["tanh"],
    ),
    objective="val_loss",
    max_trials=5,
    project_name=f"logs/htune_project_{timestamp}",
)
tuner.search_space_summary()


# In[ ]:


log_dir = f"logs/htune_{timestamp}"
print(log_dir)
tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
tuner.search(train_ds, epochs=10, validation_data=val_ds, callbacks=[tb_callback])
best_model, best_hp = tuner.get_best_models()[0], tuner.get_best_hyperparameters()[0]
best_hp.values


# In[ ]:


# with strategy.scope():
    # Re-create the model using the best hyperparameters
final_params = AtmoModel.Params(**best_hp.values)
model = AtmoModel(params=final_params)
tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
model.fit(train_ds, val_ds, epochs=1000, callbacks=[tb_callback], validation_freq=1)
model.save_model(log_dir + "/model")


# In[ ]:


# Plot results
model = AtmoModel.from_checkpoint('/home/elhajjas/climateiq-cnn-10/usl_models/notebooks/logs/htune_20250529-150138/model')
input_batch, label_batch = next(iter(train_ds))
pred_batch = model.call(input_batch)

for fig in visualizer.plot_batch(
    ds_config,
    input_batch=input_batch,
    label_batch=label_batch,
    pred_batch=pred_batch,
    st_var=vars.Spatiotemporal.TT,
    sto_var=vars.SpatiotemporalOutput.RH2,
    max_examples=None,
    dynamic_colorscale=True,  # Set to True to compute from data
    unscale= True    # Revert normalization to show true values
):
    fig.show()

