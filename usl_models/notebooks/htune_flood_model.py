#!/usr/bin/env python
# coding: utf-8

# # Flood Model Training Notebook
# 
# Train a Flood ConvLSTM Model using `usl_models` lib.

# In[ ]:


import tensorflow as tf
import keras_tuner
import time
import keras
import logging
from usl_models.flood_ml import constants
from usl_models.flood_ml.model import FloodModel
from usl_models.flood_ml.model_params import FloodModelParams
from usl_models.flood_ml.dataset import load_dataset_windowed, load_dataset

# Setup
logging.getLogger().setLevel(logging.WARNING)
keras.utils.set_random_seed(812)

for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
    
timestamp = time.strftime("%Y%m%d-%H%M%S")

# Cities and their config folders
city_config_mapping = {
    "Manhattan": "Manhattan_config",
    #  "Atlanta": "Atlanta_config",
    # "Phoenix_SM": "PHX_SM",
    # "Phoenix_PV": "PHX_PV",
}

# Rainfall files you want
rainfall_files = [5, 6]  # Only 5 and 6

# Generate sim_names
sim_names = []
for city, config in city_config_mapping.items():
    for rain_id in rainfall_files:
        sim_name = f"{city}-{config}/Rainfall_Data_{rain_id}.txt"
        sim_names.append(sim_name)

print(f"Training on {len(sim_names)} simulations.")
for s in sim_names:
    print(s)

# Now load dataset
train_dataset = load_dataset_windowed(
    sim_names=sim_names,
    batch_size=4,
    dataset_split='train'
).cache()

validation_dataset = load_dataset_windowed(
    sim_names=sim_names,
    batch_size=4,
    dataset_split='val'
).cache()

# Now you can pass these into your model training like usual


# In[ ]:


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
        max_trials=10,
        project_name=f"logs/htune_project_{timestamp}",
)

tuner.search_space_summary()


# In[ ]:


log_dir = f"logs/htune_project_{timestamp}"
print(log_dir)
tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
tuner.search(train_dataset, epochs=20, validation_data=validation_dataset , callbacks=[tb_callback])
best_model, best_hp = tuner.get_best_models()[0], tuner.get_best_hyperparameters()[0]
best_hp.values


# In[ ]:


from keras.callbacks import ModelCheckpoint, EarlyStopping

# Define final parameters and model
final_params = FloodModel.Params(**best_hp.values)
model = FloodModel(params=final_params)

# Define callbacks
callbacks = [
    keras.callbacks.TensorBoard(log_dir=log_dir),
    ModelCheckpoint(
        filepath=log_dir + "/checkpoint",
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        save_format="tf"
    ),
    EarlyStopping(               # <--- ADD THIS
        monitor="val_loss",       # What to monitor
        patience=5,              # Number of epochs with no improvement to wait
        restore_best_weights=True, # Restore model weights from best epoch
        mode="min"                # "min" because lower val_loss is better
    )
]

# Train
model.fit(
    train_dataset,
    validation_dataset,
    epochs=50,
    callbacks=callbacks
)

# Save final model
model.save_model(log_dir + "/model")


# In[ ]:


# # Test calling the model on some data.
# inputs, labels_ = next(iter(train_dataset))
# prediction = model.call(inputs)
# prediction.shape


# In[ ]:


# import tensorflow as tf
# # Path to your saved model
# model_path = "/home/elhajjas/climateiq-cnn-4/logs/htune_project_20250516-142006/model"

# # Load the model
# model = tf.keras.models.load_model(model_path)


# In[ ]:


# # # Test calling the model for n predictions
# full_dataset = load_dataset(sim_names=sim_names, batch_size=1, dataset_split= "train")
# inputs, labels = next(iter(full_dataset))
# predictions = model.call_n(inputs, n=4)
# # predictions.shape


# In[ ]:


# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt

# from usl_models.flood_ml.dataset import load_dataset_windowed
# from usl_models.flood_ml import constants

# # # Path to trained model
# # model_path = "/home/elhajjas/climateiq-cnn-4/usl_models/notebooks/logs/htune_project_20250508-202940/model"
# # model = tf.keras.models.load_model(model_path)

# # Number of samples to visualize
# n_samples = 20

# # Loop through the dataset and predict
# for i, (input_data, ground_truth) in enumerate(train_dataset.take(n_samples)):
#     ground_truth = ground_truth.numpy().squeeze()
#     prediction = model(input_data).numpy().squeeze()

#     print(f"\nSample {i+1} Prediction Stats:")
#     print("  Min:", prediction.min())
#     print("  Max:", prediction.max())
#     print("  Mean:", prediction.mean())

#     # Choose timestep to plot
#     timestep = 3
#     gt_t = ground_truth[timestep]
#     pred_t = prediction[timestep]
#     vmax_val = max(gt_t.max(), pred_t.max())

#     # Plot Ground Truth and Prediction
#     fig, axes = plt.subplots(1, 2, figsize=(14, 6))
#     fig.suptitle(f"Sample {i+1} - Timestep {timestep}", fontsize=16)

#     im1 = axes[0].imshow(gt_t, cmap="Blues", vmin=0, vmax=vmax_val)
#     axes[0].set_title("Ground Truth")
#     axes[0].axis("off")
#     plt.colorbar(im1, ax=axes[0], shrink=0.8)

#     im2 = axes[1].imshow(pred_t, cmap="Blues", vmin=0, vmax=vmax_val)
#     axes[1].set_title("Prediction")
#     axes[1].axis("off")
#     plt.colorbar(im2, ax=axes[1], shrink=0.8)

#     plt.tight_layout()
#     plt.show()


# In[ ]:


# pip install scikit-image


# In[ ]:


# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from skimage.metrics import structural_similarity as ssim
# import pandas as pd

# # Load your trained model
# model_path = "/home/elhajjas/climateiq-cnn-4/logs/htune_project_20250516-142006/model"
# model = tf.keras.models.load_model(model_path)

# # Assuming validation_dataset is already defined
# # Example:
# # from usl_models.flood_ml.dataset import load_dataset_windowed
# # validation_dataset = load_dataset_windowed(...)

# n_samples = 20
# timestep = 3
# metrics_list = []

# for i, (input_data, ground_truth) in enumerate(validation_dataset.take(n_samples)):
#     ground_truth = ground_truth.numpy().squeeze()
#     prediction = model(input_data).numpy().squeeze()

#     gt_t = ground_truth[timestep]
#     pred_t = prediction[timestep]
#     vmax_val = np.nanpercentile([gt_t, pred_t], 99.5)

#     # Mask out NaNs
#     mask = ~np.isnan(gt_t)
#     gt_flat = gt_t[mask].flatten()
#     pred_flat = pred_t[mask].flatten()

#     mae = mean_absolute_error(gt_flat, pred_flat)
#     rmse = np.sqrt(mean_squared_error(gt_flat, pred_flat))
#     bias = np.mean(pred_flat) - np.mean(gt_flat)
#     iou = np.logical_and(gt_flat > 0.1, pred_flat > 0.1).sum() / max(1, np.logical_or(gt_flat > 0.1, pred_flat > 0.1).sum())
#     ssim_val = ssim(gt_t, pred_t, data_range=gt_t.max() - gt_t.min())

#     metrics_list.append({
#         "Sample": i+1,
#         "MAE": mae,
#         "RMSE": rmse,
#         "Bias": bias,
#         "IoU > 0.1": iou,
#         "SSIM": ssim_val
#     })

#     # Plot
#     fig, axes = plt.subplots(1, 2, figsize=(14, 6))
#     fig.suptitle(f"Sample {i+1} - Timestep {timestep}", fontsize=16)

#     im1 = axes[0].imshow(gt_t, cmap="Blues", vmin=0, vmax=vmax_val)
#     axes[0].set_title("Ground Truth")
#     axes[0].axis("off")
#     plt.colorbar(im1, ax=axes[0], shrink=0.8)

#     im2 = axes[1].imshow(pred_t, cmap="Blues", vmin=0, vmax=vmax_val)
#     axes[1].set_title("Prediction")
#     axes[1].axis("off")
#     plt.colorbar(im2, ax=axes[1], shrink=0.8)

#     plt.tight_layout()
#     plt.show()

# # Convert to DataFrame
# df = pd.DataFrame(metrics_list)
# print("\n=== Metrics Summary ===")
# print(df.describe())

