import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import pathlib
import numpy as np
import tensorflow as tf
import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

from usl_models.flood_ml.model import FloodModel
from usl_models.flood_ml.model import SpatialAttention
from usl_models.flood_ml.dataset import load_dataset_windowed_patches
from usl_models.flood_ml.Eva_sup import *
import pandas as pd

SEED = 42
keras.utils.set_random_seed(SEED)

FILECACHE_DIR = pathlib.Path("/scratch/hw4402/climateiq_filecache_us_raw")
OUTPUT_DIR = pathlib.Path("/scratch/hw4402/climateiq_output/09_21_VariedLength_6_steps_loss_no_peak/percentail_20_evaluation_hourly")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE   = 256
N_FLOOD_MAPS = 5
M_RAINFALL   = 6
N_STEMPS = 12
N_SAMPLES = 1000  # batches — more = more reliable correlation estimate

CHECKPOINTS = {
    "R14 (focal loss, 8 US cities)": pathlib.Path(
        "/scratch/hw4402/climateiq_output/09_21_VariedLength_6_steps_loss_no_peak/weights_ordered.npz"
    ),
}

# test 1 step
Atlanta = [
    "Atlanta-Atlanta_config/Rainfall_Data_1.txt",
    "Atlanta-Atlanta_config/Rainfall_Data_2.txt",
    "Atlanta-Atlanta_config/Rainfall_Data_5.txt",]
Phoenix_SM = [
    # Phoenix SM (4 scenarios)
    "Phoenix_SM-PHX_SM/Rainfall_Data_1.txt",
    "Phoenix_SM-PHX_SM/Rainfall_Data_5.txt",
    "Phoenix_SM-PHX_SM/Rainfall_Data_15.txt",
    "Phoenix_SM-PHX_SM/Rainfall_Data_16.txt",
]
Manhattan = [
    # Manhattan (8 scenarios; 7 and 15 reserved for test)
    "Manhattan-Manhattan_config/Rainfall_Data_1.txt",
    "Manhattan-Manhattan_config/Rainfall_Data_2.txt",
    "Manhattan-Manhattan_config/Rainfall_Data_5.txt",
    "Manhattan-Manhattan_config/Rainfall_Data_6.txt",
    "Manhattan-Manhattan_config/Rainfall_Data_10.txt",
    "Manhattan-Manhattan_config/Rainfall_Data_13.txt",
    "Manhattan-Manhattan_config/Rainfall_Data_16.txt",
    "Manhattan-Manhattan_config/Rainfall_Data_19.txt",
]


Phoenix_PV = [
    # Phoenix PV (2 scenarios)
    "Phoenix_PV-PHX_PV/Rainfall_Data_1.txt",
    "Phoenix_PV-PHX_PV/Rainfall_Data_5.txt",
]

Denton = [
    # Denton TX (13 timesteps, 1 chunk each)
    "Denton_TX-Denton_config/Rainfall_Data_1.txt",
    "Denton_TX-Denton_config/Rainfall_Data_2.txt",
    "Denton_TX-Denton_config/Rainfall_Data_5.txt",
]

Lafayettee = [
    # Lafayette LA (13 timesteps, 1 chunk each)
    "Lafayette_LA-Lafayette_config/Rainfall_Data_1.txt",
    "Lafayette_LA-Lafayette_config/Rainfall_Data_2.txt",
    "Lafayette_LA-Lafayette_config/Rainfall_Data_5.txt",
]

Boise = [# Boise ID (13 timesteps, 1 chunk each)
    "Boise_ID-Boise_config/Rainfall_Data_1.txt",
    "Boise_ID-Boise_config/Rainfall_Data_2.txt",
    "Boise_ID-Boise_config/Rainfall_Data_5.txt",]

Navarre = [
    # Navarre FL (13 timesteps, 1-2 chunks each)
    "Navarre_FL-Navarre_config/Rainfall_Data_1.txt",
    "Navarre_FL-Navarre_config/Rainfall_Data_2.txt",
    "Navarre_FL-Navarre_config/Rainfall_Data_5.txt",
]

New_Orleans = [
    "New_Orleans-New_Orleans_config/Rainfall_Data_1.txt",
    "New_Orleans-New_Orleans_config/Rainfall_Data_2.txt",
    "New_Orleans-New_Orleans_config/Rainfall_Data_5.txt",
]

all_sim_list = [Atlanta, Phoenix_SM, Manhattan, Phoenix_PV, Denton, Lafayettee, Boise, Navarre, New_Orleans]
city_names = ["Atlanta", "Phoenix_SM", "Manhattan", "Phoenix_PV", "Denton", "Lafayettee", "Boise", "Navarre", "New_Orleans"]

def build_model():
    params = FloodModel.Params(
        lstm_units=128,
        lstm_kernel_size=5,
        lstm_dropout=0.0,
        lstm_recurrent_dropout=0.0,
        n_flood_maps=N_FLOOD_MAPS,
        m_rainfall=M_RAINFALL,
        use_rain_broadcast=True,
        use_dilated_geo=False,
        use_group_norm=True,
        loss_version="v1",
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    )
    m = FloodModel(params=params, spatial_dims=(PATCH_SIZE, PATCH_SIZE))
    dummy = {
        "geospatial":     tf.zeros((1, PATCH_SIZE, PATCH_SIZE, 12)),
        "spatiotemporal": tf.zeros((1, N_FLOOD_MAPS, PATCH_SIZE, PATCH_SIZE, 1)),
        "temporal":       tf.zeros((1, N_FLOOD_MAPS, M_RAINFALL)),
    }
    _ = m._model(dummy)
    return m
    
def load_weights(model, path):
    data = np.load(str(path))
    model._model.set_weights([data[f"w{i:02d}"] for i in range(len(data.files))])
    print(f"  Loaded {len(data.files)} weights from {path.parent.name}")

def load_val_ds(sims, split="test"):
    return load_dataset_windowed_patches(
        filecache_dir=FILECACHE_DIR,
        sim_names=sims,
        dataset_split=split,
        patch_size=PATCH_SIZE,
        stride=128,
        batch_size=1,
        n_flood_maps=N_FLOOD_MAPS,
        m_rainfall=M_RAINFALL,
        n_future_steps=N_STEMPS,
        min_flood_fraction=0.01,
        min_max_depth=0.05,
        min_label_max_depth=0.001,
        dry_timestep_fraction=0.25,
        shuffle=False,
        temporal_feature_version=2,
    )

def predict_n_steps(model, inputs, n_steps):
    keras_model = model._model
    geospatial = inputs["geospatial"]
    temporal = inputs["temporal"]
    st = inputs["spatiotemporal"]

    batch_size = tf.shape(st)[0]
    h, w = st.shape[2], st.shape[3]
    cumul_F = tf.zeros((batch_size, h, w, 1), dtype=tf.float32)
    has_temporal_per_step = len(temporal.shape) == 4

    preds = []
    for k in range(n_steps):

        temporal_k = temporal[:, k] if has_temporal_per_step else temporal
        pred = keras_model(
            {
                "geospatial": geospatial,
                "temporal": temporal_k,
                "spatiotemporal": st,
            },
            training=False
        )
        pred = tf.nn.relu(pred)
        # pred, cumul_F = keras_model.green_ampt_gate(pred, geospatial, cumul_F)
        preds.append(pred)

        st = tf.concat([st[:, 1:, :, :, :], pred[:, tf.newaxis, :, :, :]], axis=1)
    return tf.stack(preds, axis=1)

def get_metrics(all_preds, all_labels):
    flood_mask = [0.05, False]
    safe_mask = [0.05, 0.15]
    low_mask = [0.15, 0.30]
    med_mask = [0.30, 0.60]
    dan_mask = [0.6, False]

    flood_mae = masked_mae(all_preds, all_labels, flood_mask)
    safe_mae = masked_mae(all_preds, all_labels, safe_mask)
    low_mae = masked_mae(all_preds, all_labels, low_mask)
    med_mae = masked_mae(all_preds, all_labels, med_mask)
    dan_mae = masked_mae(all_preds, all_labels, dan_mask)

    flood_csi = masked_CSI(all_preds, all_labels, flood_mask)
    safe_csi = masked_CSI(all_preds, all_labels, safe_mask)
    low_csi = masked_CSI(all_preds, all_labels, low_mask)
    med_csi = masked_CSI(all_preds, all_labels, med_mask)
    dan_csi = masked_CSI(all_preds, all_labels, dan_mask)

    flood_nse = masked_NSE(all_preds, all_labels, flood_mask)
    safe_nse = masked_NSE(all_preds, all_labels, safe_mask)
    low_nse = masked_NSE(all_preds, all_labels, low_mask)
    med_nse = masked_NSE(all_preds, all_labels, med_mask)
    dan_nse = masked_NSE(all_preds, all_labels, dan_mask)

    flood_ssim = masked_SSIM(all_preds, all_labels, flood_mask)
    safe_ssim = masked_SSIM(all_preds, all_labels, safe_mask)
    low_ssim = masked_SSIM(all_preds, all_labels, low_mask)
    med_ssim = masked_SSIM(all_preds, all_labels, med_mask)
    dan_ssim = masked_SSIM(all_preds, all_labels, dan_mask)

    prediction_peak, label_peak = percentail_peak(all_preds, all_labels, 5)
    top_k_peak_correlation = float(np.corrcoef(prediction_peak, label_peak)[0, 1])
    evaluation_result = {
        "flood_mae": flood_mae,
        "safe_mae": safe_mae,
        "low_mae": low_mae,
        "med_mae": med_mae,
        "dan_mae": dan_mae,
        "flood_csi": flood_csi,
        "safe_csi": safe_csi,
        "low_csi": low_csi,
        "med_csi": med_csi,
        "dan_csi": dan_csi,
        "flood_nse": flood_nse,
        "safe_nse": safe_nse,
        "low_nse": low_nse,
        "med_nse": med_nse,
        "dan_nse": dan_nse,
        "flood_ssim": flood_ssim,
        "safe_ssim": safe_ssim,
        "low_ssim": low_ssim,
        "med_ssim": med_ssim,
        "dan_ssim": dan_ssim,
        "top_k_peak_correlation": top_k_peak_correlation,
        "n_samples": len(all_preds)
    }
    return evaluation_result

best_model_path = list(CHECKPOINTS.items())[0][1]
model = build_model()
load_weights(model, best_model_path)

for sim_idx, sim in enumerate(all_sim_list):
    
    city_name = city_names[sim_idx]
    print("testing city: ", city_name)

    save_dir = OUTPUT_DIR / city_name
    save_dir.mkdir(parents=True, exist_ok=True)

    ds = load_val_ds(sim)
    
    pred_time_split = [[] for _ in range(N_STEMPS + 1)]
    label_time_split = [[] for _ in range(N_STEMPS + 1)]

    for inputs, labels in ds.take(N_SAMPLES):
        try:
            pred = predict_n_steps(model, inputs, N_STEMPS)
            pred = tf.reshape(pred, list(pred.shape[0:-1]))
            for i in range(N_STEMPS):
                pred_time_split[i].append(pred[:, i])
                label_time_split[i].append(labels[:, i])
            
            pred_max = tf.reduce_max(pred, axis=1)
            ###########################################
            # GET percentail value map:
            
            tensor_pred = tf.transpose(pred, perm=[0, 2, 3, 1])
            T = tf.shape(tensor_pred)[-1]
            k = tf.maximum(tf.cast(tf.math.ceil(0.20*tf.cast(T, tf.float32)), tf.int32), 1)
            top_values, _ = tf.math.top_k(tensor_pred, k=k, sorted=True)

            percential_max = top_values[..., -1] # [B, H, W, 1]

            ###########################################
            label_max = tf.reduce_max(labels, axis=1)
            # pred_time_split[-1].append(pred_max)
            pred_time_split[-1].append(percential_max)
            label_time_split[-1].append(label_max)
        except:
            print("some error occurs")
            continue
    
    if len(label_time_split[0]) == 0:
        print("no data for the city, continue next city...")
    else:
        pred_time_concated = [tf.concat(p, axis=0) for p in pred_time_split]
        label_time_concated = [tf.concat(l, axis=0) for l in label_time_split]   
        
        all_eval = []
        for i in range(N_STEMPS + 1):
            p = label_time_concated[i]
            l = pred_time_concated[i]
            evaluation_metrics = get_metrics(p, l)
            all_eval.append(evaluation_metrics)
        df = pd.DataFrame(all_eval)
        df.to_csv(save_dir / "evaluation_metrics.csv", index=False)