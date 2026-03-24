"""Feature importance analysis for FloodConvLSTM via gradient-based attribution.

Computes mean |gradient| of loss w.r.t. each input channel across validation
patches. Higher gradient magnitude = channel has more influence on predictions.

Also computes channel-wise Pearson correlation between each input channel
and the flood depth label (spatial correlation averaged over samples).

Run:
    cd usl_models && CUDA_VISIBLE_DEVICES=1 python ../run_feature_importance.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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
from usl_models.flood_ml.dataset import load_dataset_windowed_patches

# ── Config ──────────────────────────────────────────────────────────────
FILECACHE_DIR = pathlib.Path("/home/shared/climateiq/filecache")
MODEL_PATH = "/home/jainr/climateiq-cnn-6/train_output/run_20260310-031133/best_model.keras"
OUTPUT_DIR = pathlib.Path("/home/jainr/climateiq-cnn-6")

PATCH_SIZE = 256
N_FLOOD_MAPS = 5
M_RAINFALL = 6
N_SAMPLES = 200  # number of validation batches to analyze

GEO_CHANNEL_NAMES = [
    "ch0: Elevation",
    "ch1: Valid Mask",
    "ch2: Buildings",
    "ch3: Green Areas",
    "ch4: K_s (Hydraulic Cond.)",
    "ch5: ψ_f (Wetting Front)",
    "ch6: θ_e (Effective Porosity)",
    "ch7: θ_i (Initial Moisture)",
    "ch8: Slope",
    "ch9: DEM Sink",
]

TEMPORAL_CHANNEL_NAMES = [
    "Rainfall Rate",
    "Norm. Cumulative",
    "Rate²",
    "Log Cumulative",
    "Running Max",
    "Fractional Time",
]

sim_names = [
    "Atlanta-Atlanta_config/Rainfall_Data_1.txt",
    "Atlanta-Atlanta_config/Rainfall_Data_2.txt",
    "Manhattan-Manhattan_config/Rainfall_Data_1.txt",
    "Manhattan-Manhattan_config/Rainfall_Data_5.txt",
    "Manhattan-Manhattan_config/Rainfall_Data_13.txt",
    "Phoenix_SM-PHX_SM/Rainfall_Data_1.txt",
    "Phoenix_PV-PHX_PV/Rainfall_Data_1.txt",
    "Denton_TX-Denton_config/Rainfall_Data_1.txt",
    "Lafayette_LA-Lafayette_config/Rainfall_Data_1.txt",
    "Boise_ID-Boise_config/Rainfall_Data_1.txt",
    "Navarre_FL-Navarre_config/Rainfall_Data_1.txt",
]
sim_names = [s for s in sim_names if (FILECACHE_DIR / s).exists()]

# ── Load model ──────────────────────────────────────────────────────────
print(f"Loading model: {MODEL_PATH}")
params = FloodModel.Params(
    lstm_units=128,
    lstm_kernel_size=5,
    lstm_dropout=0.0,  # no dropout for inference
    lstm_recurrent_dropout=0.0,
    n_flood_maps=N_FLOOD_MAPS,
    m_rainfall=M_RAINFALL,
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
)
model = FloodModel(params=params, spatial_dims=(PATCH_SIZE, PATCH_SIZE))

# Build model then load weights from ordered npz
WEIGHTS_PATH = MODEL_PATH.replace("best_model.keras", "weights_ordered.npz")
dummy = {
    "geospatial": tf.zeros((1, PATCH_SIZE, PATCH_SIZE, 10)),
    "spatiotemporal": tf.zeros((1, N_FLOOD_MAPS, PATCH_SIZE, PATCH_SIZE, 1)),
    "temporal": tf.zeros((1, N_FLOOD_MAPS, M_RAINFALL)),
}
_ = model._model(dummy)
data = np.load(WEIGHTS_PATH)
w_list = [data[f"w{i:02d}"] for i in range(len(data.files))]
model._model.set_weights(w_list)
print(f"Loaded {len(w_list)} weights.")

# ── Load validation data ────────────────────────────────────────────────
print("Loading validation data...")
val_ds = load_dataset_windowed_patches(
    filecache_dir=FILECACHE_DIR,
    sim_names=sim_names,
    dataset_split="val",
    patch_size=PATCH_SIZE,
    stride=128,
    batch_size=4,
    n_flood_maps=N_FLOOD_MAPS,
    m_rainfall=M_RAINFALL,
    max_patches_per_chunk=None,
    min_flood_fraction=0.01,
    min_max_depth=0.05,
    min_label_max_depth=0.001,
    n_future_steps=1,
    dry_timestep_fraction=0.0,
    shuffle=True,
    temporal_feature_version=2,
)

# ── 1. Gradient-based feature importance ────────────────────────────────
print(f"\nComputing gradient importance over {N_SAMPLES} batches...")

geo_grad_sum = np.zeros(10)
spt_grad_sum = np.zeros(1)
tmp_grad_sum = np.zeros(M_RAINFALL)
n_counted = 0

for i, (inputs, labels) in enumerate(val_ds.take(N_SAMPLES)):
    geo = tf.Variable(inputs["geospatial"], dtype=tf.float32)
    spt = tf.Variable(inputs["spatiotemporal"], dtype=tf.float32)
    tmp = tf.Variable(inputs["temporal"], dtype=tf.float32)

    with tf.GradientTape() as tape:
        pred = model._model({"geospatial": geo, "spatiotemporal": spt, "temporal": tmp})
        loss = tf.reduce_mean(tf.abs(pred[:, :, :, 0] - labels))

    grads = tape.gradient(loss, [geo, spt, tmp])

    if grads[0] is not None:
        # Mean |grad| per channel, averaged over spatial dims and batch
        geo_grad_sum += np.abs(grads[0].numpy()).mean(axis=(0, 1, 2))
    if grads[1] is not None:
        spt_grad_sum += np.abs(grads[1].numpy()).mean(axis=(0, 1, 2, 3))
    if grads[2] is not None:
        tmp_grad_sum += np.abs(grads[2].numpy()).mean(axis=(0, 1))

    n_counted += 1
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{N_SAMPLES} batches processed")

geo_importance = geo_grad_sum / n_counted
spt_importance = spt_grad_sum / n_counted
tmp_importance = tmp_grad_sum / n_counted

print("\n" + "=" * 60)
print("GRADIENT-BASED FEATURE IMPORTANCE (mean |∂loss/∂channel|)")
print("=" * 60)

print("\nGeospatial channels:")
geo_ranked = np.argsort(geo_importance)[::-1]
for idx in geo_ranked:
    bar = "█" * int(geo_importance[idx] / geo_importance.max() * 30)
    print(f"  {GEO_CHANNEL_NAMES[idx]:30s}  {geo_importance[idx]:.6f}  {bar}")

print(f"\nSpatiotemporal (flood maps):      {spt_importance[0]:.6f}")

print("\nTemporal channels:")
tmp_ranked = np.argsort(tmp_importance)[::-1]
for idx in tmp_ranked:
    bar = "█" * int(tmp_importance[idx] / tmp_importance.max() * 30)
    print(f"  {TEMPORAL_CHANNEL_NAMES[idx]:30s}  {tmp_importance[idx]:.6f}  {bar}")

# ── 2. Channel-label correlation ────────────────────────────────────────
print(f"\n{'=' * 60}")
print("CHANNEL-LABEL PEARSON CORRELATION (spatial, averaged over samples)")
print("=" * 60)

geo_corrs = np.zeros(10)
tmp_corrs = np.zeros(M_RAINFALL)
n_corr = 0

for i, (inputs, labels) in enumerate(val_ds.take(N_SAMPLES)):
    geo_np = inputs["geospatial"].numpy()  # (B, H, W, 10)
    lab_np = labels.numpy()                # (B, H, W)

    for b in range(geo_np.shape[0]):
        flat_lab = lab_np[b].flatten()
        if flat_lab.std() < 1e-6:
            continue
        for c in range(10):
            flat_ch = geo_np[b, :, :, c].flatten()
            if flat_ch.std() < 1e-6:
                continue
            geo_corrs[c] += np.corrcoef(flat_ch, flat_lab)[0, 1]

        # Temporal: correlate temporal channel values with per-pixel flood depth
        tmp_np = inputs["temporal"].numpy()  # (B, N, M)
        for c in range(M_RAINFALL):
            tmp_val = tmp_np[b, :, c].mean()
            if np.isfinite(tmp_val):
                tmp_corrs[c] += tmp_val * flat_lab.mean()  # accumulate product for later

        n_corr += 1

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{N_SAMPLES} batches processed")

geo_corrs /= max(n_corr, 1)
tmp_corrs /= max(n_corr, 1)

print("\nGeospatial channel vs. flood depth (Pearson r):")
geo_corr_ranked = np.argsort(np.abs(geo_corrs))[::-1]
for idx in geo_corr_ranked:
    sign = "+" if geo_corrs[idx] >= 0 else "-"
    bar = "█" * int(abs(geo_corrs[idx]) / max(abs(geo_corrs).max(), 1e-9) * 30)
    print(f"  {GEO_CHANNEL_NAMES[idx]:30s}  {sign}{abs(geo_corrs[idx]):.4f}  {bar}")

print("\nTemporal channel vs. flood depth (mean product, higher = more associated):")
tmp_corrs /= max(n_corr, 1)
tmp_corr_ranked = np.argsort(np.abs(tmp_corrs))[::-1]
for idx in tmp_corr_ranked:
    val = tmp_corrs[idx]
    sign = "+" if val >= 0 else "-"
    max_abs = max(np.nanmax(np.abs(tmp_corrs)), 1e-9)
    bar_len = int(abs(val) / max_abs * 30) if np.isfinite(val) else 0
    bar = "█" * bar_len
    print(f"  {TEMPORAL_CHANNEL_NAMES[idx]:30s}  {sign}{abs(val):.6f}  {bar}")

# ── 3. Visualization ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Geospatial gradient importance
colors_geo = plt.cm.viridis(np.linspace(0.3, 0.9, 10))
sorted_geo = np.argsort(geo_importance)
axes[0].barh(range(10), geo_importance[sorted_geo], color=colors_geo)
axes[0].set_yticks(range(10))
axes[0].set_yticklabels([GEO_CHANNEL_NAMES[i] for i in sorted_geo], fontsize=9)
axes[0].set_xlabel("Mean |gradient|")
axes[0].set_title("Geospatial Feature Importance\n(gradient-based)")
axes[0].grid(axis="x", alpha=0.3)

# Temporal gradient importance
colors_tmp = plt.cm.plasma(np.linspace(0.3, 0.9, M_RAINFALL))
sorted_tmp = np.argsort(tmp_importance)
axes[1].barh(range(M_RAINFALL), tmp_importance[sorted_tmp], color=colors_tmp)
axes[1].set_yticks(range(M_RAINFALL))
axes[1].set_yticklabels([TEMPORAL_CHANNEL_NAMES[i] for i in sorted_tmp], fontsize=9)
axes[1].set_xlabel("Mean |gradient|")
axes[1].set_title("Temporal Feature Importance\n(gradient-based)")
axes[1].grid(axis="x", alpha=0.3)

# Geospatial correlation with flood depth
sorted_corr = np.argsort(np.abs(geo_corrs))
bar_colors = ["#d62728" if geo_corrs[i] < 0 else "#2ca02c" for i in sorted_corr]
axes[2].barh(range(10), geo_corrs[sorted_corr], color=bar_colors)
axes[2].set_yticks(range(10))
axes[2].set_yticklabels([GEO_CHANNEL_NAMES[i] for i in sorted_corr], fontsize=9)
axes[2].set_xlabel("Pearson r with flood depth")
axes[2].set_title("Geospatial-Flood Correlation\n(green=positive, red=negative)")
axes[2].axvline(0, color="k", linewidth=0.5)
axes[2].grid(axis="x", alpha=0.3)

plt.tight_layout()
out_path = OUTPUT_DIR / "feature_importance.png"
plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_path}")

# ── Save numerical results ──────────────────────────────────────────────
import json
results = {
    "geo_gradient_importance": {GEO_CHANNEL_NAMES[i]: float(geo_importance[i]) for i in range(10)},
    "temporal_gradient_importance": {TEMPORAL_CHANNEL_NAMES[i]: float(tmp_importance[i]) for i in range(M_RAINFALL)},
    "spatiotemporal_gradient_importance": float(spt_importance[0]),
    "geo_flood_correlation": {GEO_CHANNEL_NAMES[i]: float(geo_corrs[i]) for i in range(10)},
    "temporal_flood_correlation": {TEMPORAL_CHANNEL_NAMES[i]: float(tmp_corrs[i]) for i in range(M_RAINFALL)},
    "n_samples": n_counted,
    "model_path": MODEL_PATH,
}
json_path = OUTPUT_DIR / "feature_importance.json"
with open(str(json_path), "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved: {json_path}")
print("\nDone!")
