"""R10 vs R14 head-to-head validation — peak_correlation, MAE, RMSE on test split."""

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

FILECACHE_DIR = pathlib.Path("/scratch/rmj7591/climateiq/train_data")
OUTPUT_DIR = pathlib.Path("/scratch/rmj7591/climateiq/climateiq-cnn/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE   = 256
N_FLOOD_MAPS = 5
M_RAINFALL   = 6
N_SAMPLES    = 200  # batches — more = more reliable correlation estimate

CHECKPOINTS = {
    "R10 (focal loss, 8 US cities)": pathlib.Path(
        "/scratch/rmj7591/climateiq/train_output/run_20260602-162532/weights_ordered.npz"
    ),
    "R14 (ABeDice, +Copenhagen +CapeTown)": pathlib.Path(
        "/scratch/rmj7591/climateiq/train_output/run_20260716-110546/weights_ordered.npz"
    ),
}

# Test split — held-out sims not seen during early stopping
TEST_SIMS = [
    "Manhattan-Manhattan_config/Rainfall_Data_7.txt",   # reserved test sim
    "Manhattan-Manhattan_config/Rainfall_Data_15.txt",  # reserved test sim
    "Atlanta-Atlanta_config/Rainfall_Data_2.txt",
    "Atlanta-Atlanta_config/Rainfall_Data_5.txt",
    "Phoenix_SM-PHX_SM/Rainfall_Data_15.txt",
    "Denton_TX-Denton_config/Rainfall_Data_5.txt",
    "Boise_ID-Boise_config/Rainfall_Data_5.txt",
    "Navarre_FL-Navarre_config/Rainfall_Data_5.txt",
    # New cities — R14 was trained on these, good to check generalisation
    "Copenhagen-Copenhagen_config/Rainfall_Data_20.txt",
    "CapeTown_Watershed_1-CapeTown_config/Rainfall_Data_20.txt",
]
TEST_SIMS = [s for s in TEST_SIMS if (FILECACHE_DIR / s).exists()]
print(f"Test sims ({len(TEST_SIMS)}):")
for s in TEST_SIMS:
    print(f"  {s}")


def build_model():
    params = FloodModel.Params(
        lstm_units=128,
        lstm_kernel_size=7,
        lstm_dropout=0.2,
        lstm_recurrent_dropout=0.2,
        n_flood_maps=N_FLOOD_MAPS,
        m_rainfall=M_RAINFALL,
        use_rain_broadcast=True,
        use_dilated_geo=True,
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


def load_val_ds(split="test"):
    return load_dataset_windowed_patches(
        filecache_dir=FILECACHE_DIR,
        sim_names=TEST_SIMS,
        dataset_split=split,
        patch_size=PATCH_SIZE,
        stride=128,
        batch_size=4,
        n_flood_maps=N_FLOOD_MAPS,
        m_rainfall=M_RAINFALL,
        n_future_steps=1,
        min_flood_fraction=0.01,
        min_max_depth=0.05,
        min_label_max_depth=0.001,
        dry_timestep_fraction=0.0,
        shuffle=False,
        temporal_feature_version=2,
    )


def evaluate(model, ds, n_samples):
    all_preds, all_labels = [], []
    for inputs, labels in ds.take(n_samples):
        pred = model._model.predict_on_batch({
            "geospatial":     inputs["geospatial"],
            "spatiotemporal": inputs["spatiotemporal"],
            "temporal":       inputs["temporal"],
        })
        pred = np.clip(pred if isinstance(pred, np.ndarray) else pred.numpy(), 0, None)
        lnp = labels.numpy()
        lnp = lnp[..., np.newaxis] if lnp.ndim == 3 else lnp
        all_preds.append(pred)
        all_labels.append(lnp)

    preds  = np.concatenate(all_preds,  axis=0).squeeze()
    labels = np.concatenate(all_labels, axis=0).squeeze()
    mae    = float(np.abs(preds - labels).mean())
    rmse   = float(np.sqrt(((preds - labels) ** 2).mean()))
    p_pred  = preds.reshape(len(preds), -1).max(axis=1)
    p_label = labels.reshape(len(labels), -1).max(axis=1)
    corr   = float(np.corrcoef(p_pred, p_label)[0, 1])
    return {"mae": mae, "rmse": rmse, "peak_correlation": corr,
            "n_samples": len(preds), "preds": preds, "labels": labels}


# ── Run evaluation ──────────────────────────────────────────────────────────
results = {}
ds = load_val_ds("test")

for name, ckpt_path in CHECKPOINTS.items():
    print(f"\nEvaluating: {name}")
    model = build_model()
    load_weights(model, ckpt_path)
    res = evaluate(model, ds, N_SAMPLES)
    results[name] = res
    print(f"  peak_correlation : {res['peak_correlation']:.4f}")
    print(f"  MAE              : {res['mae']:.5f} m")
    print(f"  RMSE             : {res['rmse']:.5f} m")
    print(f"  Samples          : {res['n_samples']}")
    del model
    tf.keras.backend.clear_session()

# ── Summary table ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("HEAD-TO-HEAD COMPARISON (test split)")
print("=" * 60)
print(f"{'Model':<42} {'PeakCorr':>9} {'MAE(m)':>8} {'RMSE(m)':>9}")
print("-" * 60)
for name, res in results.items():
    print(f"{name:<42} {res['peak_correlation']:>9.4f} {res['mae']:>8.5f} {res['rmse']:>9.5f}")
print("=" * 60)

# ── Comparison plot ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(CHECKPOINTS), figsize=(6 * len(CHECKPOINTS), 5))
for ax, (name, res) in zip(axes, results.items()):
    short = name.split("(")[0].strip()
    p = res["preds"].reshape(len(res["preds"]), -1).max(axis=1)
    l = res["labels"].reshape(len(res["labels"]), -1).max(axis=1)
    ax.scatter(l, p, alpha=0.3, s=8)
    lim = max(l.max(), p.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xlabel("GT peak depth (m)")
    ax.set_ylabel("Pred peak depth (m)")
    ax.set_title(f"{short}\nr={res['peak_correlation']:.4f}  MAE={res['mae']:.4f}m")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "r10_vs_r14_peak_scatter.png", dpi=120)
print(f"\nPlot saved: {OUTPUT_DIR}/r10_vs_r14_peak_scatter.png")
