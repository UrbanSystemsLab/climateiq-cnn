"""Full training script for FloodConvLSTM with new architecture.

Run with tmux:
    cd usl_models && python ../run_quick_train.py 2>&1 | tee ../train_output.log
"""

import os
import json
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pathlib
import numpy as np
import tensorflow as tf
import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
keras.utils.set_random_seed(SEED)
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

print(f"GPUs: {tf.config.list_physical_devices('GPU')}")

from usl_models.flood_ml.model import FloodModel
from usl_models.flood_ml.dataset import load_dataset_windowed_patches
# from usl_models.flood_ml.emissions_callback import EmissionsCallback  # disabled — causes 2-epoch stop

# =====================================================================
# CONFIGURATION
# =====================================================================
FILECACHE_DIR = pathlib.Path("/home/shared/climateiq/filecache")
OUTPUT_DIR = pathlib.Path("/home/jainr/climateiq-cnn-6/train_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 256
PATCH_STRIDE = 128
BATCH_SIZE = 4   # reverted from 6 — batch 4 had 16x better MAE (0.012 vs 0.197)
EPOCHS = 50
N_FLOOD_MAPS = 5
M_RAINFALL = 6
N_FUTURE_STEPS = 13  # match typical prediction length — model must learn to self-correct beyond step 7
DEPTH_CAP = 4.0  # metres — raised from 2.5; covers Atlanta peak (~4.4m capped at 4m)
                  # Manhattan 5-7m ponding in closed canyons above this are outliers
DRY_TIMESTEP_FRACTION = 0.12  # calibrated from prior best run to avoid dry-step dominance

CHECKPOINT_PATH = pathlib.Path(
    "/home/jainr/climateiq-cnn-6/train_output/run_20260324-044011/best_model.keras"
)

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
]

# All cities trained equally (replay phase done — model already generalizes to new cities)
# Filter to only existing sims
old_sims = [s for s in old_sims if (FILECACHE_DIR / s).exists()]
new_sims = [s for s in new_sims if (FILECACHE_DIR / s).exists()]
sim_names = old_sims + new_sims
print(f"Training on all {len(sim_names)} sims equally (batch_size={BATCH_SIZE})")
for s in sim_names:
    print(f"  {s}")


# =====================================================================
# DATA LOADING
# =====================================================================
def clip_labels(inputs, labels):
    """Cap flood depth labels at DEPTH_CAP to exclude simulator outliers."""
    return inputs, tf.minimum(labels, DEPTH_CAP)


def augment(inputs, labels):
    """Random horizontal + vertical flips — flood physics is spatially symmetric."""
    flip_lr = tf.random.uniform(()) > 0.5
    flip_ud = tf.random.uniform(()) > 0.5

    geo = inputs["geospatial"]      # (B, H, W, 9)
    spt = inputs["spatiotemporal"]  # (B, N, H, W, 1)

    # geo: H=axis1, W=axis2  |  spt: H=axis2, W=axis3
    geo = tf.cond(flip_lr, lambda: tf.reverse(geo, [2]), lambda: geo)
    geo = tf.cond(flip_ud, lambda: tf.reverse(geo, [1]), lambda: geo)
    spt = tf.cond(flip_lr, lambda: tf.reverse(spt, [3]), lambda: spt)
    spt = tf.cond(flip_ud, lambda: tf.reverse(spt, [2]), lambda: spt)

    # labels: (B, H, W) for n_future=1, or (B, K, H, W) for n_future>1
    # In both cases W is last axis, H is second-to-last.
    label_rank = len(labels.shape)
    labels = tf.cond(flip_lr, lambda: tf.reverse(labels, [label_rank - 1]), lambda: labels)
    labels = tf.cond(flip_ud, lambda: tf.reverse(labels, [label_rank - 2]), lambda: labels)

    return {**inputs, "geospatial": geo, "spatiotemporal": spt}, labels


def load_split(split, sims, shuffle=True, n_future=N_FUTURE_STEPS):
    """Load a dataset split for a given list of simulations."""
    ds = load_dataset_windowed_patches(
        filecache_dir=FILECACHE_DIR,
        sim_names=sims,
        dataset_split=split,
        patch_size=PATCH_SIZE,
        stride=PATCH_STRIDE,
        batch_size=BATCH_SIZE,
        n_flood_maps=N_FLOOD_MAPS,
        m_rainfall=M_RAINFALL,
        max_patches_per_chunk=None,
        min_flood_fraction=0.01,    # ≥1% flooded pixels — excludes all-dry patches
        min_max_depth=0.05,         # lowered from 0.1 — include early onset patches
        min_label_max_depth=0.001,  # lowered from 0.01 — include very early flood onset
        n_future_steps=n_future,
        dry_timestep_fraction=DRY_TIMESTEP_FRACTION if split == "train" else 0.0,
        shuffle=shuffle,
        temporal_feature_version=2,
    )
    ds = ds.map(clip_labels, num_parallel_calls=tf.data.AUTOTUNE)
    if split == "train":
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.repeat()
    return ds


def load_replay_train():
    """Build mixed training dataset: 70% new cities + 30% old cities (replay).

    Uses tf.data.Dataset.sample_from_datasets to interleave batches
    from new-city and old-city datasets with the configured weights.
    """
    ds_new = load_split("train", sims=new_sims, shuffle=True)
    ds_old = load_split("train", sims=old_sims, shuffle=True)
    mixed = tf.data.Dataset.sample_from_datasets(
        [ds_new, ds_old],
        weights=[REPLAY_NEW_WEIGHT, REPLAY_OLD_WEIGHT],
        seed=SEED,
    )
    return mixed


# =====================================================================
# BUILD MODEL
# =====================================================================
print("\n" + "=" * 60)
print("FLOOD MODEL TRAINING — Full Dataset")
print("=" * 60)

STEPS_PER_EPOCH = 6000
cosine_lr = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-6,        # warmup start
    decay_steps=STEPS_PER_EPOCH * EPOCHS,  # 300k total
    alpha=1e-6,                        # minimum LR at end
    warmup_target=7e-5,                # peak LR after warmup (scaled for batch=4)
    warmup_steps=1000,                 # ~1/6 of first epoch
)
params = FloodModel.Params(
    lstm_units=128,
    lstm_kernel_size=5,
    lstm_dropout=0.2,
    lstm_recurrent_dropout=0.2,
    n_flood_maps=N_FLOOD_MAPS,
    m_rainfall=M_RAINFALL,
    optimizer=keras.optimizers.Adam(learning_rate=cosine_lr, global_clipnorm=1.0),
)
model = FloodModel(params=params, spatial_dims=(PATCH_SIZE, PATCH_SIZE))

# Load pre-trained weights from ordered npz (avoids keras.models.load_model issues)
WEIGHTS_PATH = CHECKPOINT_PATH.parent / "weights_ordered.npz"
if WEIGHTS_PATH.exists():
    print(f"Loading weights: {WEIGHTS_PATH}")
    # Build model with dummy forward pass
    dummy = {
        "geospatial": tf.zeros((1, PATCH_SIZE, PATCH_SIZE, 10)),
        "spatiotemporal": tf.zeros((1, N_FLOOD_MAPS, PATCH_SIZE, PATCH_SIZE, 1)),
        "temporal": tf.zeros((1, N_FLOOD_MAPS, M_RAINFALL)),
    }
    _ = model._model(dummy)
    data = np.load(str(WEIGHTS_PATH))
    w_list = [data[f"w{i:02d}"] for i in range(len(data.files))]
    assert len(w_list) == len(model._model.weights), (
        f"Weight count mismatch: file={len(w_list)}, model={len(model._model.weights)}"
    )
    model._model.set_weights(w_list)
    print(f"Loaded {len(w_list)} weights from {WEIGHTS_PATH}")
else:
    print(f"WARNING: Weights not found at {WEIGHTS_PATH}, training from scratch")

print(f"Params: lstm_units={params.lstm_units}, kernel={params.lstm_kernel_size}")
print(f"Patch: {PATCH_SIZE}x{PATCH_SIZE}, stride={PATCH_STRIDE}")
print(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE}")
print(f"Autoregressive steps: {N_FUTURE_STEPS}")
print(f"Loss: make_hybrid_loss (log-depth MSE + 4m cap + peak penalty 0.5 + focal loss 0.5)")
print(f"Temporal: v2 (rate + cumulative + delta_rate + log_cum + running_max + frac_time)")
print("Scheduled sampling: Curriculum AR (0→0.15 ramp) + feedback noise (σ=0.02)")
print(f"Simulations: {len(sim_names)} total")
print(f"LR schedule: cosine decay 1e-6 → 7e-5 (warmup 1000 steps) → 1e-6 over {STEPS_PER_EPOCH * EPOCHS} steps")
print(f"Gradient clipping: global_clipnorm=1.0")
print(f"Feedback clip: 2.5m (consistent train/val)")

# =====================================================================
# TRAIN
# =====================================================================
print("\nLoading datasets...")
train_ds = load_split("train", sims=sim_names, shuffle=True)  # all cities equally
val_ds = load_split("val", sims=sim_names, shuffle=False)

timestamp = time.strftime("%Y%m%d-%H%M%S")
log_dir = OUTPUT_DIR / f"run_{timestamp}"
log_dir.mkdir(parents=True, exist_ok=True)

callbacks = [
    # EmissionsCallback disabled — CodeCarbon 3.2.3 stops training after 2 epochs
    # ScheduledSamplingCallback removed — now using autoregressive unrolling
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath=str(log_dir / "best_model.keras"),
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=1,
    ),
    # ReduceLROnPlateau removed — using cosine decay schedule instead
    keras.callbacks.TensorBoard(
        log_dir=str(log_dir / "tb"),
        histogram_freq=0,
    ),
]

print(f"\nTraining for up to {EPOCHS} epochs...")
print(f"Logs: {log_dir}")
t0 = time.time()

history = model.fit(
    train_dataset=train_ds,
    val_dataset=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=200,    # increased from 50 — reduces val_loss noise for LR scheduler
    callbacks=callbacks,
)

train_time = time.time() - t0
print(f"\nTraining took {train_time/60:.1f} minutes")

# Save final model
final_path = str(log_dir / "final_model.keras")
model.save_model(final_path)
print(f"Final model saved: {final_path}")

# =====================================================================
# TRAINING CURVES
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(history.history["loss"], label="Train")
axes[0].plot(history.history["val_loss"], label="Val")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

if "mean_absolute_error" in history.history:
    axes[1].plot(history.history["mean_absolute_error"], label="Train")
    axes[1].plot(history.history["val_mean_absolute_error"], label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (m)")
    axes[1].set_title("Mean Absolute Error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

if "lr" in history.history:
    axes[2].plot(history.history["lr"])
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("LR Schedule")
    axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(log_dir / "training_curves.png"), dpi=150)
plt.savefig("/home/jainr/climateiq-cnn-6/training_curve.png", dpi=150)
print("Saved training_curves.png")

# =====================================================================
# EVALUATE ON VALIDATION
# =====================================================================
print("\n" + "=" * 60)
print("PREDICTION EVALUATION")
print("=" * 60)

# Use single-step labels for eval (model predicts 1 step, compare to GT)
eval_ds = load_split("val", sims=sim_names, shuffle=False, n_future=1)
all_preds = []
all_labels = []
for inputs, labels in eval_ds.take(10):
    pred = model.call(inputs)
    pred_np = np.clip(pred.numpy(), 0, None)
    label_np = labels.numpy()
    all_preds.append(pred_np)
    all_labels.append(label_np[..., np.newaxis])  # (B,H,W) -> (B,H,W,1)

all_preds = np.concatenate(all_preds, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

print(f"Evaluated {all_preds.shape[0]} samples")
print(f"Pred range:  [{all_preds.min():.4f}, {all_preds.max():.4f}]")
print(f"Label range: [{all_labels.min():.4f}, {all_labels.max():.4f}]")
mae = np.abs(all_preds.squeeze() - all_labels.squeeze()).mean()
print(f"MAE: {mae:.5f} m")

pred_maxes = all_preds.squeeze().reshape(all_preds.shape[0], -1).max(axis=1)
label_maxes = all_labels.squeeze().reshape(all_labels.shape[0], -1).max(axis=1)
print(f"\nPer-sample pred maxes (first 10): {pred_maxes[:10].round(3)}")
print(f"Per-sample label maxes (first 10): {label_maxes[:10].round(3)}")
print(f"Pred max std: {pred_maxes.std():.4f}")

# Correlation between pred max and label max
corr = np.corrcoef(pred_maxes, label_maxes)[0, 1]
print(f"Peak depth correlation: {corr:.3f}")

# =====================================================================
# PLOT PREDICTIONS
# =====================================================================
# Select diverse samples: some dry, some light flood, some heavy flood
sorted_idx = np.argsort(label_maxes)
n_total = len(sorted_idx)
# Pick 2 low, 2 medium, 2 high
pick = []
for frac in [0.05, 0.15, 0.4, 0.6, 0.8, 0.95]:
    pick.append(sorted_idx[min(int(frac * n_total), n_total - 1)])

fig, axes = plt.subplots(len(pick), 3, figsize=(14, 4 * len(pick)))

for row, i in enumerate(pick):
    gt = all_labels[i].squeeze()
    pd = all_preds[i].squeeze()
    diff = pd - gt
    vmax = max(gt.max(), pd.max(), 0.01)

    axes[row, 0].imshow(gt, cmap="Blues", vmin=0, vmax=vmax)
    axes[row, 0].set_title(f"GT (max={gt.max():.3f}m)")
    axes[row, 0].axis("off")

    axes[row, 1].imshow(pd, cmap="Blues", vmin=0, vmax=vmax)
    axes[row, 1].set_title(f"Pred (max={pd.max():.3f}m)")
    axes[row, 1].axis("off")

    axes[row, 2].imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[row, 2].set_title(f"Diff (pred-gt)")
    axes[row, 2].axis("off")

plt.tight_layout()
plt.savefig(str(log_dir / "prediction_comparison.png"), dpi=150)
plt.savefig("/home/jainr/climateiq-cnn-6/prediction_comparison.png", dpi=150)
print("Saved prediction_comparison.png")

# =====================================================================
# SAVE CONFIG
# =====================================================================
config = {
    "timestamp": timestamp,
    "sim_names": sim_names,
    "patch_size": PATCH_SIZE,
    "patch_stride": PATCH_STRIDE,
    "batch_size": BATCH_SIZE,
    "n_future_steps": N_FUTURE_STEPS,
    "loss": "make_hybrid_loss (log-depth, 4m cap, peak_weight=2.0, focal_weight=0.5)",
    "epochs_trained": len(history.history["loss"]),
    "final_train_loss": float(history.history["loss"][-1]),
    "final_val_loss": float(history.history["val_loss"][-1]),
    "best_val_loss": float(min(history.history["val_loss"])),
    "params": params.to_dict(),
    "mae": float(mae),
    "peak_correlation": float(corr),
    "train_time_min": round(train_time / 60, 1),
    "model_path": final_path,
    "best_model_path": str(log_dir / "best_model.keras"),
}

with open(str(log_dir / "config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(f"\n{'=' * 60}")
print("TRAINING COMPLETE")
print(f"{'=' * 60}")
print(f"Epochs:            {config['epochs_trained']}")
print(f"Best val loss:     {config['best_val_loss']:.6f}")
print(f"MAE:               {mae:.5f} m")
print(f"Peak correlation:  {corr:.3f}")
print(f"Training time:     {config['train_time_min']} min")
print(f"Best model:        {config['best_model_path']}")
print(f"Final model:       {final_path}")
print(f"Logs:              {log_dir}")
