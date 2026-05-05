"""Full training script for FloodConvLSTM with new architecture.

Run with tmux:
    cd usl_models && python ../run_quick_train.py 2>&1 | tee ../train_output.log
"""

import os
import json
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Single GPU only — MirroredStrategy on replica_1 triggers NCHW transpose errors
# in ConvLSTM2D backprop (5D tensors) that crash at ~epoch 10 regardless of XLA/
# Grappler flags. Single GPU eliminates all replica-related crashes entirely.
# Multi-GPU test: use both GPUs with MirroredStrategy
# (Previous: CUDA_VISIBLE_DEVICES = "0" to avoid ConvLSTM2D backprop bug in TF 2.15)
# TF 2.16 may have fixed the bug — testing now

import pathlib
import numpy as np
import tensorflow as tf
import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
keras.utils.set_random_seed(SEED)

# ── Multi-GPU: MirroredStrategy ──────────────────────────────────────
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(f"GPUs: {gpus}")

if len(gpus) > 1:
    # ReductionToOneDevice avoids NCCL entirely — reduces on GPU:0, no peer-to-peer NCCL needed
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.ReductionToOneDevice()
    )
else:
    strategy = tf.distribute.get_strategy()
print(f"Strategy: {strategy.__class__.__name__}, num_replicas={strategy.num_replicas_in_sync}")

from usl_models.flood_ml.model import FloodModel
from usl_models.flood_ml.dataset import load_dataset_windowed_patches
# from usl_models.flood_ml.emissions_callback import EmissionsCallback  # disabled — causes 2-epoch stop

# =====================================================================
# CONFIGURATION
# =====================================================================
FILECACHE_DIR = pathlib.Path("/home/shared/climateiq/filecache")
OUTPUT_DIR = pathlib.Path("/home/rmj7591/climateiq-cnn/train_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 256
PATCH_STRIDE = 128
BATCH_SIZE = 4   # reverted from 6 — batch 4 had 16x better MAE (0.012 vs 0.197)
EPOCHS = 2  # ROUND 5 ABLATION — V3 loss only, validate before long run
N_FLOOD_MAPS = 5
M_RAINFALL = 6
N_FUTURE_STEPS = 12  # 13 → 0 windows (last_t = T - n_future = 13 - 13 = 0); 12 gives last_t=1 per chunk
DEPTH_CAP = 4.0  # metres — raised from 2.5; covers Atlanta peak (~4.4m capped at 4m)
                  # Manhattan 5-7m ponding in closed canyons above this are outliers
DRY_TIMESTEP_FRACTION = 0.12  # calibrated from prior best run to avoid dry-step dominance

# Load previous model only if explicitly needed
# For clean baseline testing, we start fresh
PRETRAINED_WEIGHTS = pathlib.Path(
    "/home/rmj7591/climateiq-cnn/train_output/run_20260416-194257/weights_ordered.npz"
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

# BASELINE TRAINING: Only old cities for clean comparison
# Filter to only existing sims
old_sims = [s for s in old_sims if (FILECACHE_DIR / s).exists()]
new_sims = [s for s in new_sims if (FILECACHE_DIR / s).exists()]
sim_names = old_sims + new_sims  # All cities — learning new flow channels
print(f"BASELINE TRAINING: {len(sim_names)} old city simulations (batch_size={BATCH_SIZE})")
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

    geo = inputs["geospatial"]      # (B, H, W, 12)
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

STEPS_PER_EPOCH = 2000  # V1 baseline — only rain_broadcast is the variable

# ── V3 ARCHITECTURE CONFIG (round 3 — proper warm-start) ──────────────
# Strategy: keep V1 loss + V1 weights, add rain_broadcast as the ONLY change.
# This way we strictly build on V1's 30+ epochs of prior knowledge.
ARCH_V3 = True                     # framework flag
WARM_START = True                  # load V3 Round 3 weights (incremental on top of V3 R3)
USE_RAIN_BROADCAST = True          # kept from R3 (neutral-positive)
USE_DILATED_GEO = False            # still disabled
USE_STORM_EMBED = False            # still disabled
USE_DEEP_DECODER = False           # ROUND 5: identical arch to V3 R3, only loss changes
USE_GROUP_NORM = True              # ROUND 7: replace decoder BN with GroupNorm — fix 2.5m ceiling
# Path to previous best checkpoint for warm-start
PRETRAINED_WEIGHTS_V3R3 = pathlib.Path(
    "/home/rmj7591/climateiq-cnn/train_output/run_20260424-041455/weights_ordered.npz"  # V3 R3 baseline
)

with strategy.scope():
    # Same LR schedule as V1 — we're fine-tuning not training from scratch
    cosine_lr = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-6,
        decay_steps=STEPS_PER_EPOCH * EPOCHS,
        alpha=1e-6,
        warmup_target=2e-5,                         # ROUND 6: V3 R3 baseline LR (back to proven)
        warmup_steps=500,
    )
    params = FloodModel.Params(
        lstm_units=128,
        lstm_kernel_size=5,
        lstm_dropout=0.0,
        lstm_recurrent_dropout=0.0,
        n_flood_maps=N_FLOOD_MAPS,
        m_rainfall=M_RAINFALL,
        use_rain_broadcast=USE_RAIN_BROADCAST,
        use_dilated_geo=USE_DILATED_GEO,
        use_storm_embed=USE_STORM_EMBED,
        use_deep_decoder=USE_DEEP_DECODER,
        use_group_norm=USE_GROUP_NORM,
        loss_version="v1",                           # ROUND 6: V1 loss — only feedback clip changed
        optimizer=keras.optimizers.Adam(learning_rate=cosine_lr, global_clipnorm=1.0),
    )
    model = FloodModel(params=params, spatial_dims=(PATCH_SIZE, PATCH_SIZE))

    if WARM_START:
        _dummy = {
            "geospatial":     tf.zeros((1, PATCH_SIZE, PATCH_SIZE, 12)),
            "spatiotemporal": tf.zeros((1, N_FLOOD_MAPS, PATCH_SIZE, PATCH_SIZE, 1)),  # external always 1ch
            "temporal":       tf.zeros((1, N_FLOOD_MAPS, M_RAINFALL)),
        }
        _ = model._model(_dummy)
        # Round 4: prefer V3 Round 3 checkpoint if available (incremental gains)
        _src_weights = PRETRAINED_WEIGHTS_V3R3 if PRETRAINED_WEIGHTS_V3R3.exists() else PRETRAINED_WEIGHTS
        print(f"✓ Warm-start source: {_src_weights.parent.name}")
        _data = np.load(str(_src_weights))
        _wlist = [_data[f"w{i:02d}"] for i in range(len(_data.files))]

        # Fix 1: geo_cnn first conv — expand 10→12 ch (existing V1 compat)
        try:
            _geo_idx = next(i for i, w in enumerate(_wlist) if w.shape == (5, 5, 10, 16))
            _wlist[_geo_idx] = np.pad(_wlist[_geo_idx], [(0,0),(0,0),(0,2),(0,0)])
            print(f"✓ Expanded geo kernel 10→12 ch at index {_geo_idx}")
        except StopIteration:
            _geo_idx = next(i for i, w in enumerate(_wlist) if w.shape == (5, 5, 12, 16))
            _wlist[_geo_idx] = _wlist[_geo_idx].copy()
            _wlist[_geo_idx][:, :, 10:12, :] = np.random.normal(0, 0.01, (5, 5, 2, 16))
            print(f"✓ Small-random init ch10/11 slice of geo kernel at index {_geo_idx}")

        # Fix 2: st_cnn_stage1 first conv — expand 1→3 ch for rain_broadcast
        # V1 had shape (5,5,1,8); V3 with rain_broadcast needs (5,5,3,8).
        # Pad 2 new input channels with small random values → they activate as model
        # learns; existing flood-depth kernel (ch0) is preserved from V1.
        if USE_RAIN_BROADCAST:
            try:
                _st_idx = next(i for i, w in enumerate(_wlist) if w.shape == (5, 5, 1, 8))
                _old = _wlist[_st_idx]
                _new = np.zeros((5, 5, 3, 8), dtype=_old.dtype)
                _new[:, :, 0:1, :] = _old                                           # preserve depth
                _new[:, :, 1:3, :] = np.random.normal(0, 0.01, (5, 5, 2, 8))        # new rain ch
                _wlist[_st_idx] = _new
                print(f"✓ Expanded st_cnn kernel 1→3 ch at index {_st_idx} (rain_broadcast)")
            except StopIteration:
                print("⚠ Could not find (5,5,1,8) st kernel — skipping rain pad")

        # Set weights carefully: V3 model may have extra layers (dilated_geo/storm_embed)
        # not in V1 weights; handle by loading only matching layers by shape.
        _model_weights = model._model.get_weights()
        if len(_wlist) == len(_model_weights):
            model._model.set_weights(_wlist)
            print(f"✓ Direct load ({len(_wlist)} tensors matched)")
        else:
            # Load by layer NAME, falling back to sequential shape-match within
            # each layer. New layers (no name-match) keep their fresh init.
            # Build R4 layer name -> (start_idx, n_weights) map from model.
            _layers = [l for l in model._model.layers if l.weights]
            _r4_names = [l.name for l in _layers]
            # R3 source: order preserved — so we can match by name if names match,
            # else by position within shared-prefix layers.
            # Simpler: walk both name lists, copy weights for matching names, skip rest.
            _r3_source = dict()  # name → list of arrays
            # Reconstruct R3 layer structure from its model (rebuild temp R3 model)
            # Simplest: shape-sequence match, but with "layer bundles" (each layer's
            # weights count must match). We use layer-by-layer: for each R4 layer,
            # if it exists in R3 (based on name present in R3's weights list sizes
            # per layer), copy. Since we can't see R3's named layers directly, we
            # fall back to sequential alignment through matching shapes WITHIN each
            # layer boundary.
            # Practical fallback: walk R4 layers, try to consume a matching sequence
            # from R3 source tensors. If layer weight-shape sequence matches next
            # batch of R3 tensors, consume them; otherwise skip the layer (fresh init).
            _matched = 0
            _j = 0
            for _l in _layers:
                _w = _l.get_weights()
                if not _w:
                    continue
                _k = len(_w)
                _is_gn = "group_normalization" in _l.name.lower()
                # ROUND 7: GroupNorm (2 weights) replacing BatchNorm (4 weights).
                # Detect the BN→GN case BEFORE the standard match (shapes coincide).
                if (_is_gn and _k == 2 and _j + 4 <= len(_wlist) and
                    all(_wlist[_j + i].shape == _w[i].shape for i in range(2))):
                    _l.set_weights([_wlist[_j], _wlist[_j + 1]])
                    _j += 4   # consume all 4 BN tensors from source
                    _matched += 2
                    print(f"  ↻ BN→GN: {_l.name} (gamma/beta loaded, moving stats skipped)")
                # Standard sequential shape-match
                elif _j + _k <= len(_wlist) and all(
                    _wlist[_j + i].shape == _w[i].shape for i in range(_k)
                ):
                    _l.set_weights([_wlist[_j + i] for i in range(_k)])
                    _j += _k
                    _matched += _k
                else:
                    # R4-only layer (or shape drift) — keep fresh init, advance past
                    # nothing in R3 (source pointer stays; we'll try next R4 layer).
                    print(f"  fresh init: {_l.name} ({_k} tensors)")
            print(f"✓ Layer-wise load: {_matched}/{sum(len(l.get_weights()) for l in _layers)} tensors restored")

        # ── ROUND 6b: reset decoder BN running stats (skipped when GN active) ──
        # decoder_bn1/bn2 running stats were corrupted by 30+ epochs of training
        # under the 2.5m feedback clip — they squash inference outputs to <1m.
        # Reset to identity (mean=0, var=1) so they repopulate cleanly under the
        # corrected 4.0m feedback clip. Conv-LSTM internal BN is healthy, leave it.
        def _find_bn_recursive(layer, prefix=""):
            found = []
            if hasattr(layer, "moving_mean") and hasattr(layer, "moving_variance"):
                found.append((prefix + layer.name, layer))
            if hasattr(layer, "layers"):
                for sl in layer.layers:
                    found += _find_bn_recursive(sl, prefix + layer.name + "/")
            return found
        _all_bns = _find_bn_recursive(model._model)
        _reset = 0
        for _name, _bn in _all_bns:
            # Reset only the corrupted decoder BN layers
            if "batch_normalization_1" in _name or "batch_normalization_2" in _name:
                _bn.moving_mean.assign(tf.zeros_like(_bn.moving_mean))
                _bn.moving_variance.assign(tf.ones_like(_bn.moving_variance))
                _reset += 1
                print(f"  ↺ Reset BN stats: {_name}")
        print(f"✓ Reset {_reset} decoder BN layers (running stats → identity)")
    else:
        print("✓ From-scratch training — no pretrained weights")

print(f"Params: lstm_units={params.lstm_units}, kernel={params.lstm_kernel_size}")
print(f"Patch: {PATCH_SIZE}x{PATCH_SIZE}, stride={PATCH_STRIDE}")
print(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE}")
print(f"Autoregressive steps: {N_FUTURE_STEPS}")
if ARCH_V3:
    print("=== V3 ROUND 4: warm-start from R3 + deep decoder ===")
    print(f"  • loss: V1 hybrid (log-depth MSE + peak + focal) — PROVEN")
    print(f"  • rain_broadcast: {params.use_rain_broadcast}  (kept from R3)")
    print(f"  • deep_decoder:   {params.use_deep_decoder}  ← NEW: 3 extra conv layers for detail")
    print(f"  • warm-start from R3: 34 tensors restored, 17 new decoder weights trained from fresh")
    print(f"  • LR peak 2e-5, STEPS_PER_EPOCH={STEPS_PER_EPOCH}, EPOCHS={EPOCHS}")
    print("  Expected: deep decoder reconstructs fine depth detail — target < 0.07 val_flooded_mae")
else:
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

class NpzCheckpointCallback(keras.callbacks.Callback):
    """Saves weights_ordered.npz alongside best_model.keras after each improvement."""

    def __init__(self, keras_path):
        super().__init__()
        self.keras_path = pathlib.Path(keras_path)
        self.best_val_loss = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        val_loss = (logs or {}).get("val_loss", float("inf"))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            npz_path = self.keras_path.parent / "weights_ordered.npz"
            ws = self.model.get_weights()
            np.savez(str(npz_path), **{f"w{i:02d}": w for i, w in enumerate(ws)})
            print(f"\n  Saved weights_ordered.npz ({len(ws)} weights)")


callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.TensorBoard(
        log_dir=str(log_dir / "tb"),
        histogram_freq=0,
        update_freq="epoch",
    ),
    NpzCheckpointCallback(keras_path=log_dir / "best_model.keras"),
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
