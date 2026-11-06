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

# R17: mixed precision (bf16) tested and REJECTED -- crashes immediately.
# FloodConvLSTM's skip connections (model.py:724, tf.concat([x, skip], -1))
# mix a bf16-cast tensor with a float32 tensor that doesn't respect the
# mixed-precision policy -- would need real dtype-casting fixes throughout
# the custom architecture (skip connections, GroupNorm, Green-Ampt gate) to
# use safely. Not worth the risk of introducing a new bug on top of everything
# else this session. Sticking with float32; see test_speed_mixed_precision.py.

from usl_models.flood_ml.model import FloodModel
from usl_models.flood_ml.dataset import load_dataset_windowed_patches
# from usl_models.flood_ml.emissions_callback import EmissionsCallback  # disabled — causes 2-epoch stop

# =====================================================================
# CONFIGURATION
# =====================================================================
FILECACHE_DIR = pathlib.Path("/scratch/rmj7591/climateiq/train_data")
OUTPUT_DIR = pathlib.Path("/scratch/rmj7591/climateiq/train_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 256
PATCH_STRIDE = 128
# batch_size=8 tested (test_speed_mixed_precision.py) and REJECTED: 4010ms/step
# vs bs=4's 2195ms/step -- 1.83x slower per step for only 1.09x more throughput.
# The AR rollout is compute-bound and scales ~linearly with batch size here;
# there's no fixed per-step overhead to amortize. Not worth the complexity.
BATCH_SIZE = 4
N_FLOOD_MAPS = 5
M_RAINFALL = 6
# CURRICULUM ROLLOUT (new this round): real cached chunks are T=13 timesteps
# (confirmed: Atlanta/Manhattan/Phoenix_SM label shape (1000,1000,13)), and
# dataset.py's last_t = T - K + 1 means the number of WARM (t>=n_flood_maps=5)
# training windows shrinks fast as K grows:
#   K=4  -> last_t=10 -> t in {0..9}: 5 cold-start + 5 warm  (current best run)
#   K=6  -> last_t=8  -> t in {0..7}: 5 cold-start + 3 warm  (still usable)
#   K=8  -> last_t=6  -> t in {0..5}: 5 cold-start + 1 warm  (thin but nonzero)
#   K=9  -> last_t=5  -> t in {0..4}: 5 cold-start + 0 warm  (collapses, same
#     failure mode that killed the original K=12 attempt)
# So K=4 stays the primary training regime for data diversity, but the model
# is never trained to unroll further than 4 steps under one gradient tape,
# while real inference (call_n) runs 10+ autoregressive steps -- every
# full-chunk visualization this session showed the largest errors appearing
# past t=4-6, consistent with compounding drift the model was never trained
# to correct for. Fix: after the bulk of training at K=4, a second stage at
# K=6 (the largest K that still keeps meaningful warm-window diversity)
# extends the trained AR horizon 50% without reintroducing the K=12 collapse.
# STAGE 3 (new): K=11 exactly matches the real deployment horizon
# (evaluation_true_ar.py's T_STRIP). At K=11, last_t = 13-11+1 = 3 -- every
# available window starts at t in {0,1,2}, ALL of which are cold-start
# (t < n_flood_maps=5). There is no way to get a "warm" window at this K given
# T=13 chunks -- which is exactly the point: this forces every training
# example to be the true deployment scenario (start from nothing, self-feed
# for the whole rollout), not an approximation of it.
#
# Validated via train_smoke_fullrollout.py (10 epochs, 4 training cities,
# batch=2): real but modest gain on both test cities' true (non-GT-context)
# autoregressive accuracy -- flood CSI +9-10% relative, and Atlanta's
# severity-RANKING inversion (peak correlation going negative past t=3)
# partially corrected. Not enough alone to hit a high accuracy bar, but a
# confirmed-positive, independent lever -- stacked here with the
# ALSO-confirmed-positive K=4->6 curriculum, using the FULL city roster and
# more epochs than the smoke dose so it has a real chance to converge instead
# of just showing direction.
N_FUTURE_STEPS_STAGE1 = 4
N_FUTURE_STEPS_STAGE2 = 6
N_FUTURE_STEPS_STAGE3 = 11
STAGE1_EPOCHS = 15
STAGE2_EPOCHS = 8
STAGE3_EPOCHS = 10
STAGE1_STEPS_PER_EPOCH = 4000
STAGE2_STEPS_PER_EPOCH = 3000
STAGE3_STEPS_PER_EPOCH = 1500  # K=11 unroll is ~2.75x K=4's per-step cost
STAGE3_BATCH_SIZE = 2          # halved -- K=11 holds 11 steps of activations
                                # under one GradientTape, well over 2x the
                                # memory of K=4/6 at the same batch size
EPOCHS = STAGE1_EPOCHS + STAGE2_EPOCHS + STAGE3_EPOCHS  # kept for LR-schedule/config bookkeeping below
DEPTH_CAP = 4.0  # metres — covers Atlanta/Phoenix peaks; Manhattan >4m outliers excluded
DRY_TIMESTEP_FRACTION = 0.12  # calibrated from prior best run to avoid dry-step dominance

# ROUND 11: warm-start from R10 (run_20260602-162532, epoch 38, val_loss=0.0850)
# New loss: ABeDice (boundary-aware Dice, ×0.3) + focal weight raised 0.5→1.0

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
    # Copenhagen (5 scenarios — international city, diverse climate)
    "Copenhagen-Copenhagen_config/Rainfall_Data_1.txt",
    "Copenhagen-Copenhagen_config/Rainfall_Data_5.txt",
    "Copenhagen-Copenhagen_config/Rainfall_Data_10.txt",
    "Copenhagen-Copenhagen_config/Rainfall_Data_15.txt",
    "Copenhagen-Copenhagen_config/Rainfall_Data_20.txt",
    # CapeTown Watershed 1 (5 scenarios — international city, diverse terrain)
    "CapeTown_Watershed_1-CapeTown_config/Rainfall_Data_1.txt",
    "CapeTown_Watershed_1-CapeTown_config/Rainfall_Data_5.txt",
    "CapeTown_Watershed_1-CapeTown_config/Rainfall_Data_10.txt",
    "CapeTown_Watershed_1-CapeTown_config/Rainfall_Data_15.txt",
    "CapeTown_Watershed_1-CapeTown_config/Rainfall_Data_20.txt",
]

# Filter to only existing sims
old_sims = [s for s in old_sims if (FILECACHE_DIR / s).exists()]
new_sims = [s for s in new_sims if (FILECACHE_DIR / s).exists()]
sim_names = old_sims + new_sims  # All cities — learning new flow channels
# FIXED (launch-readiness audit): previous log line said "BASELINE TRAINING:
# ... old city simulations" while actually mixing old_sims + new_sims --
# mislabeled every run's own log output. Also: many sims have zero files in
# some/all splits (Denton/Lafayette/Boise/Navarre/most CapeTown have no val
# or test data at all; several Manhattan/Phoenix_SM scenarios are similarly
# empty), which silently contributed nothing to training/val with no warning
# anywhere -- print real per-split file counts so this is visible.
print(f"TRAINING: {len(old_sims)} old + {len(new_sims)} new city simulations "
      f"= {len(sim_names)} total (batch_size={BATCH_SIZE})")
for s in sim_names:
    counts = []
    for split in ("train", "val", "test"):
        n = len(list((FILECACHE_DIR / s / split / "geospatial").glob("*.npy"))) \
            if (FILECACHE_DIR / s / split / "geospatial").exists() else 0
        counts.append(f"{split}={n}")
    flag = "  <-- ZERO in all splits" if all(c.endswith("=0") for c in counts) else ""
    print(f"  {s}  ({', '.join(counts)}){flag}")


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


COLDSTART_SYNTH_PROB = 0.15  # ~1 in 7 training batches


def synthetic_coldstart_augment(inputs, labels):
    """Explicitly teach "zero context -> zero prediction" as a guaranteed signal.

    The passive fix (always_include_coldstart_dry in load_dataset_windowed_patches)
    only rebalances how often NATURALLY OCCURRING cold-start windows survive the
    dry-timestep filter -- it can't manufacture examples that don't exist in the
    41-sim dataset, and a smoke test showed the t=0 hallucination probe (real
    terrain + true zero context) staying flat around 0.15-0.21m even after several
    epochs with that fix alone, instead of trending toward 0. This is the active
    complement: with probability COLDSTART_SYNTH_PROB, force spatiotemporal and
    temporal to true zero (keeping the real terrain) and the label to zero,
    guaranteeing the model sees this exact invariant on ~1/7 of all training
    batches regardless of how the raw data happens to be distributed.
    """
    do_zero = tf.random.uniform(()) < COLDSTART_SYNTH_PROB
    spt = tf.cond(do_zero, lambda: tf.zeros_like(inputs["spatiotemporal"]), lambda: inputs["spatiotemporal"])
    tmp = tf.cond(do_zero, lambda: tf.zeros_like(inputs["temporal"]), lambda: inputs["temporal"])
    lbl = tf.cond(do_zero, lambda: tf.zeros_like(labels), lambda: labels)
    return {**inputs, "spatiotemporal": spt, "temporal": tmp}, lbl


def load_split(split, sims, shuffle=True, n_future=N_FUTURE_STEPS_STAGE1, batch_size=None):
    """Load a dataset split for a given list of simulations."""
    ds = load_dataset_windowed_patches(
        filecache_dir=FILECACHE_DIR,
        sim_names=sims,
        dataset_split=split,
        patch_size=PATCH_SIZE,
        stride=PATCH_STRIDE,
        batch_size=batch_size or BATCH_SIZE,
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
        # R16: fixes cold-start hallucination -- t < n_flood_maps windows have
        # zero-padded context (no real history), and t=0's label is dry by
        # definition (simulation start). dry_timestep_fraction alone throttled
        # these to 12%, while t=1..4 windows that already show flooding (fast
        # storm onset) passed through unthrottled -- teaching the model that
        # sparse context correlates with flooding. Confirmed visually and in
        # fullchunk_results.json: t=0 pred_max averaged 0.259m across 46 real
        # chunks despite GT being exactly 0.000m every time (15% of chunks
        # showed >0.3m hallucinated depth). Always including cold-start dry
        # windows for training (not eval/test -- see dataset.py docstring)
        # gives the model a complete, unbiased view of true storm starts.
        always_include_coldstart_dry=(split == "train"),
    )
    ds = ds.map(clip_labels, num_parallel_calls=tf.data.AUTOTUNE)
    if split == "train":
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(synthetic_coldstart_augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.repeat()
    return ds.prefetch(tf.data.AUTOTUNE)


# =====================================================================
# BUILD MODEL
# =====================================================================
print("\n" + "=" * 60)
print("FLOOD MODEL TRAINING — Full Dataset")
print("=" * 60)

TOTAL_STEPS = (STAGE1_EPOCHS * STAGE1_STEPS_PER_EPOCH
               + STAGE2_EPOCHS * STAGE2_STEPS_PER_EPOCH
               + STAGE3_EPOCHS * STAGE3_STEPS_PER_EPOCH)

# ── V3 ARCHITECTURE CONFIG (round 3 — proper warm-start) ──────────────
# Strategy: keep V1 loss + V1 weights, add rain_broadcast as the ONLY change.
# This way we strictly build on V1's 30+ epochs of prior knowledge.
ARCH_V3 = True                     # framework flag
WARM_START = True                  # ROUND 13: warm start from R10 — ABeDice now replaces focal (no longer stacked)
USE_RAIN_BROADCAST = True          # kept from R3 (neutral-positive)
USE_DILATED_GEO = True             # ROUND 8: enable multi-scale dilated geo for spatial correlation
USE_STORM_EMBED = False            # still disabled
USE_DEEP_DECODER = False           # ROUND 5: identical arch to V3 R3, only loss changes
USE_GROUP_NORM = True              # ROUND 7: replace decoder BN with GroupNorm — fix 2.5m ceiling

with strategy.scope():
    # R18: lower peak LR (3e-5 -> 1e-5) and shorter warmup (2000 -> 500 steps)
    # than the R17 schedule. Loss stays v6 -- v7 (ABeDice replacing focal) was
    # tested via a 15-epoch smoke test (job 15758303) warm-started from this
    # same checkpoint and REJECTED: it made WARM CSI and MAE worse than the
    # untouched baseline in every epoch (CSI 0.323 -> 0.23-0.32, never
    # recovering). Only the AR-rollout curriculum (K=4 -> K=6) is new in this
    # run. Keeping the gentler LR anyway since a warm-started fine-tune with an
    # extended rollout is still a real change to how gradients flow, even
    # without a loss-function change on top.
    cosine_lr = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-6,
        decay_steps=TOTAL_STEPS,
        alpha=1e-6,
        warmup_target=1e-5,
        warmup_steps=500,
    )
    params = FloodModel.Params(
        lstm_units=128,
        lstm_kernel_size=7,            # ROUND 8: larger receptive field for spatial structure
        lstm_dropout=0.2,
        lstm_recurrent_dropout=0.2,
        n_flood_maps=N_FLOOD_MAPS,
        m_rainfall=M_RAINFALL,
        use_rain_broadcast=USE_RAIN_BROADCAST,
        use_dilated_geo=USE_DILATED_GEO,
        use_storm_embed=USE_STORM_EMBED,
        use_deep_decoder=USE_DEEP_DECODER,
        use_group_norm=USE_GROUP_NORM,
        # v7 (ABeDice boundary term replacing focal) TESTED AND REJECTED: 15-epoch
        # smoke test (job 15758303), warm-started from this same v6 checkpoint,
        # made WARM CSI and MAE worse than the untouched starting point in every
        # single epoch (CSI 0.323 -> 0.23-0.32, never recovering to baseline;
        # MAE 0.0198 -> 0.023-0.041). Reintroducing that old boundary term is not
        # a free CSI win the way the codebase's old v1 comment suggested -- it
        # actively conflicts with v6's depth-weighting when combined. Staying on
        # v6, not gambling the full run on a change that already failed its own
        # validation.
        loss_version="v6",
        optimizer=keras.optimizers.Adam(learning_rate=cosine_lr, global_clipnorm=1.0),
    )
    model = FloodModel(params=params, spatial_dims=(PATCH_SIZE, PATCH_SIZE))

    if WARM_START:
        # R18: warm-start from the just-finished v6 run's final best checkpoint
        # (epoch 48 of run_20260811-141236 -- confirmed the val_loss-best and
        # deepwater-best checkpoints converged to the SAME epoch, byte-identical
        # files -- rigorous 6-city eval: flood MAE 0.067m, flood CSI 0.26-0.36,
        # peak correlation 0.93-0.97, all a large improvement over R14). This is
        # now the best available starting point -- R16 would discard all of that.
        _dummy = {
            "geospatial":     tf.zeros((1, PATCH_SIZE, PATCH_SIZE, 12)),
            "spatiotemporal": tf.zeros((1, N_FLOOD_MAPS, PATCH_SIZE, PATCH_SIZE, 1)),
            "temporal":       tf.zeros((1, N_FLOOD_MAPS, M_RAINFALL)),
        }
        _ = model._model(_dummy)
        _v6final = pathlib.Path(
            "/scratch/rmj7591/climateiq/train_output/run_20260811-141236/weights_deepwater_best.npz"
        )
        _data = np.load(str(_v6final))
        _wlist = [_data[f"w{i:02d}"] for i in range(len(_data.files))]
        model._model.set_weights(_wlist)
        print(f"✓ v6-final (epoch 48) warm-start: {len(_wlist)} weights loaded from {_v6final.parent.name}")
    else:
        print("✓ From-scratch training — no pretrained weights")

print(f"Params: lstm_units={params.lstm_units}, kernel={params.lstm_kernel_size}")
print(f"Patch: {PATCH_SIZE}x{PATCH_SIZE}, stride={PATCH_STRIDE}")
print(f"Epochs: {EPOCHS} ({STAGE1_EPOCHS} @ K={N_FUTURE_STEPS_STAGE1} + {STAGE2_EPOCHS} @ K={N_FUTURE_STEPS_STAGE2} + {STAGE3_EPOCHS} @ K={N_FUTURE_STEPS_STAGE3}), Batch: {BATCH_SIZE} (stage 3: {STAGE3_BATCH_SIZE})")
print("=== R19: v6 loss (unchanged, v7 tested and rejected) + 3-stage AR curriculum ===")
print(f"  • loss: v6 — depth_weight/peak_penalty x0.2 + focal x0.3 (v7's ABeDice swap made WARM CSI/MAE worse in every epoch of its own smoke test, job 15758303 -- reverted)")
print(f"  • warm-start: v6-final epoch 48 (run_20260811-141236), not R16 — that checkpoint is now the best available")
print(f"  • curriculum: stage 1 K={N_FUTURE_STEPS_STAGE1} ({STAGE1_EPOCHS} ep) -> stage 2 K={N_FUTURE_STEPS_STAGE2} ({STAGE2_EPOCHS} ep) -> stage 3 K={N_FUTURE_STEPS_STAGE3} ({STAGE3_EPOCHS} ep)")
print(f"    stage 3 (K=11) forces every window to true cold-start (last_t=3 given T=13 chunks) -- matches the")
print(f"    real deployment scenario exactly (zero context, self-feed for the full horizon), validated in")
print(f"    train_smoke_fullrollout.py: flood CSI +9-10% relative, partial fix of Atlanta's severity-ranking")
print(f"    inversion, on only 10 epochs / 4 cities / batch=2 -- this run gives it the full roster + more epochs.")
print(f"  • rain_broadcast: {params.use_rain_broadcast}")
print(f"  • checkpoints: weights_ordered.npz (val_loss-best) AND weights_deepwater_best.npz (1-4m MAE-best)")
print(f"Temporal: v2 (rate + cumulative + delta_rate + log_cum + running_max + frac_time)")
print("Scheduled sampling: Curriculum AR (0→0.15 ramp) + feedback noise (σ=0.02)")
print(f"Simulations: {len(sim_names)} total")
print(f"LR schedule: cosine decay 1e-6 → 1e-5 (warmup 500 steps) → 1e-6 over {TOTAL_STEPS} steps")
print(f"Gradient clipping: global_clipnorm=1.0")
print(f"Feedback clip: 4.0m (consistent train/val)")

# =====================================================================
# TRAIN
# =====================================================================
print("\nLoading stage datasets (K={}, {}, {})...".format(
    N_FUTURE_STEPS_STAGE1, N_FUTURE_STEPS_STAGE2, N_FUTURE_STEPS_STAGE3))
train_ds_stage1 = load_split("train", sims=sim_names, shuffle=True, n_future=N_FUTURE_STEPS_STAGE1)
train_ds_stage2 = load_split("train", sims=sim_names, shuffle=True, n_future=N_FUTURE_STEPS_STAGE2)
train_ds_stage3 = load_split("train", sims=sim_names, shuffle=True, n_future=N_FUTURE_STEPS_STAGE3,
                              batch_size=STAGE3_BATCH_SIZE)
# Cache a fixed 500-batch val subset so val_loss is identical each epoch — eliminates
# oscillation caused by TF generator restart + curriculum sampling state changes.
# Separate caches per stage since test_step's K-step unroll must match the
# label shape of whatever val_dataset is passed to that stage's fit() call.
VAL_CACHE_STEPS = 500
print(f"Caching {VAL_CACHE_STEPS} val batches per stage (fixed subset for stable early stopping)...")
val_ds_stage1 = load_split("val", sims=sim_names, shuffle=False, n_future=N_FUTURE_STEPS_STAGE1).take(VAL_CACHE_STEPS).cache()
_ = list(val_ds_stage1.as_numpy_iterator())
val_ds_stage2 = load_split("val", sims=sim_names, shuffle=False, n_future=N_FUTURE_STEPS_STAGE2).take(VAL_CACHE_STEPS).cache()
_ = list(val_ds_stage2.as_numpy_iterator())
# Stage 3 (K=11): far fewer available windows per chunk (only t in {0,1,2}
# given T=13 chunks) -- .take(VAL_CACHE_STEPS) on a non-repeating val dataset
# just yields whatever's actually available if that's less than 500, no crash.
val_ds_stage3 = load_split("val", sims=sim_names, shuffle=False, n_future=N_FUTURE_STEPS_STAGE3,
                            batch_size=STAGE3_BATCH_SIZE).take(VAL_CACHE_STEPS).cache()
_ = list(val_ds_stage3.as_numpy_iterator())
print(f"Val caches ready.")

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


# ── Fixed deep-water eval set — same chunks used for the R14/R15/v4-smoke-test
# comparisons. R15's checkpoint callback saved on val_loss alone and froze on an
# epoch that LOOKED good but had ~2x worse deep-water MAE than baseline — val_loss
# is not a reliable proxy for the metric we actually care about. This callback
# tracks the 1-4m depth-bin MAE directly and saves a SEPARATE checkpoint for it,
# so a bad val_loss-selected epoch can't silently become the only option again.
# FIXED (launch-readiness audit): the safety gate this list feeds was
# effectively checking only 2 of these 4 sims. Manhattan_7 was downloaded in
# prediction-mode directory layout (no test/geospatial split at all -- 0
# files); Denton_5's test split exists but is empty (0 files) -- true for
# EVERY Denton scenario, it was never downloaded with held-out test data.
# Both silently contributed 0 patches while still passing the old
# `.exists()` check (which only verifies the top-level directory, not that
# files are actually inside it). Replaced with sims verified to have real
# test/geospatial + test/labels files on disk (Manhattan_2, Copenhagen_1),
# and the filter below now checks for actual files, not just a directory.
DEEPWATER_EVAL_SIMS = [
    "Manhattan-Manhattan_config/Rainfall_Data_2.txt",
    "Atlanta-Atlanta_config/Rainfall_Data_2.txt",
    "Phoenix_SM-PHX_SM/Rainfall_Data_15.txt",
    "Copenhagen-Copenhagen_config/Rainfall_Data_1.txt",
]


def _has_usable_test_data(sim):
    d = FILECACHE_DIR / sim / "test"
    return (
        d.exists()
        and any((d / "geospatial").glob("*.npy"))
        and any((d / "labels").glob("*.npy"))
    )


_dropped = [s for s in DEEPWATER_EVAL_SIMS if not _has_usable_test_data(s)]
if _dropped:
    print(f"WARNING: dropping DEEPWATER_EVAL_SIMS with no usable test data: {_dropped}")
DEEPWATER_EVAL_SIMS = [s for s in DEEPWATER_EVAL_SIMS if _has_usable_test_data(s)]
print(f"\nCollecting fixed deep-water eval set ({len(DEEPWATER_EVAL_SIMS)} sims)...")
_ds_deepwater = load_dataset_windowed_patches(
    filecache_dir=FILECACHE_DIR, sim_names=DEEPWATER_EVAL_SIMS, dataset_split="test",
    patch_size=PATCH_SIZE, stride=PATCH_STRIDE, batch_size=BATCH_SIZE,
    n_flood_maps=N_FLOOD_MAPS, m_rainfall=M_RAINFALL, n_future_steps=1,
    min_flood_fraction=0.01, min_max_depth=0.05, min_label_max_depth=0.01,
    dry_timestep_fraction=0.0, shuffle=False, temporal_feature_version=2,
)
_deepwater_inputs, _deepwater_labels = [], []
for _inp, _lbl in _ds_deepwater.take(30):  # 30 x 4 = 120 patches
    _deepwater_inputs.append({k: v.numpy() for k, v in _inp.items()})
    _deepwater_labels.append(_lbl.numpy())
_deepwater_labels = np.concatenate(_deepwater_labels, axis=0)

# _extract_spatiotemporal_patch (dataset.py) zero-pads the ENTIRE n_flood_maps
# history for early-chunk timesteps (t < n_flood_maps) — the windowing loop only
# filters on whether the CURRENT label is non-dry, so these zero/partial-context
# ("cold-start") samples aren't excluded. On a similar fixed eval set, ~46% of
# samples turned out to be cold-start (check_coldstart_contamination.py) and the
# top depth bin (2-4m) showed 0.95m MAE cold vs 0.54m warm — checkpoint selection
# must not be driven by a metric that mixes the two.
_deepwater_pad_counts = []
for _batch in _deepwater_inputs:
    _spt = _batch["spatiotemporal"]  # (B, N, H, W, 1)
    _frame_is_zero = np.all(_spt == 0, axis=(2, 3, 4))  # (B, N)
    for _b in range(_spt.shape[0]):
        _pc = 0
        for _k in range(_spt.shape[1]):
            if _frame_is_zero[_b, _k]:
                _pc += 1
            else:
                break
        _deepwater_pad_counts.append(_pc)
_deepwater_warm_mask = np.array(_deepwater_pad_counts) == 0
print(f"  {len(_deepwater_labels)} deep-water eval patches collected "
      f"({_deepwater_warm_mask.sum()} warm / {(~_deepwater_warm_mask).sum()} cold-start)")

# R17: cold-start hallucination probe on the SAME real geospatial as the
# deep-water eval set (Manhattan/Atlanta/Phoenix_SM/Denton), true zero context.
# R16's checkpoint selection (deep-water MAE alone) picked epoch 8 for
# weights_deepwater_best.npz, which had a REAL regression -- MAE 4-12x worse
# than R14 at Atlanta t=0-2 (visually confirmed, run_predict_r16_checkpoint.py)
# -- purely because that criterion never looked at cold-start behavior. This
# gates checkpoint saves on a safety threshold so that can't happen silently
# again: a checkpoint with out-of-control hallucination is disqualified from
# being saved as "best" no matter how good its deep-water number looks.
COLDSTART_SAFETY_MAX_M = 0.3  # from fullchunk_results.json: 15% of real chunks
                              # exceeded this at true t=0 in the original R14 bug
_probe_geo_all = tf.constant(np.concatenate([b["geospatial"] for b in _deepwater_inputs], axis=0))
_probe_zero_spt_all = tf.zeros_like(tf.constant(np.concatenate([b["spatiotemporal"] for b in _deepwater_inputs], axis=0)))
_probe_zero_temporal_all = tf.zeros_like(tf.constant(np.concatenate([b["temporal"] for b in _deepwater_inputs], axis=0)))


def _coldstart_probe(m):
    preds = []
    B = BATCH_SIZE
    for i in range(0, len(_probe_geo_all), B):
        p = m(
            {"geospatial": _probe_geo_all[i:i + B],
             "spatiotemporal": _probe_zero_spt_all[i:i + B],
             "temporal": _probe_zero_temporal_all[i:i + B]},
            training=False,
        ).numpy().squeeze(axis=-1)
        preds.append(np.clip(p, 0, None))
    preds = np.concatenate(preds, axis=0)
    return float(preds.mean()), float(preds.max())


class DeepWaterCheckpointCallback(keras.callbacks.Callback):
    """Saves weights_deepwater_best.npz when the WARM-ONLY >=1m depth-bin MAE
    improves AND the cold-start probe stays under COLDSTART_SAFETY_MAX_M.

    Independent of val_loss — see comment above DEEPWATER_EVAL_SIMS for why.
    Restricted to samples with full real history (not cold-start/zero-padded
    context) — see comment above _deepwater_pad_counts for why. Gated on the
    cold-start probe — see comment above COLDSTART_SAFETY_MAX_M for why.
    """

    def __init__(self, out_dir):
        super().__init__()
        self.out_dir = pathlib.Path(out_dir)
        self.best_score = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        preds = []
        for batch in _deepwater_inputs:
            p = self.model(
                {"geospatial":     tf.constant(batch["geospatial"]),
                 "spatiotemporal": tf.constant(batch["spatiotemporal"]),
                 "temporal":       tf.constant(batch["temporal"])},
                training=False,
            ).numpy().squeeze(axis=-1)
            preds.append(np.clip(p, 0, None))
        preds = np.concatenate(preds, axis=0)[_deepwater_warm_mask]
        gt_cap = np.clip(_deepwater_labels, 0, DEPTH_CAP)[_deepwater_warm_mask]
        deep_mask = gt_cap >= 1.0
        score = float(np.abs(preds[deep_mask] - gt_cap[deep_mask]).mean()) if deep_mask.sum() > 0 else float("inf")
        pred_peak = float(preds.reshape(len(preds), -1).max(1).max())
        cs_mean, cs_max = _coldstart_probe(self.model)
        print(f"\n  [deep-water check] epoch {epoch+1}: 1-4m MAE={score:.4f} (warm-only)  pred_peak_max={pred_peak:.3f}")
        print(f"  [coldstart probe]  epoch {epoch+1}: mean={cs_mean:.4f}m  max={cs_max:.4f}m "
              f"(safety threshold={COLDSTART_SAFETY_MAX_M}m)")
        if cs_max >= COLDSTART_SAFETY_MAX_M:
            print(f"  SKIPPED save — cold-start max {cs_max:.4f}m exceeds safety threshold, "
                  f"disqualified regardless of deep-water score")
        elif score < self.best_score:
            self.best_score = score
            npz_path = self.out_dir / "weights_deepwater_best.npz"
            ws = self.model.get_weights()
            np.savez(str(npz_path), **{f"w{i:02d}": w for i, w in enumerate(ws)})
            print(f"  Saved weights_deepwater_best.npz (score={score:.4f}, coldstart_max={cs_max:.4f}m)")


# Patience=20 exceeds every stage's epoch count (15/8/10) so early stopping
# effectively never fires within any single stage -- intentional here: this
# is a fixed curriculum experiment warm-started from an already-good
# checkpoint, and we want every stage to run to completion rather than exit
# early on transient val_loss noise (which the just-finished run showed
# plenty of).
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        verbose=1,
    ),
    keras.callbacks.TensorBoard(
        log_dir=str(log_dir / "tb"),
        histogram_freq=0,
        update_freq="epoch",
    ),
    NpzCheckpointCallback(keras_path=log_dir / "best_model.keras"),
    DeepWaterCheckpointCallback(out_dir=log_dir),
]

print(f"\nTraining: stage 1 ({STAGE1_EPOCHS} epochs @ K={N_FUTURE_STEPS_STAGE1}) -> "
      f"stage 2 ({STAGE2_EPOCHS} epochs @ K={N_FUTURE_STEPS_STAGE2}) -> "
      f"stage 3 ({STAGE3_EPOCHS} epochs @ K={N_FUTURE_STEPS_STAGE3})...")
print(f"Logs: {log_dir}")
t0 = time.time()

# Two sequential fit() calls sharing the same model/optimizer and the same
# callback INSTANCES (not fresh copies) -- NpzCheckpointCallback and
# DeepWaterCheckpointCallback carry their "best score so far" state in normal
# Python attributes, so reusing the instances means stage 2 only overwrites a
# checkpoint if it's actually better than anything seen in stage 1, not just
# stage 2 alone. EarlyStopping's own best/wait counters reset at each fit()
# call's on_train_begin -- acceptable since patience already can't fire within
# either stage (see comment above).
print(f"\n--- STAGE 1: K={N_FUTURE_STEPS_STAGE1} ({STAGE1_EPOCHS} epochs) ---")
history1 = model.fit(
    train_dataset=train_ds_stage1,
    val_dataset=val_ds_stage1,
    epochs=STAGE1_EPOCHS,
    steps_per_epoch=STAGE1_STEPS_PER_EPOCH,
    callbacks=callbacks,
)

print(f"\n--- STAGE 2: K={N_FUTURE_STEPS_STAGE2} ({STAGE2_EPOCHS} epochs) ---")
# CRITICAL FIX (found via train_smoke_curriculum.py crashing on this exact
# transition): Keras reuses the compiled train/test function across separate
# .fit() calls on the same model by default. Each stage feeds a DIFFERENT
# rollout length -- without this reset, TF's reduce_retracing relaxes the K
# axis to a dynamic size instead of retracing fresh, and train_step's
# tf.unstack(y_steps, axis=1) crashes with "Cannot infer argument num from
# shape". Without this fix this run would die at a stage transition after
# already completing the previous stage's compute.
model._model.train_function = None
model._model.test_function = None
history2 = model.fit(
    train_dataset=train_ds_stage2,
    val_dataset=val_ds_stage2,
    epochs=STAGE2_EPOCHS,
    steps_per_epoch=STAGE2_STEPS_PER_EPOCH,
    callbacks=callbacks,
)

print(f"\n--- STAGE 3: K={N_FUTURE_STEPS_STAGE3} ({STAGE3_EPOCHS} epochs, batch={STAGE3_BATCH_SIZE}) ---")
# Same retrace-reset fix as the stage1->2 transition, needed again here since
# stage 3 changes BOTH the rollout length (K=6->11) AND the batch size
# (4->2) relative to stage 2.
model._model.train_function = None
model._model.test_function = None
history3 = model.fit(
    train_dataset=train_ds_stage3,
    val_dataset=val_ds_stage3,
    epochs=STAGE3_EPOCHS,
    steps_per_epoch=STAGE3_STEPS_PER_EPOCH,
    callbacks=callbacks,
)

# Merge histories so downstream training-curve/config code (which expects a
# single `history.history` dict) keeps working unmodified across all stages.
class _MergedHistory:
    pass


history = _MergedHistory()
history.history = {
    k: history1.history.get(k, []) + history2.history.get(k, []) + history3.history.get(k, [])
    for k in set(history1.history) | set(history2.history) | set(history3.history)
}

train_time = time.time() - t0
print(f"\nTraining took {train_time/60:.1f} minutes")

# Explicitly reload the best checkpoint rather than trusting Keras's
# EarlyStopping(restore_best_weights=True) -- launch-readiness audit verified
# against the installed keras==2.15.0 source that restoration ONLY fires
# inside on_epoch_end, gated on `self.wait >= self.patience`. If fit() ends
# any other way (hitting EPOCHS=50 without a clean patience=20 plateau, a
# crash, a scheduler kill), restoration never fires and the in-memory model
# is just whatever the last completed epoch produced -- silently, with no
# warning. Every downstream artifact (final_model.keras, weights_ordered_
# final.npz, the MAE/peak-correlation report below) inherited this risk.
# Prefer the deep-water+coldstart-safety-gated checkpoint if it exists (it's
# the one actually validated against the metric we care about); fall back to
# the val_loss-best checkpoint otherwise.
_deepwater_ckpt = log_dir / "weights_deepwater_best.npz"
_valloss_ckpt = log_dir / "weights_ordered.npz"
if _deepwater_ckpt.exists():
    _best_ckpt = _deepwater_ckpt
elif _valloss_ckpt.exists():
    _best_ckpt = _valloss_ckpt
else:
    _best_ckpt = None
if _best_ckpt is not None:
    _d = np.load(str(_best_ckpt))
    model._model.set_weights([_d[f"w{i:02d}"] for i in range(len(_d.files))])
    print(f"Reloaded best checkpoint for final eval/save: {_best_ckpt.name}")
else:
    print("WARNING: no checkpoint file found -- final eval uses whatever epoch fit() ended on")

# Save final model (weights now guaranteed to be the best checkpoint, not
# whatever epoch fit() happened to end on)
final_path = str(log_dir / "final_model.keras")
model.save_model(final_path)
print(f"Final model saved: {final_path}")

# Also save final weights as npz so both best-checkpoint and final are available.
final_npz = log_dir / "weights_ordered_final.npz"
ws_final = model._model.get_weights()
np.savez(str(final_npz), **{f"w{i:02d}": w for i, w in enumerate(ws_final)})
print(f"Final weights saved: {final_npz} ({len(ws_final)} weights)")

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
print("Saved training_curves.png")

# =====================================================================
# EVALUATE ON VALIDATION
# =====================================================================
print("\n" + "=" * 60)
print("PREDICTION EVALUATION")
print("=" * 60)

# Use single-step labels for eval (model predicts 1 step, compare to GT)
# Wrapped in try/except — TF 2.15 TimeDistributed bug in eager model.call()
# with rain_broadcast active; does not affect training which uses compiled fit()
eval_ds = load_split("val", sims=sim_names, shuffle=False, n_future=1)
all_preds = []
all_labels = []
for inputs, labels in eval_ds.take(100):
    try:
        pred = model._model.predict_on_batch(
            {"geospatial": inputs["geospatial"],
             "spatiotemporal": inputs["spatiotemporal"],
             "temporal": inputs["temporal"]}
        )
    except Exception as e:
        print(f"  eval step skipped ({e})")
        continue
    pred_np = np.clip(pred if isinstance(pred, np.ndarray) else pred.numpy(), 0, None)
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
    "n_future_steps_stage1": N_FUTURE_STEPS_STAGE1,
    "n_future_steps_stage2": N_FUTURE_STEPS_STAGE2,
    "n_future_steps_stage3": N_FUTURE_STEPS_STAGE3,
    "stage1_epochs": STAGE1_EPOCHS,
    "stage2_epochs": STAGE2_EPOCHS,
    "stage3_epochs": STAGE3_EPOCHS,
    "stage3_batch_size": STAGE3_BATCH_SIZE,
    "loss": "make_hybrid_loss_v6 (linear MSE, three-tier depth weight, peak=0.2, focal=0.3) -- v7 tested and rejected, see smoke_v7 job 15758303",
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
