"""Full-chunk (1000x1000) prediction — autoregressive rollout with strip plots.

Generates one wide strip figure per chunk showing AR predictions for t=0..10,
plus GT-context single-step at peak timestep for reference.

Usage:
    cd usl_models && python ../run_predict_fullchunk.py 2>&1 | tee ../predict_fullchunk_new.log
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
import matplotlib.cm as cm
import matplotlib.colors as mcolors

SEED = 42
keras.utils.set_random_seed(SEED)
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

print(f"GPUs: {tf.config.list_physical_devices('GPU')}")

from scipy.stats import pearsonr

from usl_models.flood_ml.dataset import compute_dem_sink_channel, compute_flow_features
from usl_models.flood_ml.model import FloodModel, SpatialAttention, FloodConvLSTM, GreenAmptGate
from usl_models.flood_ml import customloss

# =====================================================================
# CONFIGURATION
# =====================================================================
FILECACHE_DIR = pathlib.Path("/home/shared/climateiq/filecache")
OUTPUT_DIR = pathlib.Path("/home/rmj7591/climateiq-cnn/train_output/run_20260417-164506")
MODEL_PATH = OUTPUT_DIR / "best_model.keras"

PATCH_SIZE = 256
PATCH_STRIDE = 128
N_FLOOD_MAPS = 5
M_RAINFALL = 6
CHUNK_SIZE = 1000
MAX_DEPTH_CLAMP = 4.0  # Match training label cap (raised to 4m)
T_STRIP = 11           # Show t=0 through t=10 in the strip figure

# Set True only for models trained with GreenAmptGate in their train_step
# (i.e. run_20260224-233139 onward).  For earlier checkpoints that learned
# to predict GT depths directly (without infiltration correction in the loop),
# applying GreenAmptGate at inference would double-subtract and under-predict.
APPLY_GREEN_AMPT = True

# Temporal feature version — must match training config.
# v1: all M channels identical (old runs before run_20260303-214043)
# v2: [rate, cum_norm, rate^2, log_cum, running_max, frac_time] (new runs)
# BACKTRACK: set to 1 for older model checkpoints
TEMPORAL_FEATURE_VERSION = 2

# Chunks to evaluate: (city_label, sim_name, split)
EVAL_CHUNKS = [
    # ── Test split (unseen patches, seen sims) ──────────────────────────
    ("Manhattan_test", "Manhattan-Manhattan_config/Rainfall_Data_7.txt", "test"),
    ("Manhattan_test", "Manhattan-Manhattan_config/Rainfall_Data_15.txt", "test"),
    ("Phoenix_SM_test", "Phoenix_SM-PHX_SM/Rainfall_Data_7.txt", "test"),
    # ── Unseen city (never trained on) ──────────────────────────────────
    ("Phoenix_PV_unseen", "Phoenix_PV-PHX_PV/Rainfall_Data_1.txt", "train"),
    ("Phoenix_PV_unseen", "Phoenix_PV-PHX_PV/Rainfall_Data_5.txt", "train"),
    # ── Training split (seen during training — checks if model fits) ────
    ("Atlanta_train", "Atlanta-Atlanta_config/Rainfall_Data_1.txt", "train"),
    ("Manhattan_train", "Manhattan-Manhattan_config/Rainfall_Data_1.txt", "train"),
    # ── New cities (fine-tuned on these) ──────────────────────────────
    ("Denton_TX", "Denton_TX-Denton_config/Rainfall_Data_1.txt", "train"),
    ("Lafayette_LA", "Lafayette_LA-Lafayette_config/Rainfall_Data_1.txt", "train"),
    ("Boise_ID", "Boise_ID-Boise_config/Rainfall_Data_1.txt", "train"),
    ("Navarre_FL", "Navarre_FL-Navarre_config/Rainfall_Data_1.txt", "train"),
]


# =====================================================================
# LOAD MODEL — rebuilt at CHUNK_SIZE x CHUNK_SIZE for direct inference.
#
# The model is fully convolutional (all Conv2D/ConvLSTM use padding="same")
# so it can run at any spatial size, not just the 256x256 it was trained on.
# We rebuild with spatial_dims=(CHUNK_SIZE, CHUNK_SIZE) and transfer weights.
# This eliminates patch stitching entirely and removes all grid artifacts.
# =====================================================================
WEIGHTS_PATH = OUTPUT_DIR / "weights_ordered.npz"
print(f"Loading model from: {MODEL_PATH}")
print(f"  Using weights: {WEIGHTS_PATH}")

params = FloodModel.Params(
    n_flood_maps=N_FLOOD_MAPS,
    m_rainfall=M_RAINFALL,
    lstm_kernel_size=5,
)

# Build at full chunk size — same weights, all spatial ops are padding="same"
loaded_model = FloodConvLSTM(params=params, spatial_dims=(CHUNK_SIZE, CHUNK_SIZE))

dummy_input = {
    "geospatial":     tf.zeros((1, CHUNK_SIZE, CHUNK_SIZE, 12)),
    "temporal":       tf.zeros((1, N_FLOOD_MAPS, M_RAINFALL)),
    "spatiotemporal": tf.zeros((1, N_FLOOD_MAPS, CHUNK_SIZE, CHUNK_SIZE, 1)),
}
_ = loaded_model(dummy_input, training=False)

if WEIGHTS_PATH.exists():
    data = np.load(str(WEIGHTS_PATH))
    w_list = [data[f"w{i:02d}"] for i in range(len(data.files))]
    # Expand geo_cnn first Conv2D kernel (5,5,10,16) → (5,5,12,16) if needed.
    try:
        GEO_IDX = next(i for i, w in enumerate(w_list) if w.shape == (5, 5, 10, 16))
        w_list[GEO_IDX] = np.pad(w_list[GEO_IDX], [(0, 0), (0, 0), (0, 2), (0, 0)])
        print(f"  Expanded geo kernel 10→12 channels at index {GEO_IDX}")
    except StopIteration:
        pass  # Already 12-channel weights (trained with flow features)
    loaded_model.set_weights(w_list)
    print(f"  Loaded {len(w_list)} weights from npz. Model at {CHUNK_SIZE}x{CHUNK_SIZE}.")
else:
    # Fall back: load weights directly from .keras archive via h5py.
    # keras.models.load_model fails due to global layer-counter naming
    # collisions.  We bypass it by reading model.weights.h5 directly and
    # mapping each saved array to the model's get_weights() slot by explicit
    # path.  The mapping was derived by comparing model.get_weights() shapes
    # against the h5 leaf structure of a saved checkpoint.
    import zipfile, h5py, io
    print(f"  weights_ordered.npz not found — loading from h5 inside {MODEL_PATH}")

    # Explicit ordered path list matching model.get_weights() index 0..34
    _H5_PATHS = [
        # st_cnn_stage1 Conv2D
        "st_cnn_stage1/layers/time_distributed_1/layer/vars/0",   # (5,5,1,8)
        "st_cnn_stage1/layers/time_distributed_1/layer/vars/1",   # (8,)
        # st_cnn_stage2 Conv2D
        "st_cnn_stage2/layers/time_distributed_1/layer/vars/0",   # (5,5,8,16)
        "st_cnn_stage2/layers/time_distributed_1/layer/vars/1",   # (16,)
        # geo_cnn conv1
        "layers/sequential/layers/conv2d/vars/0",                 # (5,5,10,16)
        "layers/sequential/layers/conv2d/vars/1",                 # (16,)
        # geo_cnn conv2
        "layers/sequential/layers/conv2d_1/vars/0",               # (5,5,16,64)
        "layers/sequential/layers/conv2d_1/vars/1",               # (64,)
        # ConvLSTM1
        "layers/sequential_1/layers/conv_lstm2d/cell/vars/0",     # (5,5,86,512)
        "layers/sequential_1/layers/conv_lstm2d/cell/vars/1",     # (5,5,128,512)
        "layers/sequential_1/layers/conv_lstm2d/cell/vars/2",     # (512,)
        # BN inside conv_lstm
        "layers/sequential_1/layers/batch_normalization/vars/0",  # (128,) gamma
        "layers/sequential_1/layers/batch_normalization/vars/1",  # (128,) beta
        "layers/sequential_1/layers/batch_normalization/vars/2",  # (128,) moving_mean
        "layers/sequential_1/layers/batch_normalization/vars/3",  # (128,) moving_var
        # ConvLSTM2
        "layers/sequential_1/layers/conv_lstm2d_1/cell/vars/0",   # (5,5,128,512)
        "layers/sequential_1/layers/conv_lstm2d_1/cell/vars/1",   # (5,5,128,512)
        "layers/sequential_1/layers/conv_lstm2d_1/cell/vars/2",   # (512,)
        # spatial attention
        "layers/spatial_attention/conv/vars/0",                   # (7,7,2,1)
        "layers/spatial_attention/conv/vars/1",                   # (1,)
        # decoder_conv1
        "layers/conv2d/vars/0",                                   # (3,3,136,32)
        "layers/conv2d/vars/1",                                   # (32,)
        # decoder_bn1
        "layers/batch_normalization/vars/0",                      # (32,) gamma
        "layers/batch_normalization/vars/1",                      # (32,) beta
        "layers/batch_normalization/vars/2",                      # (32,) moving_mean
        "layers/batch_normalization/vars/3",                      # (32,) moving_var
        # decoder_conv2
        "layers/conv2d_1/vars/0",                                 # (3,3,32,16)
        "layers/conv2d_1/vars/1",                                 # (16,)
        # decoder_bn2
        "layers/batch_normalization_1/vars/0",                    # (16,) gamma
        "layers/batch_normalization_1/vars/1",                    # (16,) beta
        "layers/batch_normalization_1/vars/2",                    # (16,) moving_mean
        "layers/batch_normalization_1/vars/3",                    # (16,) moving_var
        # output_conv
        "output_conv/vars/0",                                     # (3,3,16,1)
        "output_conv/vars/1",                                     # (1,)
        # sampling_prob (non-trainable scalar)
        "vars/0",                                                  # ()
    ]

    with zipfile.ZipFile(str(MODEL_PATH), "r") as zf:
        h5_bytes = zf.read("model.weights.h5")
    with h5py.File(io.BytesIO(h5_bytes), "r") as hf:
        w_list = [np.array(hf[p]) for p in _H5_PATHS]

    loaded_model.set_weights(w_list)
    print(f"  Loaded {len(w_list)} weights from h5 (direct). Model at {CHUNK_SIZE}x{CHUNK_SIZE}.")
    # Also save as npz so future runs use the fast path
    npz_path = WEIGHTS_PATH
    np.savez(str(npz_path), **{f"w{i:02d}": w for i, w in enumerate(w_list)})
    print(f"  Saved weights_ordered.npz for future use.")


def predict_full_chunk(geo_tf, temporal_window, spatiotemporal, cumul_F):
    """Single forward pass on a full CHUNK_SIZE x CHUNK_SIZE spatial input.

    Green-Ampt infiltration correction is applied after the raw model output
    because train_step/test_step apply it during training — calling the model
    without it would be a training/inference mismatch.

    Args:
        geo_tf:          (1, H, W, 12) tf.Tensor (pre-converted, reused across steps)
        temporal_window: (N_FLOOD_MAPS, M_RAINFALL)
        spatiotemporal:  (N_FLOOD_MAPS, H, W, 1)
        cumul_F:         (1, H, W, 1) tf.Tensor, cumulative infiltration depth (m)

    Returns:
        pred_np:   (H, W) corrected flood depth, clipped to [0, MAX_DEPTH_CLAMP].
        cumul_F:   (1, H, W, 1) updated cumulative infiltration tf.Tensor.
    """
    inputs = {
        "geospatial":     geo_tf,
        "temporal":       tf.expand_dims(tf.convert_to_tensor(temporal_window, dtype=tf.float32), 0),
        "spatiotemporal": tf.expand_dims(tf.convert_to_tensor(spatiotemporal,  dtype=tf.float32), 0),
    }
    pred_raw = loaded_model(inputs, training=False)   # (1, H, W, 1)
    pred_relu = tf.nn.relu(pred_raw)
    if APPLY_GREEN_AMPT:
        pred_corrected, cumul_F = loaded_model.green_ampt_gate(pred_relu, geo_tf, cumul_F)
    else:
        pred_corrected = pred_relu
    pred_np = np.clip(pred_corrected.numpy()[0, :, :, 0], 0, MAX_DEPTH_CLAMP)
    return pred_np, cumul_F


def flood_corr(pred, gt, thresh=0.01):
    """Pearson corr on flooded pixels (GT > thresh). Returns nan if <50 px."""
    mask = gt.ravel() > thresh
    if mask.sum() < 50:
        return float("nan")
    return float(pearsonr(pred.ravel()[mask], gt.ravel()[mask])[0])


def _build_temporal_v2(temporal_vec):
    """Build full (T, 6) temporal feature matrix matching training v2."""
    v = temporal_vec.astype(np.float32)
    T = len(v)
    cumsum = np.cumsum(v)
    total = cumsum[-1] if cumsum[-1] > 0 else 1.0
    running_max = np.maximum.accumulate(v)
    delta_rate = np.concatenate([[0.0], np.diff(v)])  # storm phase transitions
    return np.stack([
        v,
        cumsum / total,
        delta_rate,           # replaces rate^2 — encodes storm onset/recession
        np.log1p(cumsum),
        running_max,
        np.arange(T, dtype=np.float32) / max(T - 1, 1),
    ], axis=1).astype(np.float32)  # (T, 6)


def prepare_temporal_window(temporal_vec, t, n, m):
    """Build (n, m) temporal window at timestep t (zero-padded for t < n).

    Uses TEMPORAL_FEATURE_VERSION to match training config:
      v1: all m channels = same scalar (old runs)
      v2: 6-channel features [rate, cum_norm, rate^2, log_cum, max, frac_time]
    """
    if TEMPORAL_FEATURE_VERSION == 2:
        temporal_2d = _build_temporal_v2(temporal_vec)
    else:
        temporal_2d = np.tile(temporal_vec[:, None], (1, m)).astype(np.float32)
    start = max(t - n, 0)
    pad_len = max(n - t, 0)
    data = temporal_2d[start:t]
    if pad_len > 0:
        zeros = np.zeros((pad_len, temporal_2d.shape[1]), dtype=np.float32)
        window = np.concatenate([zeros, data], axis=0)
    else:
        window = data
    return window


def save_2row_strip_figure(city_label, stem, sim_label, ar_preds, labels, T_strip,
                           title_tag="GT-context single-step", t_offset=0,
                           fname_tag="gtctx"):
    """Clean 2-row (GT | Prediction) comparison figure with correlation metrics.

    Uses 99th-percentile GT vmax per column so shallow street flooding is
    visible. Subplot titles include spatial correlation and MAE.
    """
    from matplotlib.gridspec import GridSpec
    from scipy.stats import pearsonr

    T_strip = min(T_strip, len(ar_preds), labels.shape[0])
    H, W = labels.shape[1], labels.shape[2]

    BORDER = 32
    flood_union = np.zeros((H, W), dtype=bool)
    for t in range(T_strip):
        flood_union |= (labels[t] > 0.01)

    if flood_union.any():
        rr = np.where(np.any(flood_union, axis=1))[0]
        cc = np.where(np.any(flood_union, axis=0))[0]
        pad = 20
        r0 = max(BORDER, int(rr[0])  - pad)
        r1 = min(H - BORDER, int(rr[-1]) + pad + 1)
        c0 = max(BORDER, int(cc[0])  - pad)
        c1 = min(W - BORDER, int(cc[-1]) + pad + 1)
    else:
        r0, r1 = BORDER, H - BORDER
        c0, c1 = BORDER, W - BORDER

    crop_h = r1 - r0
    crop_w = c1 - c0
    aspect = crop_h / max(crop_w, 1)

    # Per-column vmax: 99th percentile of GT at flooded pixels
    vmaxes = []
    for t in range(T_strip):
        gt_crop = labels[t][r0:r1, c0:c1]
        flooded = gt_crop > 0.01
        if flooded.sum() > 50:
            p99 = float(np.percentile(gt_crop[flooded], 99))
            vmaxes.append(max(min(p99, MAX_DEPTH_CLAMP), 0.05))
        else:
            vmaxes.append(0.05)

    panel_w = 3.2
    panel_h = min(max(panel_w * aspect, 2.0), 5.0)
    fig_w = T_strip * panel_w + 0.8
    fig_h = 2 * panel_h + 1.5

    col_widths = [1.0] * T_strip + [0.05]
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        2, T_strip + 1, figure=fig,
        width_ratios=col_widths,
        hspace=0.45, wspace=0.05,
        left=0.07, right=0.97, top=0.88, bottom=0.03,
    )

    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(T_strip)] for r in range(2)])
    cbar_ax = fig.add_subplot(gs[:, T_strip])

    t0, t1 = t_offset, t_offset + T_strip - 1
    fig.suptitle(
        f"{city_label}  |  chunk {stem} ({sim_label})\n"
        f"{title_tag}  t={t0}..{t1}  (5 min/step)",
        fontsize=11, fontweight="bold",
    )

    axes[0, 0].set_ylabel("Ground Truth", fontsize=9, rotation=90, labelpad=6)
    axes[1, 0].set_ylabel("Prediction", fontsize=9, rotation=90, labelpad=6)

    for ti in range(T_strip):
        gt_crop   = labels[ti][r0:r1, c0:c1]
        pred_crop = ar_preds[ti][r0:r1, c0:c1]
        vm        = vmaxes[ti]
        pred_disp = np.minimum(pred_crop, vm)

        mae_ti  = float(np.abs(ar_preds[ti] - labels[ti]).mean())
        flooded = gt_crop.ravel() > 0.01
        if flooded.sum() > 10:
            corr = float(pearsonr(gt_crop.ravel()[flooded], pred_crop.ravel()[flooded])[0])
        else:
            corr = float('nan')

        axes[0, ti].imshow(np.minimum(gt_crop, vm), cmap="Blues", vmin=0, vmax=vm, aspect="auto")
        axes[0, ti].set_title(
            f"t={ti + t_offset} ({(ti+t_offset)*5}min)\nGT p99={vm:.2f}m", fontsize=7
        )
        axes[0, ti].axis("off")

        axes[1, ti].imshow(pred_disp, cmap="Blues", vmin=0, vmax=vm, aspect="auto")
        axes[1, ti].set_title(f"corr={corr:.2f}  MAE={mae_ti:.4f}m", fontsize=7)
        axes[1, ti].axis("off")

    last_vm = vmaxes[-1]
    sm = cm.ScalarMappable(cmap="Blues", norm=mcolors.Normalize(0, last_vm))
    sm.set_array([])
    plt.colorbar(sm, cax=cbar_ax, label=f"Depth (m)")

    fname = f"{fname_tag}_strip2row_{city_label}_{stem}"
    plt.savefig(str(OUTPUT_DIR / f"{fname}.png"), dpi=150, bbox_inches="tight")
    plt.savefig(f"/home/rmj7591/climateiq-cnn/predictions/{fname}.png", dpi=150, bbox_inches="tight")
    print(f"  Saved 2-row: {fname}.png")
    plt.close()


def save_strip_figure(city_label, stem, sim_label, ar_preds, labels, T_strip,
                      title_tag="Autoregressive rollout", fname_tag="ar",
                      t_offset=0):
    """Save a 3-row x T_strip-col strip figure for autoregressive predictions.

    Two key design decisions:

    1. CROP TO FLOOD REGION
       Manhattan floods in thin street channels (~2-5 pixels wide in 1000x1000).
       Displaying the full 1000x1000 in a small subplot makes them sub-pixel and
       invisible. We find the union of all flooded pixels across the strip, then
       crop every panel to that bounding box (+50px padding).

    2. PER-COLUMN vmax
       A shared vmax=2.0m (peak) makes early timesteps (0.02-0.3m) look blank
       because their flood is at 1-15% of the color scale.  Each column gets its
       own GT-driven vmax so the spatial pattern is always fully visible.
       Raw Pred max and MAE are still shown in the subtitle for quantitative read.
    """
    from matplotlib.gridspec import GridSpec

    T_strip = min(T_strip, len(ar_preds), labels.shape[0])

    H, W = labels.shape[1], labels.shape[2]

    # ── 1. Find flood bounding box, excluding chunk-edge boundary artifacts ──
    # ConvLSTM boundary artifact: model trained on 256x256 patches uses
    # zero-padding at patch edges. At full-chunk inference (1000x1000) the same
    # zero-padding at the chunk boundary produces spurious high-value pixels
    # along the top/left/right corners. Diagnostic confirms max is at (0,0).
    # We exclude BORDER pixels from all edges to hide these artifacts and show
    # only the interior where predictions are physically meaningful.
    BORDER = 32
    flood_union = np.zeros((H, W), dtype=bool)
    for t in range(T_strip):
        flood_union |= (labels[t] > 0.01)

    if flood_union.any():
        rr = np.where(np.any(flood_union, axis=1))[0]
        cc = np.where(np.any(flood_union, axis=0))[0]
        pad = 20
        r0 = max(BORDER, int(rr[0])  - pad)
        r1 = min(H - BORDER, int(rr[-1]) + pad + 1)
        c0 = max(BORDER, int(cc[0])  - pad)
        c1 = min(W - BORDER, int(cc[-1]) + pad + 1)
    else:
        r0, r1 = BORDER, H - BORDER
        c0, c1 = BORDER, W - BORDER

    crop_h = r1 - r0
    crop_w = c1 - c0
    aspect = crop_h / max(crop_w, 1)

    # ── 2. Per-column depth scale — 99th percentile of GT at flooded pixels ───
    # Manhattan flood depths are extremely skewed: 9% of pixels flooded but
    # average depth only ~4 cm, with a single peak pixel at >1 m.  Using max
    # as vmax squishes the shallow street flooding to <3% of the colour range
    # (invisible), while using 99th pctile makes typical street depths fill the
    # full scale.  Pixels deeper than vmax simply saturate to the darkest blue.
    vmaxes = []
    for t in range(T_strip):
        gt_crop = labels[t][r0:r1, c0:c1]
        flooded = gt_crop > 0.01
        if flooded.sum() > 50:
            p99 = float(np.percentile(gt_crop[flooded], 99))
            vmaxes.append(max(min(p99, MAX_DEPTH_CLAMP), 0.05))
        else:
            vmaxes.append(0.05)

    # Shared error scale — use clipped-pred diffs so outliers don't dominate
    dmax = max(
        max(
            abs(float((np.minimum(ar_preds[t], vmaxes[t]) - labels[t])[r0:r1, c0:c1].min()))
            for t in range(T_strip)
        ),
        max(
            abs(float((np.minimum(ar_preds[t], vmaxes[t]) - labels[t])[r0:r1, c0:c1].max()))
            for t in range(T_strip)
        ),
        0.01,
    )

    # ── 3. Figure layout ─────────────────────────────────────────────────────
    panel_w = 3.5
    panel_h = min(max(panel_w * aspect, 2.0), 5.0)   # clamp panel height
    fig_w = T_strip * panel_w + 0.8
    fig_h = 3 * panel_h + 2.0    # 3 rows + title/label space

    col_widths = [1.0] * T_strip + [0.05]
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(
        3, T_strip + 1, figure=fig,
        width_ratios=col_widths,
        hspace=0.4, wspace=0.05,
        left=0.06, right=0.97, top=0.90, bottom=0.02,
    )

    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(T_strip)] for r in range(3)])
    # Two separate colorbar axes — one is dummy, each column has its own scale
    # We draw a representative colorbar for the last column's vmax
    cbar_ax_depth = fig.add_subplot(gs[0:2, T_strip])
    cbar_ax_diff  = fig.add_subplot(gs[2,   T_strip])

    t0, t1 = t_offset, t_offset + T_strip - 1
    fig.suptitle(
        f"{city_label}  |  chunk {stem} ({sim_label})\n"
        f"{title_tag}  t={t0}..{t1}  (5 min/step) — "
        f"cropped {crop_w}x{crop_h} px,  per-column depth scale",
        fontsize=11, fontweight="bold",
    )

    row_labels = ["Ground Truth", "AR Prediction", "Difference (Pred - GT)"]
    for row_i, lbl in enumerate(row_labels):
        axes[row_i, 0].set_ylabel(lbl, fontsize=9, rotation=90, labelpad=6)

    for ti in range(T_strip):
        gt_crop   = labels[ti][r0:r1, c0:c1]
        pred_crop = ar_preds[ti][r0:r1, c0:c1]
        vm        = vmaxes[ti]
        pred_disp = np.minimum(pred_crop, vm)
        diff_disp = pred_disp - gt_crop
        mae_ti    = float(np.abs(ar_preds[ti] - labels[ti]).mean())  # full-map MAE

        # Ground truth — each column on its own [0, vm] scale
        axes[0, ti].imshow(np.minimum(gt_crop, MAX_DEPTH_CLAMP),
                           cmap="Blues", vmin=0, vmax=vm, aspect="auto")
        axes[0, ti].set_title(f"t={ti + t_offset}\nGT={labels[ti].max():.2f}m", fontsize=7)
        axes[0, ti].axis("off")

        # Prediction — clipped at this column's vm
        axes[1, ti].imshow(pred_disp, cmap="Blues", vmin=0, vmax=vm, aspect="auto")
        axes[1, ti].set_title(f"Pred={ar_preds[ti].max():.2f}m\nMAE={mae_ti:.4f}m", fontsize=7)
        axes[1, ti].axis("off")

        # Difference
        axes[2, ti].imshow(diff_disp, cmap="RdBu_r", vmin=-dmax, vmax=dmax, aspect="auto")
        axes[2, ti].set_title(f"err={abs(diff_disp).max():.3f}m", fontsize=7)
        axes[2, ti].axis("off")

    # Representative colorbars (last column's vmax for depth, shared dmax for diff)
    last_vm = vmaxes[-1]
    sm_depth = cm.ScalarMappable(cmap="Blues",  norm=mcolors.Normalize(0,      last_vm))
    sm_diff  = cm.ScalarMappable(cmap="RdBu_r", norm=mcolors.Normalize(-dmax,  dmax))
    sm_depth.set_array([])
    sm_diff.set_array([])
    plt.colorbar(sm_depth, cax=cbar_ax_depth, label=f"Depth (m)\n[t={T_strip-1} scale]")
    plt.colorbar(sm_diff,  cax=cbar_ax_diff,  label="Error (m)")

    fname = f"strip_{fname_tag}_{city_label}_{stem}"
    plt.savefig(str(OUTPUT_DIR / f"{fname}.png"), dpi=150, bbox_inches="tight")
    plt.savefig(f"/home/rmj7591/climateiq-cnn/predictions/{fname}.png", dpi=150, bbox_inches="tight")
    print(f"  Saved strip: {fname}.png  crop=({r0}:{r1},{c0}:{c1})  vmaxes={[f'{v:.2f}' for v in vmaxes]}")
    plt.close()
    return vmaxes, dmax


# =====================================================================
# MAIN: PREDICT FULL CHUNKS
# =====================================================================
all_results = []

for city_label, sim_name, split in EVAL_CHUNKS:
    sim_dir = FILECACHE_DIR / sim_name
    if not sim_dir.exists():
        print(f"Skipping {sim_name} — not found")
        continue

    feat_dir = sim_dir / split / "geospatial"
    label_dir = sim_dir / split / "labels"
    temporal_path = sim_dir / "temporal.npy"

    if not feat_dir.exists() or not label_dir.exists() or not temporal_path.exists():
        print(f"Skipping {sim_name}/{split} — missing data")
        continue

    feat_files  = sorted(feat_dir.glob("*.npy"))
    label_files = {f.stem: f for f in label_dir.glob("*.npy")}
    temporal_vec = np.load(temporal_path)
    sim_label = sim_name.split("/")[0]

    print(f"\n{'=' * 60}")
    print(f"  {city_label}: {sim_name} ({split})")
    print(f"  Chunks: {len(feat_files)}")
    print(f"{'=' * 60}")

    for feat_file in feat_files[:1]:  # One chunk per sim
        stem = feat_file.stem
        if stem not in label_files:
            continue

        print(f"\n  Chunk: {stem}")

        geo_raw    = np.load(feat_file).astype(np.float32)     # (1000, 1000, 9)
        dem_sink   = compute_dem_sink_channel(geo_raw)          # (1000, 1000, 1)
        flow_feats = compute_flow_features(geo_raw, cell_size=2.0)  # (1000, 1000, 2)
        geospatial = np.concatenate([geo_raw, dem_sink, flow_feats], axis=-1)  # (1000, 1000, 12)
        label_arr  = np.load(label_files[stem])  # (1000, 1000, T)
        labels     = np.transpose(label_arr, (2, 0, 1))  # (T, H, W)
        T_max      = labels.shape[0]
        max_per_t  = labels.reshape(T_max, -1).max(axis=1)
        peak_t     = int(np.argmax(max_per_t))
        print(f"  T={T_max} timesteps, peak flood at t={peak_t} ({max_per_t[peak_t]:.3f}m)")

        # Pre-convert geospatial once (reused every timestep)
        geo_tf = tf.expand_dims(tf.convert_to_tensor(geospatial, dtype=tf.float32), 0)
        zero_F = tf.zeros((1, CHUNK_SIZE, CHUNK_SIZE, 1), dtype=tf.float32)

        # ── Run A: GT-context (teacher-forced single step at peak) ───────
        # Upper bound — model sees perfect prior context.
        # Reveals whether spatial pattern capacity is sufficient.
        print("  [A] GT-context single-step at peak timestep...")
        if peak_t >= N_FLOOD_MAPS and labels[peak_t].max() >= 0.01:
            st = labels[peak_t - N_FLOOD_MAPS : peak_t][..., np.newaxis]
            tw = prepare_temporal_window(temporal_vec, peak_t, N_FLOOD_MAPS, M_RAINFALL)
            pred_peak, _ = predict_full_chunk(geo_tf, tw, st, zero_F)
            mae_peak = float(np.abs(pred_peak - labels[peak_t]).mean())
            print(f"    peak t={peak_t}: GT max={labels[peak_t].max():.3f}m  "
                  f"Pred max={pred_peak.max():.3f}m  MAE={mae_peak:.4f}m")

            # Save single peak GT-context figure
            gt_pk = labels[peak_t]
            shared_vmax = min(MAX_DEPTH_CLAMP, max(float(gt_pk.max()), float(pred_peak.max()), 0.01))
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(
                f"{city_label} — {stem} ({sim_label})\n"
                f"GT-context single step  t={peak_t} (peak)",
                fontsize=13, fontweight="bold",
            )
            axes[0].imshow(np.minimum(gt_pk, MAX_DEPTH_CLAMP), cmap="Blues", vmin=0, vmax=shared_vmax)
            axes[0].set_title(f"Ground Truth\nmax={gt_pk.max():.3f}m", fontsize=11)
            axes[0].axis("off")
            plt.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(0, shared_vmax), cmap="Blues"),
                         ax=axes[0], fraction=0.046, pad=0.04, label="Depth (m)")

            axes[1].imshow(pred_peak, cmap="Blues", vmin=0, vmax=shared_vmax)
            axes[1].set_title(f"Prediction\nmax={pred_peak.max():.3f}m", fontsize=11)
            axes[1].axis("off")
            plt.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(0, shared_vmax), cmap="Blues"),
                         ax=axes[1], fraction=0.046, pad=0.04, label="Depth (m)")

            diff_pk = pred_peak - gt_pk
            dmax_pk = max(abs(diff_pk.min()), abs(diff_pk.max()), 0.01)
            axes[2].imshow(diff_pk, cmap="RdBu_r", vmin=-dmax_pk, vmax=dmax_pk)
            axes[2].set_title(f"Difference\nMAE={mae_peak:.4f}m", fontsize=11)
            axes[2].axis("off")
            plt.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(-dmax_pk, dmax_pk), cmap="RdBu_r"),
                         ax=axes[2], fraction=0.046, pad=0.04, label="Error (m)")

            plt.tight_layout()
            pk_fname = f"gtctx_peak_{city_label}_{stem}_t{peak_t}"
            plt.savefig(str(OUTPUT_DIR / f"{pk_fname}.png"), dpi=150)
            plt.savefig(f"/home/rmj7591/climateiq-cnn/predictions/{pk_fname}.png", dpi=150)
            print(f"  Saved GT-context peak: {pk_fname}.png")
            plt.close()

            all_results.append({
                "city": city_label, "sim": sim_label, "chunk": stem,
                "timestep": peak_t, "label": "gtctx_peak",
                "gt": gt_pk, "pred": pred_peak,
                "mae": mae_peak, "rmse": float(np.sqrt(np.square(pred_peak - gt_pk).mean())),
            })

        # ── Run B: GT-context single-step strip t=0..T_STRIP-1 ──────────
        # Teacher-forced: model sees actual GT prior flood maps at every step.
        # This is the correct evaluation for "can the model predict the next
        # flood map?" — it matches how the model was trained (patches always
        # had real flood context) and avoids the zero-start AR failure mode.
        #
        # WHY AR-FROM-ZEROS FAILS:
        # Training used flood-filtered patches (min_max_depth>0.1m), so the
        # model never saw all-zero prior context.  Feeding zeros → near-zero
        # prediction → zero feedback → the model never ignites flood prediction
        # in the interior (diagnostic showed 1376 nonzero vs 85K GT flooded px,
        # correlation=-0.003 at t=7).  The "low MAE" was fake — 90% dry pixels
        # are trivially correct while ALL flooded pixels are missed.
        t_strip = min(T_STRIP, T_max)
        print(f"  [B] GT-context single-step strip t=0..{t_strip - 1}...")
        gtctx_preds = []
        for t in range(t_strip):
            if t < N_FLOOD_MAPS:
                # Zero-pad context for early timesteps
                pad = np.zeros((N_FLOOD_MAPS - t, CHUNK_SIZE, CHUNK_SIZE, 1), np.float32)
                real = labels[:t][..., np.newaxis] if t > 0 else np.zeros((0,CHUNK_SIZE,CHUNK_SIZE,1),np.float32)
                st = np.concatenate([pad, real], axis=0)
            else:
                st = labels[t - N_FLOOD_MAPS : t][..., np.newaxis]
            tw = prepare_temporal_window(temporal_vec, t, N_FLOOD_MAPS, M_RAINFALL)
            pred_t, _ = predict_full_chunk(geo_tf, tw, st, zero_F)
            gtctx_preds.append(pred_t)
            gt_t  = labels[t]
            mae_t = float(np.abs(pred_t - gt_t).mean())
            corr_t = flood_corr(pred_t, gt_t)
            if t % 3 == 0 or t == t_strip - 1:
                print(f"    t={t}: GT={gt_t.max():.3f}m  Pred={pred_t.max():.3f}m  "
                      f"MAE={mae_t:.4f}m  corr={corr_t:.3f}")

        save_strip_figure(city_label, stem, sim_label, gtctx_preds, labels, t_strip,
                          title_tag="GT-context single-step", fname_tag="gtctx")
        save_2row_strip_figure(city_label, stem, sim_label, gtctx_preds, labels, t_strip,
                               title_tag="GT-context single-step", fname_tag="gtctx")

        # ── Run C: Warm-start AR — GT primes first N_FLOOD_MAPS steps ────
        # PRIMARY PRODUCTION MODE: GT provides initial flood state context
        # for the first N_FLOOD_MAPS timesteps, then model runs fully
        # autoregressive for the entire remaining simulation.
        #
        # This is physically correct: real deployments always have antecedent
        # conditions (spinup, sensor data, or early sim steps).  Zero-start
        # fails because the model never saw all-zero context during training.
        ws_start = N_FLOOD_MAPS   # first AR timestep (GT primes t=0..N_FLOOD_MAPS-1)
        ws_end   = T_max          # run to end of simulation
        n_ar     = ws_end - ws_start
        print(f"  [C] Warm-start AR t={ws_start}..{ws_end - 1} (GT primes t=0..{ws_start-1})...")

        # Initialise context from GT
        spatiotemporal = labels[0 : N_FLOOD_MAPS][..., np.newaxis].astype(np.float32)
        cumul_F = tf.zeros((1, CHUNK_SIZE, CHUNK_SIZE, 1), dtype=tf.float32)
        ws_preds   = []
        ws_labels  = []

        for t in range(ws_start, ws_end):
            tw = prepare_temporal_window(temporal_vec, t, N_FLOOD_MAPS, M_RAINFALL)
            pred_t, cumul_F = predict_full_chunk(geo_tf, tw, spatiotemporal, cumul_F)
            pred_t = np.clip(pred_t, 0, MAX_DEPTH_CLAMP)
            ws_preds.append(pred_t)
            ws_labels.append(labels[t])
            spatiotemporal = np.concatenate(
                [spatiotemporal[1:], pred_t[np.newaxis, :, :, np.newaxis]], axis=0
            )
            gt_t  = labels[t]
            mae_t = float(np.abs(pred_t - gt_t).mean())
            corr_t = flood_corr(pred_t, gt_t)
            if (t - ws_start) % 3 == 0 or t == ws_end - 1:
                print(f"    t={t}: GT={gt_t.max():.3f}m  Pred={pred_t.max():.3f}m  "
                      f"MAE={mae_t:.4f}m  corr={corr_t:.3f}")

        ws_labels_arr = np.array(ws_labels)

        # 3-row strip (GT | Pred | Diff) — first T_STRIP AR steps
        save_strip_figure(city_label, stem, sim_label, ws_preds,
                          ws_labels_arr, min(T_STRIP, n_ar),
                          title_tag=f"Warm-start AR (GT primes t<{ws_start})", fname_tag="ws_ar",
                          t_offset=ws_start)

        # 2-row strip (GT | Pred) — clean team-shareable figure
        save_2row_strip_figure(city_label, stem, sim_label, ws_preds,
                               ws_labels_arr, min(T_STRIP, n_ar),
                               title_tag=f"Warm-start AR (GT primes t<{ws_start})",
                               t_offset=ws_start, fname_tag="ws_ar")

        # Add warm-start AR results to summary
        for ti, t_abs in enumerate(range(ws_start, ws_end)):
            diff_ti = ws_preds[ti] - ws_labels[ti]
            all_results.append({
                "city": city_label, "sim": sim_label, "chunk": stem,
                "timestep": t_abs, "label": f"ws_ar_t{t_abs:02d}",
                "gt": ws_labels[ti], "pred": ws_preds[ti],
                "mae": float(np.abs(diff_ti).mean()),
                "rmse": float(np.sqrt(np.square(diff_ti).mean())),
            })

        # Summary results from GT-context strip
        for ti in range(t_strip):
            diff_ti = gtctx_preds[ti] - labels[ti]
            all_results.append({
                "city": city_label, "sim": sim_label, "chunk": stem,
                "timestep": ti, "label": f"gtctx_t{ti:02d}",
                "gt": labels[ti], "pred": gtctx_preds[ti],
                "mae": float(np.abs(diff_ti).mean()),
                "rmse": float(np.sqrt(np.square(diff_ti).mean())),
            })


# =====================================================================
# SUMMARY TABLE
# =====================================================================
print(f"\n{'=' * 75}")
print("FULL-CHUNK EVALUATION SUMMARY  (warm-start AR [primary] + GT-context [upper bound])")
print(f"{'=' * 75}")
print(f"{'City':<22} {'Chunk':<8} {'Label':<18} {'t':>4} "
      f"{'GT max':>8} {'Pred max':>9} {'MAE':>8} {'RMSE':>8}")
print("-" * 75)
for res in all_results:
    print(
        f"{res['city']:<22} {res['chunk']:<8} {res['label']:<18} {res['timestep']:>4} "
        f"{res['gt'].max():>8.3f} {res['pred'].max():>9.3f} "
        f"{res['mae']:>8.4f} {res['rmse']:>8.4f}"
    )
print(f"\nModel: {MODEL_PATH}")
