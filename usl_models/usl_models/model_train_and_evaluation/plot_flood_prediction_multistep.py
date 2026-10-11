"""Plot a 12-step autoregressive flood prediction vs. ground truth vs. diff.

For each city, gathers a few random test patches and rolls the model forward
n_future_steps autoregressively (mirroring FloodConvLSTM.test_step: each step's
prediction, after Green-Ampt correction, is fed back in as the newest
spatiotemporal frame). Each example is saved as its own figure with three
rows — ground truth, prediction, difference — and one column per step.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import math
import pathlib
import random
import numpy as np
import tensorflow as tf
import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import PowerNorm

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

from usl_models.flood_ml.model import FloodModel
from usl_models.flood_ml.model import SpatialAttention
from usl_models.flood_physical_ml.dataset import load_dataset_windowed_patches

FILECACHE_DIR = pathlib.Path("/scratch/hw4402/climateiq_filecache_us")

# Defaults; both can be overridden from the command line (see parse_args()).
DEFAULT_OUTPUT_DIR = pathlib.Path(
    "/scratch/hw4402/climateiq_output/semi_physical_test/examples_test"
)
DEFAULT_GT_MAX_DEPTH_RANGE = (0, 100)

OUTPUT_DIR = DEFAULT_OUTPUT_DIR

CHECKPOINT_PATH = pathlib.Path(
    "/scratch/hw4402/climateiq_output/semi_physical_test/weights_ordered.npz"
)

PATCH_SIZE   = 1000
N_FLOOD_MAPS = 5
M_RAINFALL   = 6
N_STEPS = 12            # autoregressive rollout length
EXAMPLES_PER_CITY = 10   # figures to generate per city
# RANDOM_SEED = 42        # fixed seed so example selection is reproducible

# Only plot examples whose ground-truth max flood depth (m), across all
# N_STEPS steps, falls in this range. Examples outside the range are skipped
# (not counted toward the quota).
GT_MAX_DEPTH_RANGE = DEFAULT_GT_MAX_DEPTH_RANGE

# Color display settings for the depth maps. Flood depth is heavily
# right-skewed (mostly shallow/dry pixels, a few deep ones), so a plain
# linear 0..max colormap washes out nearly all spatial detail. To fix this:
#  - DEPTH_COLOR_PERCENTILE clips the colorbar range to a percentile of the
#    data (instead of the true max), so a few extreme outlier pixels don't
#    stretch the whole scale (those pixels just saturate to the top color).
#  - DEPTH_GAMMA < 1 applies a power-law stretch that boosts contrast among
#    the many low/mid-depth pixels, at the cost of compressing the (already
#    saturated) high end.
# GT and prediction each get their own range (computed from their own data),
# since their value distributions can differ.
DEPTH_COLOR_PERCENTILE = 99.0
DEPTH_GAMMA = 0.4
DIFF_COLOR_PERCENTILE = 99.0


def robust_vmax(arr, percentile, min_vmax=1e-3):
    """Returns a percentile-based vmax, robust to a few extreme outlier pixels."""
    return max(float(np.percentile(arr, percentile)), min_vmax)

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

# Copenhagen = [
#     # Copenhagen
#     "Copenhagen-Copenhagen_config/Rainfall_Data_1.txt",
#     "Copenhagen-Copenhagen_config/Rainfall_Data_5.txt",
#     "Copenhagen-Copenhagen_config/Rainfall_Data_10.txt",
#     "Copenhagen-Copenhagen_config/Rainfall_Data_15.txt",
#     "Copenhagen-Copenhagen_config/Rainfall_Data_20.txt",
# ]

# CapeTown = [
#     # CapeTown
#     "CapeTown_Watershed_1-CapeTown_config/Rainfall_Data_1.txt",
#     "CapeTown_Watershed_1-CapeTown_config/Rainfall_Data_5.txt",
#     "CapeTown_Watershed_1-CapeTown_config/Rainfall_Data_10.txt",
#     "CapeTown_Watershed_1-CapeTown_config/Rainfall_Data_15.txt",
#     "CapeTown_Watershed_1-CapeTown_config/Rainfall_Data_20.txt",
# ]

TEST_SIMS = [Atlanta, Phoenix_SM, Manhattan, Phoenix_PV, Denton, Lafayettee, Boise, Navarre]
city_names = ["Atlanta", "Phoenix_SM", "Manhattan", "Phoenix_PV", "Denton", "Lafayettee", "Boise", "Navarre"]


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


# def load_weights(model, path):
#     custom_objects = {"SpatialAttention": SpatialAttention}
#     loaded_model = tf.keras.models.load_model(
#         path, custom_objects=custom_objects, compile=False
#     )
#     model._model.set_weights(loaded_model.get_weights())
def load_weights(model, path):
    data = np.load(str(path))
    model._model.set_weights([data[f"w{i:02d}"] for i in range(len(data.files))])
    print(f"  Loaded {len(data.files)} weights from {path.parent.name}")

def load_test_ds(sim_list, n_future_steps):
    return load_dataset_windowed_patches(
        filecache_dir=FILECACHE_DIR,
        sim_names=sim_list,
        dataset_split="test",
        patch_size=PATCH_SIZE,
        stride=PATCH_SIZE,
        batch_size=1,
        n_flood_maps=N_FLOOD_MAPS,
        m_rainfall=M_RAINFALL,
        n_future_steps=n_future_steps,
        min_flood_fraction=0.01,
        min_max_depth=0.05,
        min_label_max_depth=0.001,
        dry_timestep_fraction=0.5,
        shuffle=True,
        temporal_feature_version=2,
    )


def predict_n_steps(model, inputs, n_steps):
    """Autoregressive n-step rollout, mirroring FloodConvLSTM.test_step.

    Args:
        model: A FloodModel wrapper (uses model._model internally).
        inputs: Dict with "geospatial" [B,H,W,F], "spatiotemporal" [B,N,H,W,1]
            (initial context), and "temporal" [B,n_steps,N,M] (one rainfall
            window per future step, as returned when n_future_steps>1).
        n_steps: Number of autoregressive steps to roll out.

    Returns:
        [B, n_steps, H, W, 1] tensor of predicted flood depths.
    """
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
            training=False,
        )
        pred = tf.nn.relu(pred)
        # pred, cumul_F = keras_model.green_ampt_gate(pred, geospatial, cumul_F)
        preds.append(pred)

        # Feed the (clipped) corrected prediction back as the newest frame.
        # pred_fb = tf.minimum(pred, 4.0)
        # st = tf.concat([st[:, 1:, :, :, :], pred_fb[:, tf.newaxis, :, :, :]], axis=1)
        st = tf.concat([st[:, 1:, :, :, :], pred[:, tf.newaxis, :, :, :]], axis=1)

    return tf.stack(preds, axis=1)


def plot_example_block(
    img_axes, header_ax, cbar_ax_gt, cbar_ax_pred, cbar_ax_diff, pred, label, mae
):
    """Plots one example: a header line plus 3 rows (GT / pred / diff) x N_STEPS.

    GT and prediction share one color range per example (a robust,
    percentile-based max over both — see DEPTH_COLOR_PERCENTILE) and a
    power-law color normalization (see DEPTH_GAMMA) that boosts contrast
    among low/mid depth values, which would otherwise be washed out by a few
    deep outlier pixels under a linear scale. This shared range differs from
    example to example, since each example's own depth distribution differs.

    An extra column after the N_STEPS columns shows, per row, the pixelwise
    maximum over the rollout: max-over-time GT depth, max-over-time predicted
    depth, and the difference between those two max maps. It shares its row's
    color scale (and colorbar) with the per-step columns.

    Args:
        img_axes: [3, n_steps + 1] array of Axes to draw the maps onto (last
            column is the pixelwise max-over-time map).
        header_ax: Axes to draw the summary stats text onto (ticks/spines off).
        cbar_ax_gt: Axes to draw the ground-truth depth colorbar onto.
        cbar_ax_pred: Axes to draw the prediction depth colorbar onto.
        cbar_ax_diff: Axes to draw the difference colorbar onto.
        pred: [n_steps, H, W] predicted flood depth (m).
        label: [n_steps, H, W] ground-truth flood depth (m).
        mae: Mean absolute error for this example.
    """
    n_steps = pred.shape[0]
    max_col = n_steps

    depth_vmax = robust_vmax(np.concatenate([label, pred]), DEPTH_COLOR_PERCENTILE)
    gt_norm = PowerNorm(gamma=DEPTH_GAMMA, vmin=0, vmax=depth_vmax)
    pred_norm = gt_norm

    diff = pred - label
    diff_max = robust_vmax(np.abs(diff), DIFF_COLOR_PERCENTILE)

    gt_max_map = label.max(axis=0)
    pred_max_map = pred.max(axis=0)
    diff_max_map = pred_max_map - gt_max_map

    gt_im = pred_im = diff_im = None
    for t in range(n_steps):
        gt_im = img_axes[0, t].imshow(label[t], norm=gt_norm, cmap="Blues")
        img_axes[0, t].set_title(f"t={t + 1}", fontsize=8)
        pred_im = img_axes[1, t].imshow(pred[t], norm=pred_norm, cmap="Blues")
        diff_im = img_axes[2, t].imshow(diff[t], vmin=-diff_max, vmax=diff_max, cmap="RdBu_r")

    gt_im = img_axes[0, max_col].imshow(gt_max_map, norm=gt_norm, cmap="Blues")
    img_axes[0, max_col].set_title("Max", fontsize=8)
    pred_im = img_axes[1, max_col].imshow(pred_max_map, norm=pred_norm, cmap="Blues")
    diff_im = img_axes[2, max_col].imshow(
        diff_max_map, vmin=-diff_max, vmax=diff_max, cmap="RdBu_r"
    )

    row_labels = ["Ground\ntruth", "Prediction", "Difference"]
    for row in range(3):
        for t in range(n_steps + 1):
            img_axes[row, t].set_xticks([])
            img_axes[row, t].set_yticks([])
            img_axes[row, t].invert_yaxis()
        img_axes[row, 0].set_ylabel(row_labels[row], fontsize=8)
        # Visually set the max-over-time column apart from the per-step ones.
        for spine in img_axes[row, max_col].spines.values():
            spine.set_linewidth(1.5)

    fig = img_axes[0, 0].figure
    fig.colorbar(gt_im, cax=cbar_ax_gt, label=f"GT depth (m)\n(clip p{DEPTH_COLOR_PERCENTILE:g}, shared w/ pred)")
    fig.colorbar(pred_im, cax=cbar_ax_pred, label=f"Pred depth (m)\n(clip p{DEPTH_COLOR_PERCENTILE:g}, shared w/ GT)")
    fig.colorbar(diff_im, cax=cbar_ax_diff, label="Δ Depth (m)")
    for cax in (cbar_ax_gt, cbar_ax_pred, cbar_ax_diff):
        cax.tick_params(labelsize=7)
        cax.yaxis.label.set_size(7)

    header_ax.axis("off")
    header_ax.text(
        0.5, 0.5,
        f"GT: max={label.max():.3f}  min={label.min():.3f}  mean={label.mean():.3f} m   |   "
        f"Pred: max={pred.max():.3f}  min={pred.min():.3f}  mean={pred.mean():.3f} m   |   "
        f"MAE={mae:.4f} m",
        ha="center", va="center", fontsize=9, transform=header_ax.transAxes,
    )


def plot_city_multistep(city_name, examples, out_path):
    """Plots all examples for one city in a single stacked figure.

    Each example occupies its own header line plus a 3-row (GT / prediction
    / diff) block, stacked one after another via an explicit GridSpec so the
    header text and colorbars each get dedicated, non-overlapping space.

    Args:
        city_name: City name, used in the figure title.
        examples: List of dicts with keys "pred", "label", "mae".
        out_path: File path to save the figure to.
    """
    n_examples = len(examples)
    n_steps = examples[0]["pred"].shape[0]
    rows_per_example = 4  # header + GT + prediction + diff

    # +1 column for the pixelwise max-over-time map, +1 for the colorbars.
    fig = plt.figure(figsize=(1.9 * (n_steps + 1) + 1.3, 2.3 * 3 * n_examples))
    gs = gridspec.GridSpec(
        rows_per_example * n_examples, n_steps + 2,
        height_ratios=[0.5, 1, 1, 1] * n_examples,
        width_ratios=[1] * n_steps + [1, 0.12],
        hspace=0.45, wspace=0.08,
        figure=fig,
    )

    for e, example in enumerate(examples):
        base = rows_per_example * e
        header_ax = fig.add_subplot(gs[base, :-1])
        img_axes = np.array([
            [fig.add_subplot(gs[base + 1 + row, t]) for t in range(n_steps + 1)]
            for row in range(3)
        ])
        cbar_ax_gt = fig.add_subplot(gs[base + 1, -1])
        cbar_ax_pred = fig.add_subplot(gs[base + 2, -1])
        cbar_ax_diff = fig.add_subplot(gs[base + 3, -1])

        plot_example_block(
            img_axes, header_ax, cbar_ax_gt, cbar_ax_pred, cbar_ax_diff,
            example["pred"], example["label"], example["mae"],
        )

    fig.suptitle(f"{city_name} — {n_examples} test examples", fontsize=13, y=0.995)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def collect_city_examples(model, city_sims, max_examples, n_steps):
    """Collects `max_examples` uniformly random N_STEPS rollouts for one city.

    Scans the full stream of samples for the city (dataset yields one sample
    at a time, since load_test_ds uses batch_size=1), keeping only those
    whose ground-truth max depth (across all n_steps and the whole patch)
    falls within GT_MAX_DEPTH_RANGE.

    The dataset only shuffles the order of spatial chunks it visits; within
    a chunk, patches are scanned in raster order and, for a given patch,
    consecutive starting timesteps are yielded back-to-back. Two consecutive
    windows share n_steps-1 of their n_steps ground-truth frames (shifted by
    one step), so simply taking the first `max_examples` qualifying samples
    tends to pick near-duplicate windows of the same patch. Reservoir
    sampling (Algorithm R) instead selects uniformly at random from the
    *entire* qualifying pool in one pass, and only the final selections are
    run through the (expensive, autoregressive) model prediction.
    """
    if not city_sims:
        return []

    reservoir = []  # list of (inputs, label); label is a [n_steps, H, W] array
    n_seen = 0
    ds = load_test_ds(city_sims, n_steps)

    for inputs, labels in ds:
        # print(labels.shape)
        # Check this sample's ground-truth depth criteria first, before
        # running any (expensive, autoregressive) prediction.
        label = labels.numpy()[0]  # [n_steps, H, W]
        gt_max_depth = float(label.max())
        if not (GT_MAX_DEPTH_RANGE[0] <= gt_max_depth <= GT_MAX_DEPTH_RANGE[1]):
            continue

        n_seen += 1
        if len(reservoir) < max_examples:
            reservoir.append((inputs, label))
        else:
            j = random.randint(0, 3 * max_examples)
            if j < max_examples:
                reservoir[j] = (inputs, label)

    # Only now run prediction, for the reservoir's final selections.
    examples = []
    for inputs, label in reservoir:
        pred_seq = predict_n_steps(model, inputs, n_steps)
        pred = np.clip(pred_seq.numpy(), 0, 4).squeeze(axis=-1)[0]  # [n_steps, H, W]
        mae = float(np.abs(pred - label).mean())
        examples.append({
            "pred": pred,
            "label": label,
            "mae": mae,
        })

    return examples


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot autoregressive flood prediction rollouts vs. ground truth."
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save figures to (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--gt-max-depth-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=DEFAULT_GT_MAX_DEPTH_RANGE,
        help="Only plot examples whose ground-truth max flood depth (m) falls "
        f"in this range (default: {DEFAULT_GT_MAX_DEPTH_RANGE})",
    )
    return parser.parse_args()


def main():
    global OUTPUT_DIR, GT_MAX_DEPTH_RANGE

    args = parse_args()
    OUTPUT_DIR = args.output_dir
    GT_MAX_DEPTH_RANGE = tuple(args.gt_max_depth_range)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # random.seed(RANDOM_SEED)
    SEED = 42
    keras.utils.set_random_seed(SEED)

    model = build_model()
    load_weights(model, CHECKPOINT_PATH)

    for city_sims, city_name in zip(TEST_SIMS, city_names):
        print(f"\n{city_name} ({len(city_sims)} sims):")
        for s in city_sims:
            print(f"  {s}")

        examples = collect_city_examples(model, city_sims, EXAMPLES_PER_CITY, N_STEPS)
        if not examples:
            print(f"  No valid {N_STEPS}-step test patches found for {city_name}, skipping.")
            continue

        out_path = OUTPUT_DIR / f"{city_name}_multistep_examples.png"
        plot_city_multistep(city_name, examples, out_path)
        print(f"Saved: {out_path} ({len(examples)} examples)")

    del model
    tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
