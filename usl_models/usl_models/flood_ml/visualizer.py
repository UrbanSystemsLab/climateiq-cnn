"""Visualize predictions and errors for the FloodML model.

Visualize predictions and errors for the FloodML model.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


def plot_temporal_errors(
    name: str,
    rainfall_duration: int,
    temporal_mae: tf.Tensor,
    temporal_rmse: tf.Tensor,
):
    """Plots temporal errors of a FloodML prediction."""
    plt.figure(figsize=(8, 4))
    plt.title(f"Temporal Test Errors for {name}")
    time = range(1, rainfall_duration + 1)
    plt.plot(time, temporal_mae, label="MAE")
    plt.plot(time, temporal_rmse, label="RMSE")
    plt.xticks(time)
    plt.xlabel("Time Step")
    plt.ylabel("Error")
    plt.legend()
    plt.show()


def plot_maps(
    name: str,
    spatial_mae: tf.Tensor,
    nse: tf.Tensor,
    pred: tf.Tensor,
    label: tf.Tensor,
):
    """Plot metrics for a simulation map chunk."""
    fig, axs = plt.subplots(2, 2, figsize=(9, 7))
    fig.suptitle(f"Highest MAE Test Chunk for {name}")
    fontsize = 11

    # ax1: Spatial MAE heatmap
    ax1 = axs[0, 0]
    max_mae = tf.reduce_mean(spatial_mae)
    sns.heatmap(
        spatial_mae,
        vmax=0.5,
        ax=ax1,
        xticklabels=[],
        yticklabels=[],
        cbar_kws={"label": "MAE (clipped to 0.5)"},
    )
    ax1.invert_yaxis()
    ax1.set_title(f"MAE = {max_mae:.7f}", fontsize=fontsize)

    # ax2: Spatial NSE heatmap
    ax2 = axs[0, 1]
    sns.heatmap(
        nse,
        vmax=1.0,
        vmin=-10.0,
        ax=ax2,
        xticklabels=[],
        yticklabels=[],
        cmap=sns.cm.rocket_r,
    )
    ax2.invert_yaxis()
    ax2.set_title("Spatial NSE", fontsize=fontsize)

    # ax3: Prediction
    ax3 = axs[1, 0]
    sns.heatmap(pred, vmax=0.5, ax=ax3, xticklabels=[], yticklabels=[])
    ax3.invert_yaxis()
    ax3.set_title("Prediction", fontsize=fontsize)

    # ax4: Label
    ax4 = axs[1, 1]
    sns.heatmap(label, vmax=0.5, ax=ax4, xticklabels=[], yticklabels=[])
    ax4.invert_yaxis()
    ax4.set_title("Label", fontsize=fontsize)

    fig.tight_layout()
    plt.show()
