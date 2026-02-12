"""Visualization functions for AtmoML model """

import itertools
from typing import Iterator

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn
import tensorflow as tf

from usl_models.atmo_ml import dataset, vars


def init_plt():
    """Initialize matplotlib settings for consistent plotting."""
    plt.style.use("dark_background")
    plt.rcParams["font.size"] = 6


def get_min_max(data: np.ndarray | tf.Tensor) -> tuple[float, float]:
    """Returns the minimum and maximum values from data.

    If data is a TensorFlow tensor, uses tf.reduce_min and tf.reduce_max.
    Otherwise, uses numpy functions.
    """
    if isinstance(data, tf.Tensor):
        return float(tf.reduce_min(data).numpy()), float(tf.reduce_max(data).numpy())
    else:
        return float(np.min(data)), float(np.max(data))


def plot_2d_timeseries(
    data: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    spatial_ticks: int = 5,
    title: str = "2d Timeseries Plot",
    t_interval: float = 1.0,
    t_start: float = 0.0,
    dynamic_colorscale: bool = False,
) -> matplotlib.figure.Figure:
    """Plot a timeseries of 2D maps.

    If dynamic_colorscale is True, let seaborn determine vmin and vmax automatically.
    """
    T, H, W, *_ = data.shape

    # FIX #1: When dynamic_colorscale=True, we want seaborn to auto-scale
    # So we should NOT compute vmin/vmax, we pass None to let seaborn handle it
    if dynamic_colorscale:
        vmin = None  # Let seaborn determine
        vmax = None  # Let seaborn determine

    fig, axs = plt.subplots(1, T, figsize=(2 * (T + 0.2), 2), sharey=True)
    cbar_ax = fig.add_axes((0.91, 0.3, 0.06 / (T + 0.2), 0.5))

    for t in range(T):
        _plot_2d(
            data=data[t],
            ax=axs[t] if T > 1 else axs,
            spatial_ticks=spatial_ticks,
            vmin=vmin,
            vmax=vmax,
            title=f"t={t_start + (t * t_interval)}",
            cbar_ax=cbar_ax,
            dynamic_colorscale=dynamic_colorscale,
        )

    fig.subplots_adjust(top=0.85)
    fig.suptitle(title, y=1.0)
    fig.set_dpi(200)

    return fig


def _plot_2d(
    data: np.ndarray,
    ax: matplotlib.axes.Axes,
    spatial_ticks: int,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str = "2d Plot",
    cbar_ax: matplotlib.axes.Axes | None = None,
    dynamic_colorscale: bool = False,
) -> matplotlib.axes.Axes:
    """Plot a single 2D map."""
    H, W, *_ = data.shape

    # FIX #1: Correct implementation - when dynamic, use None for auto-scaling
    if dynamic_colorscale:
        heatmap_kwargs = {
            "vmin": None,  # Let seaborn auto-scale
            "vmax": None,  # Let seaborn auto-scale
            "robust": True,  # Use robust quantiles for better auto-scaling
        }
    else:
        # Use provided values or let seaborn decide if None
        heatmap_kwargs = dict(vmin=vmin, vmax=vmax, robust=True)

    sbn.heatmap(
        data,
        ax=ax,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar_ax=cbar_ax,
        **heatmap_kwargs,
    )
    xticks = np.linspace(0, W, spatial_ticks, dtype=np.int32)
    yticks = np.linspace(0, H, spatial_ticks, dtype=np.int32)
    ax.set_xticks(xticks, labels=xticks)
    ax.set_yticks(yticks, labels=yticks)
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    ax.set_title(title)
    return ax


def plot_spatial(
    data: np.ndarray,
    features: list[int],
    spatial_ticks: int = 5,
    dynamic_colorscale: bool = False,
) -> matplotlib.figure.Figure:
    """Plot multiple spatial features side by side."""
    F = len(features)
    fig, axs = plt.subplots(1, F, figsize=(2 * (F + 1), 2), sharey=True)

    # Handle single feature case
    if F == 1:
        axs = [axs]

    for i, f in enumerate(features):
        _ = _plot_2d(
            data=data[:, :, f],
            ax=axs[i],
            title=f"Spatial feature {f}",
            spatial_ticks=spatial_ticks,
            dynamic_colorscale=dynamic_colorscale,
        )
    fig.subplots_adjust(top=0.85)
    fig.suptitle("Spatial features")
    fig.set_dpi(200)
    return fig


def plot(
    config: dataset.Config,
    inputs: dict[str, tf.Tensor],
    label: tf.Tensor | None = None,
    pred: tf.Tensor | None = None,
    st_var: vars.Spatiotemporal = vars.Spatiotemporal.RH,
    sto_var: vars.SpatiotemporalOutput = vars.SpatiotemporalOutput.RH2,
    spatial_features: list[int] | None = None,
    spatial_ticks: int = 6,
    dynamic_colorscale: bool = False,
    unscale: bool = True,
) -> Iterator[matplotlib.figure.Figure]:
    """Plots inputs, label, prediction, and difference maps for debugging.

    If dynamic_colorscale is True, the color limits are determined automatically by seaborn.
    """
    sim_name = inputs["sim_name"].numpy().decode("utf-8")
    date = inputs["date"].numpy().decode("utf-8")
    if spatial_features is not None:
        for i in range(len(spatial_features) // 5):
            yield plot_spatial(
                inputs["spatial"],
                spatial_ticks=spatial_ticks,
                features=spatial_features[5 * i : 5 * (i + 1)],
                dynamic_colorscale=dynamic_colorscale,
            )

    st_var_config = vars.ST_VAR_CONFIGS[st_var]

    # When dynamic_colorscale=True, pass None for vmin/vmax
    input_vmin = None if dynamic_colorscale else st_var_config.vmin
    input_vmax = None if dynamic_colorscale else st_var_config.vmax

    yield plot_2d_timeseries(
        inputs["spatiotemporal"][:, :, :, st_var.value],
        title=st_var.name + f" ({sim_name} {date})",
        vmin=input_vmin,
        vmax=input_vmax,
        t_start=-1.0,
        t_interval=1.0,
        dynamic_colorscale=dynamic_colorscale,
    )

    sto_var_config = vars.STO_VAR_CONFIGS[sto_var]

    # If unscaling is enabled, revert the normalization for GT and predictions.
    if label is not None and unscale:
        label = sto_var_config.unscale(label.numpy())
    if pred is not None and unscale:
        pred = sto_var_config.unscale(pred.numpy())

    sto_i = config.sto_vars.index(sto_var)

    # When dynamic_colorscale=True, pass None for vmin/vmax
    output_vmin = None if dynamic_colorscale else sto_var_config.vmin
    output_vmax = None if dynamic_colorscale else sto_var_config.vmax

    if label is not None:
        yield plot_2d_timeseries(
            label[:, :, :, sto_i],
            title=sto_var.name + f" [true] ({sim_name} {date})",
            vmin=output_vmin,
            vmax=output_vmax,
            t_start=0.0,
            t_interval=0.5,
            dynamic_colorscale=dynamic_colorscale,
        )
    if pred is not None:
        yield plot_2d_timeseries(
            pred[:, :, :, sto_i],
            title=sto_var.name + f" [pred] ({sim_name} {date})",
            vmin=output_vmin,
            vmax=output_vmax,
            t_start=0.0,
            t_interval=0.5,
            dynamic_colorscale=dynamic_colorscale,
        )

    # Plot the difference between prediction and ground truth
    if label is not None and pred is not None:
        diff = pred[:, :, :, sto_i] - label[:, :, :, sto_i]

        if dynamic_colorscale:
            # Let seaborn determine the range for diff
            diff_vmin = None
            diff_vmax = None
        else:
            # Use symmetric range based on config
            diff_range = max(abs(sto_var_config.vmin), abs(sto_var_config.vmax))
            diff_vmin = -diff_range
            diff_vmax = diff_range

        yield plot_2d_timeseries(
            diff,
            title=sto_var.name + f" [diff] ({sim_name} {date})",
            vmin=diff_vmin,
            vmax=diff_vmax,
            t_start=0.0,
            t_interval=0.5,
            dynamic_colorscale=dynamic_colorscale,
        )


def plot_batch(
    config: dataset.Config,
    input_batch: tf.Tensor,
    label_batch: tf.Tensor,
    pred_batch: tf.Tensor,
    st_var: vars.Spatiotemporal = vars.Spatiotemporal.RH,
    sto_var: vars.SpatiotemporalOutput = vars.SpatiotemporalOutput.RH2,
    max_examples: int | None = None,
    dynamic_colorscale: bool = False,
    unscale: bool = True,
) -> Iterator[matplotlib.figure.Figure]:
    """Plot a batch of AtmoML Examples.

    If dynamic_colorscale is True, the color limits are determined automatically by seaborn.
    Otherwise, a general range from the configuration is used for GT and prediction.
    """
    for b, _ in itertools.islice(enumerate(label_batch), max_examples):
        for fig in plot(
            config,
            inputs={k: v[b] for k, v in input_batch.items()},
            label=label_batch[b],
            pred=pred_batch[b],
            st_var=st_var,
            sto_var=sto_var,
            dynamic_colorscale=dynamic_colorscale,
            unscale=unscale,
        ):
            yield fig


# ============================================================================
# HELPER FUNCTION (merged from plot_task_balance into plot_training_metrics)
# ============================================================================


def _extract_task_losses_from_history(history, sto_vars):
    """Helper to extract per-task losses from training history.

    This is a helper function used internally by plot_training_metrics.
    """
    task_losses = {}
    for var in sto_vars:
        key = f"mse_{var.name}"
        if key in history.history:
            task_losses[var.name] = history.history[key]
    return task_losses


# ============================================================================
# PUBLIC FUNCTION - MERGED plot_task_balance into plot_training_metrics
# ============================================================================


def plot_training_metrics(history, sto_vars=None, save_path=None):
    """Plot comprehensive training metrics including per-task losses.

    This function merges the functionality of plot_task_balance since they overlap.

    Args:
        history: Keras training history
        sto_vars: List of output variables (for task balance plotting)
        save_path: Optional path to save the figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Determine number of subplots needed
    n_plots = 4  # loss, mae, rmse, task_balance
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # 1. Plot overall loss
    axes[0].plot(history.history["loss"], label="Train Loss")
    if "val_loss" in history.history:
        axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Model Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 2. Plot MAE if available
    if "mean_absolute_error" in history.history:
        axes[1].plot(history.history["mean_absolute_error"], label="Train MAE")
        if "val_mean_absolute_error" in history.history:
            axes[1].plot(history.history["val_mean_absolute_error"], label="Val MAE")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE")
        axes[1].set_title("Mean Absolute Error")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    # 3. Plot RMSE if available
    if "root_mean_squared_error" in history.history:
        axes[2].plot(history.history["root_mean_squared_error"], label="Train RMSE")
        if "val_root_mean_squared_error" in history.history:
            axes[2].plot(
                history.history["val_root_mean_squared_error"], label="Val RMSE"
            )
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("RMSE")
        axes[2].set_title("Root Mean Squared Error")
        axes[2].legend()
        axes[2].grid(alpha=0.3)

    # 4. MERGED: Task Balance (previously plot_task_balance functionality)
    if sto_vars:
        task_losses = _extract_task_losses_from_history(history, sto_vars)
        if task_losses:
            for task_name, losses in task_losses.items():
                # Normalize to [0, 1] for comparison
                losses_norm = (losses - np.min(losses)) / (
                    np.max(losses) - np.min(losses) + 1e-8
                )
                axes[3].plot(losses_norm, label=task_name)
            axes[3].set_xlabel("Epoch")
            axes[3].set_ylabel("Normalized Task Loss")
            axes[3].set_title("Task Balance (Normalized Per-Task MSE)")
            axes[3].legend()
            axes[3].grid(alpha=0.3)
    else:
        # If no sto_vars provided, try to plot raw per-task losses
        task_losses = [
            k for k in history.history.keys() if "mse_" in k and "val" not in k
        ]
        if task_losses:
            for task_loss in task_losses:
                task_name = task_loss.replace("mse_", "")
                axes[3].plot(history.history[task_loss], label=f"{task_name}")
            axes[3].set_xlabel("Epoch")
            axes[3].set_ylabel("Task MSE")
            axes[3].set_title("Per-Task MSE")
            axes[3].legend()
            axes[3].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
