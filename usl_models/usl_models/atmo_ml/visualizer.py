from typing import Iterator

import itertools
import matplotlib.figure
import matplotlib.axes
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
import tensorflow as tf

from usl_models.atmo_ml import vars
from usl_models.atmo_ml import dataset


def init_plt():
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

    If dynamic_colorscale is True, compute vmin and vmax from the data.
    """
    T, H, W, *_ = data.shape

    if dynamic_colorscale:
        vmin, vmax = get_min_max(data)

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
    H, W, *_ = data.shape

    # Keep one scale per figure when dynamic_colorscale=True
    if dynamic_colorscale:
        heatmap_kwargs: dict[str, float | bool | None] = {
            "vmin": vmin,
            "vmax": vmax,
            "robust": False,
        }
    else:
        heatmap_kwargs = dict(vmin=vmin, vmax=vmax, robust=True)

    # Center diff maps at zero for a clean diverging mapping
    if "[diff]" in title:
        heatmap_kwargs["center"] = 0.0
        # Use a diverging colormap for difference plots
        heatmap_kwargs["cmap"] = "RdBu_r"

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
    F = len(features)
    fig, axs = plt.subplots(1, F, figsize=(2 * (F + 1), 2), sharey=True)
    if F == 1:
        axs = [axs]  # Make it consistent when only one subplot

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

    If dynamic_colorscale is True, the color limits are computed from the data.
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
    yield plot_2d_timeseries(
        inputs["spatiotemporal"][:, :, :, st_var.value],
        title=st_var.name + f" ({sim_name} {date})",
        vmin=st_var_config.vmin,
        vmax=st_var_config.vmax,
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
    if label is not None:
        if dynamic_colorscale:
            gt_min, gt_max = get_min_max(label)
            use_vmin, use_vmax = gt_min, gt_max
        else:
            use_vmin, use_vmax = sto_var_config.vmin, sto_var_config.vmax
        yield plot_2d_timeseries(
            label[:, :, :, sto_i],
            title=sto_var.name + f" [true] ({sim_name} {date})",
            vmin=use_vmin,
            vmax=use_vmax,
            t_start=0.0,
            t_interval=0.5,
            dynamic_colorscale=dynamic_colorscale,
        )
    if pred is not None:
        if dynamic_colorscale:
            pred_min, pred_max = get_min_max(pred)
            use_vmin, use_vmax = pred_min, pred_max
        else:
            use_vmin, use_vmax = sto_var_config.vmin, sto_var_config.vmax
        yield plot_2d_timeseries(
            pred[:, :, :, sto_i],
            title=sto_var.name + f" [pred] ({sim_name} {date})",
            vmin=use_vmin,
            vmax=use_vmax,
            t_start=0.0,
            t_interval=0.5,
            dynamic_colorscale=dynamic_colorscale,
        )

    # Plot the difference between prediction and ground truth
    if label is not None and pred is not None:
        diff = pred[:, :, :, sto_i] - label[:, :, :, sto_i]
        diff_range = max(abs(sto_var_config.vmin), abs(sto_var_config.vmax))
        yield plot_2d_timeseries(
            diff,
            title=sto_var.name + f" [diff] ({sim_name} {date})",
            vmin=-diff_range,
            vmax=diff_range,
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

    If dynamic_colorscale is True, the color limits are computed from the data.
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


def plot_task_balance_simple(history, sto_vars, save_path=None):
    """Simple plot of task balance during training."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot individual task losses
    task_losses = {}
    for sto_var in sto_vars:
        metric_name = f"mse_{sto_var.name}"
        if metric_name in history.history:
            ax1.plot(history.history[metric_name], label=sto_var.name, linewidth=2)
            task_losses[sto_var.name] = history.history[metric_name]

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Task MSE Loss")
    ax1.set_title("Individual Task Performance")
    ax1.legend()
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # Calculate and plot balance over time
    if task_losses:
        balance_history = []
        min_len = min(len(losses) for losses in task_losses.values())

        for epoch in range(min_len):
            epoch_losses = [losses[epoch] for losses in task_losses.values()]
            if len(epoch_losses) > 1:
                mean_loss = np.mean(epoch_losses)
                std_loss = np.std(epoch_losses)
                balance = std_loss / mean_loss if mean_loss > 0 else 0
                balance_history.append(balance)
            else:
                balance_history.append(0)

        ax2.plot(balance_history, "r-", linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Task Balance (CV)")
        ax2.set_title("Task Balance Over Time (Lower = Better)")
        ax2.grid(True, alpha=0.3)

        # Add reference lines
        ax2.axhline(
            y=0.3, color="green", linestyle="--", alpha=0.7, label="Good Balance"
        )
        ax2.axhline(
            y=0.5, color="orange", linestyle="--", alpha=0.7, label="Fair Balance"
        )
        ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def compare_task_performance(model, val_ds, ds_config):
    """Simple comparison of task performance."""
    # Get one batch for analysis
    for inputs, labels in val_ds.take(1):
        predictions = model.call(inputs)
        break

    # Calculate simple metrics per task
    print("\nTask Performance Analysis:")
    print("=" * 40)

    for i, sto_var in enumerate(ds_config.sto_vars):
        if predictions.shape[-1] > i and labels.shape[-1] > i:
            pred_task = predictions[..., i].numpy()
            true_task = labels[..., i].numpy()

            # Simple metrics
            mse = np.mean((pred_task - true_task) ** 2)
            mae = np.mean(np.abs(pred_task - true_task))

            # Correlation
            pred_flat = pred_task.flatten()
            true_flat = true_task.flatten()
            corr = np.corrcoef(pred_flat, true_flat)[0, 1] if len(pred_flat) > 1 else 0

            print(f"{sto_var.name}:")
            print(f"  MSE: {mse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  Correlation: {corr:.4f}")


def plot_training_metrics(history, save_path=None):
    """Plot comprehensive training metrics."""
    fig = plt.figure(figsize=(15, 10))

    # Create a grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Main loss plot
    ax1 = fig.add_subplot(gs[0, :2])
    if "loss" in history.history:
        ax1.plot(history.history["loss"], label="Training Loss", linewidth=2)
    if "val_loss" in history.history:
        ax1.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # RMSE plot
    ax2 = fig.add_subplot(gs[0, 2])
    if "root_mean_squared_error" in history.history:
        ax2.plot(
            history.history["root_mean_squared_error"], label="Train RMSE", linewidth=2
        )
    if "val_root_mean_squared_error" in history.history:
        ax2.plot(
            history.history["val_root_mean_squared_error"],
            label="Val RMSE",
            linewidth=2,
        )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("RMSE")
    ax2.set_title("Root Mean Squared Error")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # MAE plot
    ax3 = fig.add_subplot(gs[1, 0])
    if "mean_absolute_error" in history.history:
        ax3.plot(history.history["mean_absolute_error"], label="Train MAE", linewidth=2)
    if "val_mean_absolute_error" in history.history:
        ax3.plot(
            history.history["val_mean_absolute_error"], label="Val MAE", linewidth=2
        )
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("MAE")
    ax3.set_title("Mean Absolute Error")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # SSIM plot
    ax4 = fig.add_subplot(gs[1, 1])
    if "ssim_metric" in history.history:
        ax4.plot(history.history["ssim_metric"], label="Train SSIM", linewidth=2)
    if "val_ssim_metric" in history.history:
        ax4.plot(history.history["val_ssim_metric"], label="Val SSIM", linewidth=2)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("SSIM")
    ax4.set_title("Structural Similarity Index")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # PSNR plot
    ax5 = fig.add_subplot(gs[1, 2])
    if "psnr_metric" in history.history:
        ax5.plot(history.history["psnr_metric"], label="Train PSNR", linewidth=2)
    if "val_psnr_metric" in history.history:
        ax5.plot(history.history["val_psnr_metric"], label="Val PSNR", linewidth=2)
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("PSNR")
    ax5.set_title("Peak Signal-to-Noise Ratio")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Learning rate plot (if available)
    ax6 = fig.add_subplot(gs[2, 0])
    if "lr" in history.history:
        ax6.plot(history.history["lr"], linewidth=2, color="orange")
        ax6.set_xlabel("Epoch")
        ax6.set_ylabel("Learning Rate")
        ax6.set_title("Learning Rate Schedule")
        ax6.grid(True, alpha=0.3)
        ax6.set_yscale("log")
    else:
        ax6.text(
            0.5,
            0.5,
            "Learning Rate\nNot Available",
            ha="center",
            va="center",
            transform=ax6.transAxes,
        )
        ax6.set_xticks([])
        ax6.set_yticks([])

    # Task balance metrics (if multiple tasks)
    ax7 = fig.add_subplot(gs[2, 1:])
    task_metrics = [key for key in history.history.keys() if key.startswith("mse_")]
    if len(task_metrics) > 1:
        for metric in task_metrics:
            task_name = metric.replace("mse_", "")
            ax7.plot(history.history[metric], label=task_name, linewidth=2)
        ax7.set_xlabel("Epoch")
        ax7.set_ylabel("Task MSE")
        ax7.set_title("Individual Task Performance")
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        ax7.set_yscale("log")
    else:
        ax7.text(
            0.5,
            0.5,
            "Multi-task Metrics\nNot Available",
            ha="center",
            va="center",
            transform=ax7.transAxes,
        )
        ax7.set_xticks([])
        ax7.set_yticks([])

    plt.suptitle("Training Metrics Overview", fontsize=16, y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_prediction_statistics(predictions, labels, sto_vars, save_path=None):
    """Plot statistical analysis of predictions vs ground truth."""
    n_vars = len(sto_vars)
    fig, axes = plt.subplots(2, n_vars, figsize=(4 * n_vars, 8))

    if n_vars == 1:
        axes = axes.reshape(2, 1)

    for i, sto_var in enumerate(sto_vars):
        pred_var = predictions[..., i].flatten()
        true_var = labels[..., i].flatten()

        # Scatter plot
        ax1 = axes[0, i]
        ax1.scatter(true_var, pred_var, alpha=0.6, s=1)

        # Perfect prediction line
        min_val, max_val = min(true_var.min(), pred_var.min()), max(
            true_var.max(), pred_var.max()
        )
        ax1.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            linewidth=2,
            label="Perfect Prediction",
        )

        # Linear fit
        z = np.polyfit(true_var, pred_var, 1)
        p = np.poly1d(z)
        ax1.plot(
            true_var,
            p(true_var),
            "g-",
            linewidth=2,
            alpha=0.8,
            label=f"Fit: y={z[0]:.3f}x+{z[1]:.3f}",
        )

        ax1.set_xlabel("True Values")
        ax1.set_ylabel("Predicted Values")
        ax1.set_title(f"{sto_var.name} - Predictions vs Truth")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Residual plot
        ax2 = axes[1, i]
        residuals = pred_var - true_var
        ax2.scatter(true_var, residuals, alpha=0.6, s=1)
        ax2.axhline(y=0, color="r", linestyle="--", linewidth=2)
        ax2.set_xlabel("True Values")
        ax2.set_ylabel("Residuals (Pred - True)")
        ax2.set_title(f"{sto_var.name} - Residual Analysis")
        ax2.grid(True, alpha=0.3)

        # Add statistics text
        corr = np.corrcoef(true_var, pred_var)[0, 1]
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))

        stats_text = f"Corr: {corr:.3f}\nRMSE: {rmse:.3f}\nMAE: {mae:.3f}"
        ax2.text(
            0.05,
            0.95,
            stats_text,
            transform=ax2.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
