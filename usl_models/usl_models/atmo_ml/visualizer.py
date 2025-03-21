from typing import Iterator

import itertools
import matplotlib.figure
import matplotlib.axes
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
import tensorflow as tf

from usl_models.atmo_ml import vars


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
    if dynamic_colorscale:
        vmin, vmax = get_min_max(data)
        heatmap_kwargs: dict[str, float | bool | None] = {
            "vmin": None,
            "vmax": None,
            "robust": False,
        }
    else:
        # When not normalizing, let seaborn use the raw data range.
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
    F = len(features)
    fig, axs = plt.subplots(1, F, figsize=(2 * (F + 1), 2), sharey=True)
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

    if label is not None:
        if dynamic_colorscale:
            gt_min, gt_max = get_min_max(label)
            use_vmin, use_vmax = gt_min, gt_max
        else:
            use_vmin, use_vmax = sto_var_config.vmin, sto_var_config.vmax
        yield plot_2d_timeseries(
            label[:, :, :, sto_var.value],
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
            pred[:, :, :, sto_var.value],
            title=sto_var.name + f" [pred] ({sim_name} {date})",
            vmin=use_vmin,
            vmax=use_vmax,
            t_start=0.0,
            t_interval=0.5,
            dynamic_colorscale=dynamic_colorscale,
        )

    # Plot the difference between prediction and ground truth
    if label is not None and pred is not None:
        diff = pred[:, :, :, sto_var.value] - label[:, :, :, sto_var.value]
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
            inputs={k: v[b] for k, v in input_batch.items()},
            label=label_batch[b],
            pred=pred_batch[b],
            st_var=st_var,
            sto_var=sto_var,
            dynamic_colorscale=dynamic_colorscale,
            unscale=unscale,
        ):
            yield fig
