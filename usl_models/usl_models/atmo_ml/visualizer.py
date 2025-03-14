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


def plot_2d_timeseries(
    data: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    spatial_ticks: int = 5,
    title: str = "2d Timeseries Plot",
    t_interval: float = 1.0,
    t_start: float = 0.0,
    normalize: bool = False,
) -> matplotlib.figure.Figure:
    """Plot a timeseries of 2D maps without normalization if desired."""
    T, H, W, *_ = data.shape

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
            normalize=normalize,
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
    normalize: bool = False,
) -> matplotlib.axes.Axes:
    H, W, *_ = data.shape
    if normalize:
        heatmap_kwargs = dict(vmin=vmin, vmax=vmax, robust=True)
    else:
        # When not normalizing, let seaborn use the raw data range.
        heatmap_kwargs = dict(vmin=None, vmax=None, robust=False)
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
    normalize: bool = False,
) -> matplotlib.figure.Figure:
    F = len(features)
    fig, axs = plt.subplots(1, F, figsize=(2 * (F + 1), 2), sharey=True)
    for i, f in enumerate(features):
        _ = _plot_2d(
            data=data[:, :, f],
            ax=axs[i],
            title=f"Spatial feature {f}",
            spatial_ticks=spatial_ticks,
            normalize=normalize,
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
    normalize: bool = False,  # Set normalize to False to plot raw data,
) -> Iterator[matplotlib.figure.Figure]:
    """Plots inputs, label, prediction, and difference maps for debugging."""
    sim_name = inputs["sim_name"].numpy().decode("utf-8")
    date = inputs["date"].numpy().decode("utf-8")
    if spatial_features is not None:
        for i in range(len(spatial_features) // 5):
            yield plot_spatial(
                inputs["spatial"],
                spatial_ticks=spatial_ticks,
                features=spatial_features[5 * i : 5 * (i + 1)],
                normalize=normalize,
            )

    st_var_config = vars.ST_VAR_CONFIGS[st_var]
    yield plot_2d_timeseries(
        inputs["spatiotemporal"][:, :, :, st_var.value],
        title=st_var.name + f" ({sim_name} {date})",
        vmin=st_var_config.vmin,
        vmax=st_var_config.vmax,
        t_start=-1.0,
        t_interval=1.0,
        normalize=normalize,
    )
    sto_var_config = vars.STO_VAR_CONFIGS[sto_var]
    sto_i = config.sto_vars.index(sto_var)
    if label is not None:
        yield plot_2d_timeseries(
            label[:, :, :, sto_i],
            title=sto_var.name + f" [true] ({sim_name} {date})",
            vmin=sto_var_config.norm_vmin,
            vmax=sto_var_config.norm_vmax,
            t_start=0.0,
            t_interval=0.5,
            normalize=normalize,
        )

    if pred is not None:
        yield plot_2d_timeseries(
            pred[:, :, :, sto_i],
            title=sto_var.name + f" [pred] ({sim_name} {date})",
            vmin=sto_var_config.norm_vmin,
            vmax=sto_var_config.norm_vmax,
            t_start=0.0,
            t_interval=0.5,
            normalize=normalize,
        )

    # Plot the difference between prediction and ground truth
    if label is not None and pred is not None:
        diff = pred[:, :, :, sto_i] - label[:, :, :, sto_i]
        # Use symmetric limits centered at zero for the difference
        diff_range = max(
            abs(sto_var_config.norm_vmin),
            abs(sto_var_config.norm_vmax),
        )
        yield plot_2d_timeseries(
            diff,
            title=sto_var.name + f" [diff] ({sim_name} {date})",
            vmin=-diff_range,
            vmax=diff_range,
            t_start=0.0,
            t_interval=0.5,
            normalize=normalize,
        )


def plot_batch(
    config: dataset.Config,
    input_batch: tf.Tensor,
    label_batch: tf.Tensor,
    pred_batch: tf.Tensor,
    st_var: vars.Spatiotemporal = vars.Spatiotemporal.RH,
    sto_var: vars.SpatiotemporalOutput = vars.SpatiotemporalOutput.RH2,
    normalize: bool = False,
    max_examples: int | None = None,
) -> Iterator[matplotlib.figure.Figure]:
    """Plot a batch of AtmoML Examples."""
    for b, _ in itertools.islice(enumerate(label_batch), max_examples):
        for fig in plot(
            config,
            inputs={k: v[b] for k, v in input_batch.items()},
            label=label_batch[b],
            pred=pred_batch[b],
            st_var=st_var,
            sto_var=sto_var,
            normalize=normalize,
        ):
            yield fig
