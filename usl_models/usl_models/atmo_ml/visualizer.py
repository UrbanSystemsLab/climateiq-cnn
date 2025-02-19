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


def plot_2d_timeseries(
    data: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    spatial_ticks: int = 5,
    title: str = "2d Timeseries Plot",
    t_interval: float = 1.0,
    t_start: float = 0.0,
    normalize: bool = False,  # Set to False to disable normalization
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
    normalize: bool = True,  # When False, plot raw data
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
    normalize: bool = True,
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
    inputs: dict[str, tf.Tensor],
    label: tf.Tensor | None = None,
    pred: tf.Tensor | None = None,
    st_var: vars.Spatiotemporal = vars.Spatiotemporal.RH,
    sto_var: vars.SpatiotemporalOutput = vars.SpatiotemporalOutput.RH2,
    spatial_features: list[int] | None = None,
    spatial_ticks: int = 6,
    normalize: bool = True,  # Set normalize to False to plot raw data
) -> list[matplotlib.figure.Figure]:
    """Plots inputs, label, prediction, and difference maps for debugging."""
    sim_name = inputs["sim_name"].numpy().decode("utf-8")
    date = inputs["date"].numpy().decode("utf-8")
    figs = []
    if spatial_features is not None:
        for i in range(len(spatial_features) // 5):
            figs.append(
                plot_spatial(
                    inputs["spatial"],
                    spatial_ticks=spatial_ticks,
                    features=spatial_features[5 * i : 5 * (i + 1)],
                    normalize=normalize,
                )
            )

    st_var_config = vars.ST_VAR_CONFIGS[st_var]
    figs.append(
        plot_2d_timeseries(
            inputs["spatiotemporal"][:, :, :, st_var.value],
            title=st_var.name + f" ({sim_name} {date})",
            vmin=st_var_config.vmin,
            vmax=st_var_config.vmax,
            t_start=-1.0,
            t_interval=1.0,
            normalize=normalize,
        )
    )
    sto_var_config = vars.STO_VAR_CONFIGS[sto_var]
    if label is not None:
        figs.append(
            plot_2d_timeseries(
                label[:, :, :, sto_var.value],
                title=sto_var.name + f" [true] ({sim_name} {date})",
                vmin=sto_var_config.norm_vmin,
                vmax=sto_var_config.norm_vmax,
                t_start=0.0,
                t_interval=0.5,
                normalize=normalize,
            )
        )
    if pred is not None:
        figs.append(
            plot_2d_timeseries(
                pred[:, :, :, sto_var.value],
                title=sto_var.name + f" [pred] ({sim_name} {date})",
                vmin=sto_var_config.norm_vmin,
                vmax=sto_var_config.norm_vmax,
                t_start=0.0,
                t_interval=0.5,
                normalize=normalize,
            )
        )
    # Plot the difference between prediction and ground truth
    if label is not None and pred is not None:
        diff = pred[:, :, :, sto_var.value] - label[:, :, :, sto_var.value]
        # Use symmetric limits centered at zero for the difference
        diff_range = max(
            (
                abs(sto_var_config.norm_vmin)
                if sto_var_config.norm_vmin is not None
                else 0
            ),
            (
                abs(sto_var_config.norm_vmax)
                if sto_var_config.norm_vmax is not None
                else 0
            ),
        )
        figs.append(
            plot_2d_timeseries(
                diff,
                title=sto_var.name + f" [diff] ({sim_name} {date})",
                vmin=-diff_range,
                vmax=diff_range,
                t_start=0.0,
                t_interval=0.5,
                normalize=normalize,
            )
        )
    return figs
