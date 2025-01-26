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
    vmin: float,
    vmax: float,
    spatial_ticks: int = 5,
    title: str = "2d Timeseries Plot",
    t_interval: float = 1.0,
    t_start: float = 0.0,
) -> matplotlib.figure.Figure:
    """Plot a map of atmo data."""
    T, H, W, *_ = data.shape

    fig, axs = plt.subplots(1, T, figsize=(10, 2), sharey=True)
    cbar_ax = fig.add_axes((0.91, 0.3, 0.01, 0.5))
    fig.subplots_adjust()

    for t in range(T):
        _plot_2d(
            data=data[t],
            ax=axs[t],
            spatial_ticks=spatial_ticks,
            vmin=vmin,
            vmax=vmax,
            title=f"t={t_start + (t * t_interval)}",
            cbar_ax=cbar_ax,
        )

    fig.suptitle(title)
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
) -> matplotlib.axes.Axes:
    H, W, *_ = data.shape
    sbn.heatmap(
        data,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        square=True,
        robust=True,
        xticklabels=False,
        yticklabels=False,
        cbar_ax=cbar_ax,
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
    data: np.ndarray, features: list[int], spatial_ticks: int = 5
) -> matplotlib.figure.Figure:
    F = len(features)
    fig, axs = plt.subplots(1, F, figsize=(10, 2), constrained_layout=True)
    for i, f in enumerate(features):
        _ = _plot_2d(
            data=data[:, :, f],
            ax=axs[i],
            title=f"Spatial feature {f}",
            spatial_ticks=spatial_ticks,
        )
    fig.suptitle("Spatial features")
    fig.set_dpi(200)
    return fig


def plot(
    inputs: dict[str, tf.Tensor],
    label: tf.Tensor,
    st_var: vars.Spatiotemporal,
    sto_var: vars.SpatiotemporalOutput,
    sim_name: str,
    date: str,
) -> tuple[matplotlib.figure.Figure, ...]:
    """Plots an inputs, label pair for debugging."""
    sp_fig0 = plot_spatial(
        inputs["spatial"], spatial_ticks=6, features=list(range(0, 5))
    )
    sp_fig1 = plot_spatial(
        inputs["spatial"], spatial_ticks=6, features=list(range(5, 10))
    )
    st_var_config = vars.ST_VAR_CONFIGS[st_var]
    st_fig = plot_2d_timeseries(
        inputs["spatiotemporal"][:, :, :, st_var.value],
        title=st_var.name + f" ({sim_name} {date})",
        vmin=st_var_config.vmin,
        vmax=st_var_config.vmax,
        t_start=-1.0,
        t_interval=1.0,
    )

    sto_var_config = vars.STO_VAR_CONFIGS[sto_var]
    sto_fig = plot_2d_timeseries(
        label[:, :, :, sto_var.value],
        title=sto_var.name + f" ({sim_name} {date})",
        vmin=sto_var_config.vmin,
        vmax=sto_var_config.vmax,
        t_start=0.0,
        t_interval=0.5,
    )

    return sp_fig0, sp_fig1, st_fig, sto_fig
