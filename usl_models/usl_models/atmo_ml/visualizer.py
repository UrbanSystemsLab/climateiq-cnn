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
    fig, axs = plt.subplots(1, F, figsize=(2 * (F + 1), 2), sharey=True)
    for i, f in enumerate(features):
        _ = _plot_2d(
            data=data[:, :, f],
            ax=axs[i],
            title=f"Spatial feature {f}",
            spatial_ticks=spatial_ticks,
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
    sim_name: str = "test",
    date: str = "undated",
    spatial_features: list[int] | None = None,
    spatial_ticks: int = 6,
) -> list[matplotlib.figure.Figure]:
    """Plots an inputs, label pair for debugging."""
    figs = []
    if spatial_features is not None:
        for i in range(len(spatial_features) // 5):
            figs.append(
                plot_spatial(
                    inputs["spatial"],
                    spatial_ticks=spatial_ticks,
                    features=spatial_features[5 * i : 5 * (i + 1)],
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
        )
    )

    sto_var_config = vars.STO_VAR_CONFIGS[sto_var]
    if label is not None:
        figs.append(
            plot_2d_timeseries(
                label[:, :, :, sto_var.value],
                title=sto_var.name + f" ({sim_name} {date})",
                vmin=sto_var_config.vmin,
                vmax=1.0,
                t_start=0.0,
                t_interval=0.5,
            )
        )
    if pred is not None:
        figs.append(
            plot_2d_timeseries(
                pred[:, :, :, sto_var.value],
                title=sto_var.name + f" [pred] ({sim_name} {date})",
                vmin=sto_var_config.vmin,
                vmax=1.0,
                t_start=0.0,
                t_interval=0.5,
            )
        )

    return figs
