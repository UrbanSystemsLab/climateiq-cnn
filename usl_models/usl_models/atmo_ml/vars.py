from enum import Enum
import dataclasses

import numpy as np
import tensorflow as tf
import keras


@dataclasses.dataclass
class VarConfig:
    """Config per variable."""

    vmin: float = 0.0
    vmax: float = 1.0

    norm_vmin: float = 0.0
    norm_vmax = 1.0

    def scale(self, x: np.ndarray | tf.Tensor) -> np.ndarray | tf.Tensor:
        """Apply min max scaling."""
        x[x > self.vmax] = self.vmax
        x[x < self.vmin] = self.vmin
        return (x - self.vmin) / (self.vmax - self.vmin)


class Spatiotemporal(Enum):
    """Spatiotemporal variables used by the ML model."""

    PRES = 0
    GHT = 1
    RH = 2
    TT = 3
    WSPD = 4
    ALBEDO12M = 5
    GREENFRAC = 6
    WDIR_SIN = 7
    WDIR_COS = 8
    LAI12M = 9
    SOLAR_TIME_SIN = 10
    SOLAR_TIME_COS = 11

    def scale(self, x: np.ndarray | tf.Tensor):
        """Apply min max scaling."""
        return ST_VAR_CONFIGS[self].scale(x)


ST_VAR_CONFIGS: dict[Spatiotemporal, VarConfig] = {
    Spatiotemporal.PRES: VarConfig(vmin=0.0, vmax=0.1),
    Spatiotemporal.GHT: VarConfig(vmin=0.0, vmax=0.1),
    Spatiotemporal.RH: VarConfig(vmin=0.0, vmax=1.0),
    Spatiotemporal.TT: VarConfig(vmin=0.0, vmax=1.0),
    Spatiotemporal.WSPD: VarConfig(vmin=0.0, vmax=0.1),
}


@keras.saving.register_keras_serializable()
class SpatiotemporalOutput(Enum):
    """Spatiotemporal output channels used by the ML model."""

    RH2 = 0
    T2 = 1
    WSPD_WDIR10 = 2
    WSPD_WDIR10_SIN = 3
    WSPD_WDIR10_COS = 4

    def get_config(self) -> dict:
        """Get config for tensorflow serialization."""
        return {"value": self.value}

    @classmethod
    def from_config(cls, config: dict) -> "SpatiotemporalOutput":
        """Get config for tensorflow serialization."""
        return cls(config["value"])

    def scale(self, x: np.ndarray | tf.Tensor):
        """Apply min max scaling."""
        return STO_VAR_CONFIGS[self].scale(x)


STO_VAR_CONFIGS: dict[SpatiotemporalOutput, VarConfig] = {
    SpatiotemporalOutput.RH2: VarConfig(vmin=0.0, vmax=100.0),
    SpatiotemporalOutput.T2: VarConfig(vmin=263.15, vmax=333.15),
    SpatiotemporalOutput.WSPD_WDIR10: VarConfig(vmin=0.0, vmax=100.0),
    SpatiotemporalOutput.WSPD_WDIR10_SIN: VarConfig(
        vmin=-1.0, vmax=1.0, norm_vmin=-1.0
    ),
    SpatiotemporalOutput.WSPD_WDIR10_COS: VarConfig(
        vmin=-1.0, vmax=1.0, norm_vmin=-1.0
    ),
}
