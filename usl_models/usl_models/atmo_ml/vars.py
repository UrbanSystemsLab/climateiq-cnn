from enum import Enum
import dataclasses


@dataclasses.dataclass
class VarConfig:
    """Config per variable."""

    vmin: float = 0.0
    vmax: float = 1.0


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


ST_VAR_CONFIGS: dict[Spatiotemporal, VarConfig] = {
    Spatiotemporal.PRES: VarConfig(vmin=0.0, vmax=0.1),
    Spatiotemporal.GHT: VarConfig(vmin=0.0, vmax=0.1),
    Spatiotemporal.RH: VarConfig(vmin=0.0, vmax=1.0),
    Spatiotemporal.TT: VarConfig(vmin=0.0, vmax=1.0),
    Spatiotemporal.WSPD: VarConfig(vmin=0.0, vmax=0.1),
}


class SpatiotemporalOutput(Enum):
    """Spatiotemporal output channels used by the ML model."""

    RH2 = 0
    T2 = 1
    WSPD_WDIR10 = 2
    WSPD_WDIR10_SIN = 3
    WSPD_WDIR10_COS = 4


STO_VAR_CONFIGS: dict[SpatiotemporalOutput, VarConfig] = {
    SpatiotemporalOutput.RH2: VarConfig(vmin=0.0, vmax=100.0),
    SpatiotemporalOutput.T2: VarConfig(vmin=0.0, vmax=1.0),
    SpatiotemporalOutput.WSPD_WDIR10: VarConfig(vmin=0.0, vmax=10.0),
    SpatiotemporalOutput.WSPD_WDIR10_SIN: VarConfig(vmin=-1.0, vmax=1.0),
    SpatiotemporalOutput.WSPD_WDIR10_COS: VarConfig(vmin=-1.0, vmax=1.0),
}
