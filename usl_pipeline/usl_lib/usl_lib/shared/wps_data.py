"""List of required WPS variables that will be used ML modeling.

NOTE: Existing variables in the list should not be re-ordered.
New variables should be added to the end of the list.
"""

from typing import TypedDict
from enum import Enum, unique


@unique
class ScalingType(Enum):
    """WPS variable scaling type."""

    NONE = 1
    GLOBAL = 2
    LOCAL = 3


@unique
class Unit(Enum):
    """WPS variable unit type."""

    NONE = 1
    PASCALS = 2
    METERS = 3
    PERCENTAGE = 4
    KELVIN = 5
    FRACTION = 6
    METERSPERSEC = 7


class ScalingConfig(TypedDict, total=False):
    """WPS variable scaling config."""

    type: ScalingType
    min: float
    max: float


class VarConfig(TypedDict, total=False):
    """WPS variable config."""

    unit: Unit
    scaling: ScalingConfig


class VarType(Enum):
    """WPS variable type."""

    SPATIOTEMPORAL = "spatiotemporal"
    SPATIAL = "spatial"
    LU_INDEX = "lu_index"


class Var(Enum):
    """WPS variables used by the ML model."""

    PRES = 0
    GHT = 1
    RH = 2
    TT = 3
    LU_INDEX = 4
    ALBEDO12M = 5
    GREENFRAC = 6
    HGT_M = 7
    WSPD = 8
    WDIR_SIN = 9
    WDIR_COS = 10
    LAI12M = 11
    ST000010 = 12
    SM000010 = 13
    HGT_DIST_5m = 14
    HGT_DIST_10m = 15
    HGT_DIST_15m = 16
    HGT_DIST_20m = 17
    HGT_DIST_25m = 18
    HGT_DIST_30m = 19
    HGT_DIST_35m = 20
    HGT_DIST_40m = 21
    HGT_DIST_45m = 22
    HGT_DIST_50m = 23
    HGT_DIST_55m = 24
    HGT_DIST_60m = 25
    HGT_DIST_65m = 26
    HGT_DIST_70m = 27
    HGT_DIST_75m = 28
    AW_BUILD_HEIGHT = 29
    STDH_URB2D = 30
    BUILDING_AREA_FRACTION = 31
    FRC_URB2D = 32
    SOLAR_TIME_SIN = 33
    SOLAR_TIME_COS = 34
    # BUILD_HEIGHT = 35
    # BUILD_SURF_RATIO = 36


# Spatiotemporal variables used by the ML model (dimension H X W X T)
ML_REQUIRED_VARS: dict[VarType, list[Var]] = {
    VarType.SPATIOTEMPORAL: [
        Var.PRES,
        Var.GHT,
        Var.RH,
        Var.TT,
        Var.WSPD,
        Var.ALBEDO12M,
        Var.GREENFRAC,
        Var.WDIR_SIN,
        Var.WDIR_COS,
        Var.LAI12M,
        Var.SOLAR_TIME_SIN,
        Var.SOLAR_TIME_COS,
    ],
    VarType.SPATIAL: [
        Var.HGT_M,
        Var.ST000010,
        Var.SM000010,
        Var.HGT_DIST_5m,
        Var.HGT_DIST_10m,
        Var.HGT_DIST_15m,
        Var.HGT_DIST_20m,
        Var.HGT_DIST_25m,
        Var.HGT_DIST_30m,
        Var.HGT_DIST_35m,
        Var.HGT_DIST_40m,
        Var.HGT_DIST_45m,
        Var.HGT_DIST_50m,
        Var.HGT_DIST_55m,
        Var.HGT_DIST_60m,
        Var.HGT_DIST_65m,
        Var.HGT_DIST_70m,
        Var.HGT_DIST_75m,
        Var.AW_BUILD_HEIGHT,
        Var.STDH_URB2D,
        Var.BUILDING_AREA_FRACTION,
        Var.FRC_URB2D,
        # Var.BUILD_HEIGHT,
        # Var.BUILD_SURF_RATIO,
    ],
    VarType.LU_INDEX: [Var.LU_INDEX],
}

# Configs for each variable.
VAR_CONFIGS: dict[Var, VarConfig] = {
    # Surface pressure FNL level 0
    Var.PRES: VarConfig(
        unit=Unit.PASCALS,
        scaling=ScalingConfig(
            type=ScalingType.GLOBAL,
            min=98000,
            max=121590,
        ),
    ),
    # Geopotential height FNL level 0
    Var.GHT: VarConfig(
        unit=Unit.METERS,
        scaling=ScalingConfig(
            type=ScalingType.GLOBAL,
            min=0,
            max=6000,
        ),
    ),
    # Relative humidity FNL level 0
    Var.RH: VarConfig(
        unit=Unit.PERCENTAGE,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Temperature FNL level 0
    Var.TT: VarConfig(
        unit=Unit.KELVIN,
        scaling=ScalingConfig(
            type=ScalingType.GLOBAL,
            min=263.15,
            max=333.15,
        ),
    ),
    # LANDUSEF is a percentage of each LU_INDEX category (61)
    # Var.LANDUSEF: VarConfig(
    #     unit=Unit.FRACTION,
    #     scaling=ScalingConfig(
    #         type=ScalingType.NONE,
    #     ),
    # ),
    # LU_INDEX is 61 cat. LCZ data 1 feature
    Var.LU_INDEX: VarConfig(
        unit=Unit.NONE,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Monthly Climatology MODIS surface albedo
    Var.ALBEDO12M: VarConfig(
        unit=Unit.PERCENTAGE,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Monthly Climatology MODIS green fraction (MODIS FPAR)
    Var.GREENFRAC: VarConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # GMTED2010 30-arc-second topography height
    Var.HGT_M: VarConfig(
        unit=Unit.METERS,
        scaling=ScalingConfig(
            type=ScalingType.GLOBAL,
            min=0,
            max=5100,
        ),
    ),
    # [Derived] FNL level 0 (~10m) Wind Speed from UU and VV
    Var.WSPD: VarConfig(
        unit=Unit.METERSPERSEC,
        scaling=ScalingConfig(
            type=ScalingType.GLOBAL,
            min=0,
            max=100,
        ),
    ),
    # [Derived] FNL level 0 (~10m) Wind Direction (Cyclic feature) from UU and VV
    #  Sine Component of WDIR10
    Var.WDIR_SIN: VarConfig(
        unit=Unit.NONE,
        scaling=ScalingConfig(
            type=ScalingType.GLOBAL,
            min=-1,
            max=1,
        ),
    ),
    # [Derived] FNL level 0 (~10m) Wind Direction (Cyclic feature) from UU and VV
    # Cosine Component of WDIR10
    Var.WDIR_COS: VarConfig(
        unit=Unit.NONE,
        scaling=ScalingConfig(
            type=ScalingType.GLOBAL,
            min=-1,
            max=1,
        ),
    ),
    # Monthly Climatology MODIS Leaf Area Index
    Var.LAI12M: VarConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.GLOBAL,
            min=0,
            max=10,
        ),
    ),
    # Soil Temp layer 0-10cm below ground (WPS Initial Condition)
    # THIS IS FOR SUMMER!!! (-10C to 60C)
    Var.ST000010: VarConfig(
        unit=Unit.KELVIN,
        scaling=ScalingConfig(
            type=ScalingType.GLOBAL,
            min=263.15,
            max=333.15,
        ),
    ),
    # Soil Moisture layer 0-10cm below ground (WPS Initial Condition)
    Var.SM000010: VarConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP1 mean building height
    # Var.BUILD_HEIGHT: VarConfig(
    #     unit=Unit.METERS,
    #     scaling=ScalingConfig(
    #         type=ScalingType.GLOBAL,
    #         min=0,
    #         max=150,
    #     ),
    # ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 0-5m frequency bin
    Var.HGT_DIST_5m: VarConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 5-10m frequency bin
    Var.HGT_DIST_10m: VarConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 10-15m frequency bin
    Var.HGT_DIST_15m: VarConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 15-20m frequency bin
    Var.HGT_DIST_20m: VarConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 20-25m frequency bin
    Var.HGT_DIST_25m: VarConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 25-30m frequency bin
    Var.HGT_DIST_30m: VarConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 30-35m frequency bin
    Var.HGT_DIST_35m: VarConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 35-40m frequency bin
    Var.HGT_DIST_40m: VarConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 40-45m frequency bin
    Var.HGT_DIST_45m: VarConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 45-50m frequency bin
    Var.HGT_DIST_50m: VarConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 50-55m frequency bin
    Var.HGT_DIST_55m: VarConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 55-60m frequency bin
    Var.HGT_DIST_60m: VarConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 60-65m frequency bin
    Var.HGT_DIST_65m: VarConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 65-70m frequency bin
    Var.HGT_DIST_70m: VarConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 70-75m frequency bin
    Var.HGT_DIST_75m: VarConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP3 Area weighted mean building height
    Var.AW_BUILD_HEIGHT: VarConfig(
        unit=Unit.METERS,
        scaling=ScalingConfig(
            type=ScalingType.GLOBAL,
            min=0,
            max=250,
        ),
    ),
    # Custom UCPs for cities.
    # UCP4 Standard deviation of building height
    Var.STDH_URB2D: VarConfig(
        unit=Unit.METERS,
        scaling=ScalingConfig(
            type=ScalingType.GLOBAL,
            min=0,
            max=200,
        ),
    ),
    # Custom UCPs for cities.
    # UCP5 Plan area fraction
    Var.BUILDING_AREA_FRACTION: VarConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP6 Building surface to plan area ratio. Update for Very Dense Cities
    # Var.BUILD_SURF_RATIO: VarConfig(
    #     unit=Unit.FRACTION,
    #     scaling=ScalingConfig(
    #         type=ScalingType.NONE,
    #         min=0,
    #         max=200,
    #     ),
    # ),
    # Custom UCPs for cities.
    # UCP7 Urban fraction
    Var.FRC_URB2D: VarConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # [Derived] Solar Time from UTC (Cyclic feature)
    # in Minutes of Day (MIN)
    # Sine Component of Solar Time
    Var.SOLAR_TIME_SIN: VarConfig(
        unit=Unit.NONE,
        scaling=ScalingConfig(
            type=ScalingType.GLOBAL,
            min=-1,
            max=1,
        ),
    ),
    # [Derived] Solar Time from UTC (Cyclic feature)
    # in Minutes of Day (MIN)
    # Cosine Component of Solar Time
    Var.SOLAR_TIME_COS: VarConfig(
        unit=Unit.NONE,
        scaling=ScalingConfig(
            type=ScalingType.GLOBAL,
            min=-1,
            max=1,
        ),
    ),
}
