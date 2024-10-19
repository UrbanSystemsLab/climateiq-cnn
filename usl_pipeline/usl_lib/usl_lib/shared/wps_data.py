"""List of required WPS variables that will be used ML modeling.

NOTE: Existing variables in the list should not be re-ordered.
New variables should be added to the end of the list.
"""

from typing import TypedDict
from enum import Enum


class ScalingType(Enum):
    NONE = 1
    GLOBAL = 2
    LOCAL = 3


class Unit(Enum):
    NONE = 1
    PASCALS = 2
    METERS = 3
    PERCENTAGE = 4
    KELVIN = 5
    FRACTION = 6
    METERSPERSEC = 7


class Variable(Enum):
    PRES = 0
    GHT = 1
    RH = 2
    TT = 3
    LANDUSEF = 4
    LU_INDEX = 5
    ALBEDO12M = 6
    GREENFRAC = 7
    HGT_M = 8
    WSPD = 9
    WDIR_SIN = 10
    WDIR_COS = 11
    LAI12M = 12
    ST000010 = 13
    SM000010 = 14
    BUILD_HEIGHT = 15
    HGT_DIST_5m = 16
    HGT_DIST_10m = 17
    HGT_DIST_15m = 18
    HGT_DIST_20m = 19
    HGT_DIST_25m = 20
    HGT_DIST_30m = 21
    HGT_DIST_35m = 22
    HGT_DIST_40m = 23
    HGT_DIST_45m = 24
    HGT_DIST_50m = 25
    HGT_DIST_55m = 26
    HGT_DIST_60m = 27
    HGT_DIST_65m = 28
    HGT_DIST_70m = 29
    HGT_DIST_75m = 30
    AW_BUILD_HEIGHT = 31
    STDH_URB2D = 32
    BUILDING_AREA_FRACTION = 33
    BUILD_SURF_RATIO = 34
    FRC_URB2D = 35
    SOLAR_TIME_SIN = 36
    SOLAR_TIME_COS = 37


class ScalingConfig(TypedDict, total=False):
    type: ScalingType
    min: float
    max: float


class VariableConfig(TypedDict, total=False):
    unit: Unit
    scaling: ScalingConfig


ML_REQUIRED_VARS_REPO = {
    # Surface pressure FNL level 0
    Variable.PRES: VariableConfig(
        unit=Unit.PASCALS,
        scaling=ScalingConfig(
            type=ScalingType.GLOBAL,
            min=98000,
            max=121590,
        ),
    ),
    # Geopotential height FNL level 0
    Variable.GHT: VariableConfig(
        unit=Unit.METERS,
        scaling=ScalingConfig(
            type=ScalingType.GLOBAL,
            min=0,
            max=6000,
        ),
    ),
    # Relative humidity FNL level 0
    Variable.RH: VariableConfig(
        unit=Unit.PERCENTAGE,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Temperature FNL level 0
    Variable.TT: VariableConfig(
        unit=Unit.KELVIN,
        scaling=ScalingConfig(
            type=ScalingType.GLOBAL,
            min=263.15,
            max=333.15,
        ),
    ),
    # LANDUSEF is a percentage of each LU_INDEX category (61)
    Variable.LANDUSEF: VariableConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # LU_INDEX is 61 cat. LCZ data 1 feature
    Variable.LU_INDEX: VariableConfig(
        unit=Unit.NONE,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Monthly Climatology MODIS surface albedo
    Variable.ALBEDO12M: VariableConfig(
        unit=Unit.PERCENTAGE,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Monthly Climatology MODIS green fraction (MODIS FPAR)
    Variable.GREENFRAC: VariableConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # GMTED2010 30-arc-second topography height
    Variable.HGT_M: VariableConfig(
        unit=Unit.METERS,
        scaling=ScalingConfig(
            type=ScalingType.GLOBAL,
            min=0,
            max=5100,
        ),
    ),
    # [Derived] FNL level 0 (~10m) Wind Speed from UU and VV
    Variable.WSPD: VariableConfig(
        unit=Unit.METERSPERSEC,
        scaling=ScalingConfig(
            type=ScalingType.GLOBAL,
            min=0,
            max=100,
        ),
    ),
    # [Derived] FNL level 0 (~10m) Wind Direction (Cyclic feature) from UU and VV
    #  Sine Component of WDIR10
    Variable.WDIR_SIN: VariableConfig(
        unit=Unit.NONE,
        scaling=ScalingConfig(
            type=ScalingType.GLOBAL,
            min=-1,
            max=1,
        ),
    ),
    # [Derived] FNL level 0 (~10m) Wind Direction (Cyclic feature) from UU and VV
    # Cosine Component of WDIR10
    Variable.WDIR_COS: VariableConfig(
        unit=Unit.NONE,
        scaling=ScalingConfig(
            type=ScalingType.GLOBAL,
            min=-1,
            max=1,
        ),
    ),
    # Monthly Climatology MODIS Leaf Area Index
    Variable.LAI12M: VariableConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.GLOBAL,
            min=0,
            max=10,
        ),
    ),
    # Soil Temp layer 0-10cm below ground (WPS Initial Condition)
    # THIS IS FOR SUMMER!!! (-10C to 60C)
    Variable.ST000010: VariableConfig(
        unit=Unit.KELVIN,
        scaling=ScalingConfig(
            type=ScalingType.GLOBAL,
            min=263.15,
            max=333.15,
        ),
    ),
    # Soil Moisture layer 0-10cm below ground (WPS Initial Condition)
    Variable.SM000010: VariableConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP1 mean building height
    Variable.BUILD_HEIGHT: VariableConfig(
        unit=Unit.METERS,
        scaling=ScalingConfig(
            type=ScalingType.GLOBAL,
            min=0,
            max=150,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 0-5m frequency bin
    Variable.HGT_DIST_5m: VariableConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 5-10m frequency bin
    Variable.HGT_DIST_10m: VariableConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 10-15m frequency bin
    Variable.HGT_DIST_15m: VariableConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 15-20m frequency bin
    Variable.HGT_DIST_20m: VariableConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 20-25m frequency bin
    Variable.HGT_DIST_25m: VariableConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 25-30m frequency bin
    Variable.HGT_DIST_30m: VariableConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 30-35m frequency bin
    Variable.HGT_DIST_35m: VariableConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 35-40m frequency bin
    Variable.HGT_DIST_40m: VariableConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 40-45m frequency bin
    Variable.HGT_DIST_45m: VariableConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 45-50m frequency bin
    Variable.HGT_DIST_50m: VariableConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 50-55m frequency bin
    Variable.HGT_DIST_55m: VariableConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 55-60m frequency bin
    Variable.HGT_DIST_60m: VariableConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 60-65m frequency bin
    Variable.HGT_DIST_65m: VariableConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 65-70m frequency bin
    Variable.HGT_DIST_70m: VariableConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP2 Distribution of building heights 70-75m frequency bin
    Variable.HGT_DIST_75m: VariableConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP3 Area weighted mean building height
    Variable.AW_BUILD_HEIGHT: VariableConfig(
        unit=Unit.METERS,
        scaling=ScalingConfig(
            type=ScalingType.GLOBAL,
            min=0,
            max=250,
        ),
    ),
    # Custom UCPs for cities.
    # UCP4 Standard deviation of building height
    Variable.STDH_URB2D: VariableConfig(
        unit=Unit.METERS,
        scaling=ScalingConfig(
            type=ScalingType.GLOBAL,
            min=0,
            max=200,
        ),
    ),
    # Custom UCPs for cities.
    # UCP5 Plan area fraction
    Variable.BUILDING_AREA_FRACTION: VariableConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # Custom UCPs for cities.
    # UCP6 Building surface to plan area ratio. Update for Very Dense Cities
    Variable.BUILD_SURF_RATIO: VariableConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
            min=0,
            max=200,
        ),
    ),
    # Custom UCPs for cities.
    # UCP7 Urban fraction
    Variable.FRC_URB2D: VariableConfig(
        unit=Unit.FRACTION,
        scaling=ScalingConfig(
            type=ScalingType.NONE,
        ),
    ),
    # [Derived] Solar Time from UTC (Cyclic feature)
    # in Minutes of Day (MIN)
    # Sine Component of Solar Time
    Variable.SOLAR_TIME_SIN: VariableConfig(
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
    Variable.SOLAR_TIME_COS: VariableConfig(
        unit=Unit.NONE,
        scaling=ScalingConfig(
            type=ScalingType.GLOBAL,
            min=-1,
            max=1,
        ),
    ),
}
