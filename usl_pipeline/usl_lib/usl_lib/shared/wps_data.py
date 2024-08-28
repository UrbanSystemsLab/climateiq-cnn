"""List of required WPS variables that will be used ML modeling.

NOTE: Existing variables in the list should not be re-ordered.
New variables should be added to the end of the list.
"""

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


ML_REQUIRED_VARS_REPO = dict(
    {
        # Surface pressure FNL level 0
        "PRES": {
            "unit": Unit.PASCALS,
            "scaling": {
                "type": ScalingType.GLOBAL,
                "min": 98000,
                "max": 121590,
            },
        },
        # Geopotential height FNL level 0
        "GHT": {
            "unit": Unit.METERS,
            "scaling": {
                "type": ScalingType.GLOBAL,
                "min": 0,
                "max": 6000,
            },
        },
        # Relative humidity FNL level 0
        "RH": {
            "unit": Unit.PERCENTAGE,
            "scaling": {
                "type": ScalingType.NONE,
            },
        },
        # Temperature FNL level 0
        "TT": {
            "unit": Unit.KELVIN,
            "scaling": {
                "type": ScalingType.GLOBAL,
                "min": 263.15,
                "max": 333.15,
            },
        },
        # LANDUSEF is a percentage of each LU_INDEX category (61)
        "LANDUSEF": {
            "unit": Unit.FRACTION,
            "scaling": {
                "type": ScalingType.NONE,
            },
        },
        # Monthly Climatology MODIS surface albedo
        "ALBEDO12M": {
            "unit": Unit.PERCENTAGE,
            "scaling": {
                "type": ScalingType.NONE,
            },
        },
        # Monthly Climatology MODIS green fraction (MODIS FPAR)
        "GREENFRAC": {
            "unit": Unit.FRACTION,
            "scaling": {
                "type": ScalingType.NONE,
            },
        },
        # GMTED2010 30-arc-second topography height
        "HGT_M": {
            "unit": Unit.METERS,
            "scaling": {
                "type": ScalingType.GLOBAL,
                "min": 0,
                "max": 5100,
            },
        },
        # [Derived] FNL level 0 (~10m) Wind Speed from UU and VV
        "WSPD": {
            "unit": Unit.METERSPERSEC,
            "scaling": {
                "type": ScalingType.GLOBAL,
                "min": 0,
                "max": 100,
            },
        },
        # [Derived] FNL level 0 (~10m) Wind Direction (Cyclic feature) from UU and VV
        #  Sine Component of WDIR10
        "WDIR_SIN": {
            "unit": Unit.NONE,
            "scaling": {
                "type": ScalingType.GLOBAL,
                "min": -1,
                "max": 1,
            },
        },
        # [Derived] FNL level 0 (~10m) Wind Direction (Cyclic feature) from UU and VV
        # Cosine Component of WDIR10
        "WDIR_COS": {
            "unit": Unit.NONE,
            "scaling": {
                "type": ScalingType.GLOBAL,
                "min": -1,
                "max": 1,
            },
        },
        # Monthly Climatology MODIS Leaf Area Index
        "LAI12M": {
            "unit": Unit.FRACTION,
            "scaling": {
                "type": ScalingType.GLOBAL,
                "min": 0,
                "max": 10,
            },
        },
        # Soil Temp layer 0-10cm below ground (WPS Initial Condition)
        # THIS IS FOR SUMMER!!! (-10C to 60C)
        "ST000010": {
            "unit": Unit.KELVIN,
            "scaling": {
                "type": ScalingType.GLOBAL,
                "min": 263.15,
                "max": 333.15,
            },
        },
        # Soil Moisture layer 0-10cm below ground (WPS Initial Condition)
        "SM000010": {
            "unit": Unit.FRACTION,
            "scaling": {
                "type": ScalingType.NONE,
            },
        },
        # Custom UCPs for cities.
        # UCP1 mean building height
        "BUILD_HEIGHT": {
            "unit": Unit.METERS,
            "scaling": {
                "type": ScalingType.GLOBAL,
                "min": 0,
                "max": 150,
            },
        },
        # Custom UCPs for cities.
        # UCP2 Distribution of building heights 0-5m frequency bin
        "HGT_DIST_5m": {
            "unit": Unit.FRACTION,
            "scaling": {
                "type": ScalingType.NONE,
            },
        },
        # Custom UCPs for cities.
        # UCP2 Distribution of building heights 5-10m frequency bin
        "HGT_DIST_10m": {
            "unit": Unit.FRACTION,
            "scaling": {
                "type": ScalingType.NONE,
            },
        },
        # Custom UCPs for cities.
        # UCP2 Distribution of building heights 10-15m frequency bin
        "HGT_DIST_15m": {
            "unit": Unit.FRACTION,
            "scaling": {
                "type": ScalingType.NONE,
            },
        },
        # Custom UCPs for cities.
        # UCP2 Distribution of building heights 15-20m frequency bin
        "HGT_DIST_20m": {
            "unit": Unit.FRACTION,
            "scaling": {
                "type": ScalingType.NONE,
            },
        },
        # Custom UCPs for cities.
        # UCP2 Distribution of building heights 20-25m frequency bin
        "HGT_DIST_25m": {
            "unit": Unit.FRACTION,
            "scaling": {
                "type": ScalingType.NONE,
            },
        },
        # Custom UCPs for cities.
        # UCP2 Distribution of building heights 25-30m frequency bin
        "HGT_DIST_30m": {
            "unit": Unit.FRACTION,
            "scaling": {
                "type": ScalingType.NONE,
            },
        },
        # Custom UCPs for cities.
        # UCP2 Distribution of building heights 30-35m frequency bin
        "HGT_DIST_35m": {
            "unit": Unit.FRACTION,
            "scaling": {
                "type": ScalingType.NONE,
            },
        },
        # Custom UCPs for cities.
        # UCP2 Distribution of building heights 35-40m frequency bin
        "HGT_DIST_40m": {
            "unit": Unit.FRACTION,
            "scaling": {
                "type": ScalingType.NONE,
            },
        },
        # Custom UCPs for cities.
        # UCP2 Distribution of building heights 40-45m frequency bin
        "HGT_DIST_45m": {
            "unit": Unit.FRACTION,
            "scaling": {
                "type": ScalingType.NONE,
            },
        },
        # Custom UCPs for cities.
        # UCP2 Distribution of building heights 45-50m frequency bin
        "HGT_DIST_50m": {
            "unit": Unit.FRACTION,
            "scaling": {
                "type": ScalingType.NONE,
            },
        },
        # Custom UCPs for cities.
        # UCP2 Distribution of building heights 50-55m frequency bin
        "HGT_DIST_55m": {
            "unit": Unit.FRACTION,
            "scaling": {
                "type": ScalingType.NONE,
            },
        },
        # Custom UCPs for cities.
        # UCP2 Distribution of building heights 55-60m frequency bin
        "HGT_DIST_60m": {
            "unit": Unit.FRACTION,
            "scaling": {
                "type": ScalingType.NONE,
            },
        },
        # Custom UCPs for cities.
        # UCP2 Distribution of building heights 60-65m frequency bin
        "HGT_DIST_65m": {
            "unit": Unit.FRACTION,
            "scaling": {
                "type": ScalingType.NONE,
            },
        },
        # Custom UCPs for cities.
        # UCP2 Distribution of building heights 65-70m frequency bin
        "HGT_DIST_70m": {
            "unit": Unit.FRACTION,
            "scaling": {
                "type": ScalingType.NONE,
            },
        },
        # Custom UCPs for cities.
        # UCP2 Distribution of building heights 70-75m frequency bin
        "HGT_DIST_75m": {
            "unit": Unit.FRACTION,
            "scaling": {
                "type": ScalingType.NONE,
            },
        },
        # Custom UCPs for cities.
        # UCP3 Area weighted mean building height
        "AW_BUILD_HEIGHT": {
            "unit": Unit.METERS,
            "scaling": {
                "type": ScalingType.GLOBAL,
                "min": 0,
                "max": 250,
            },
        },
        # Custom UCPs for cities.
        # UCP4 Standard deviation of building height
        "STDH_URB2D": {
            "unit": Unit.METERS,
            "scaling": {
                "type": ScalingType.GLOBAL,
                "min": 0,
                "max": 200,
            },
        },
        # Custom UCPs for cities.
        # UCP5 Plan area fraction
        "BUILDING_AREA_FRACTION": {
            "unit": Unit.FRACTION,
            "scaling": {
                "type": ScalingType.NONE,
            },
        },
        # Custom UCPs for cities.
        # UCP6 Building surface to plan area ratio. Update for Very Dense Cities
        "BUILD_SURF_RATIO": {
            "unit": Unit.FRACTION,
            "scaling": {
                "type": ScalingType.NONE,
                "min": 0,
                "max": 200,
            },
        },
        # Custom UCPs for cities.
        # UCP7 Urban fraction
        "FRC_URB2D": {
            "unit": Unit.FRACTION,
            "scaling": {
                "type": ScalingType.NONE,
            },
        },
        # LU_INDEX
        "LU_INDEX": {
            "unit": Unit.NONE,
            "scaling": {
                "type": ScalingType.NONE,
            },
        },
    }
)
