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

REQUIRED_VARS = {
    # Surface pressure
    "PRES": {
        "unit": Unit.PASCALS,
        "scaling": {
            "type": ScalingType.GLOBAL,
            "min": 98000,
            "max": 121590,
        }
    },
    # Geopotential height
    "GHT": {
        "unit": Unit.METERS,
        "scaling": {
            # TODO: Apply global scaling
            "type": ScalingType.NONE,
        }
    },
    # Relative humidity
    "RH": {
        "unit": Unit.PERCENTAGE,
        "scaling": {
            "type": ScalingType.NONE,
        }
    },
    # Temperature
    "TT": {
        "unit": Unit.KELVIN,
        "scaling": {
            "type": ScalingType.GLOBAL,
            "min": 263.15,
            "max": 333.15,
        }
    },
}
