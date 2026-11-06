"""Metadata collected while building a feature matrix."""

import dataclasses
import datetime


@dataclasses.dataclass(slots=True)
class FeatureMetadata:
    """Additional information about the extracted features.

    Attributes:
        elevation_min: The lowest elevation height encountered.
        elevation_max: The highest elevation height encountered.
        slope_min: The lowest slope value encountered (percent rise).
        slope_max: The highest slope value encountered (percent rise).
    """

    elevation_min: float | None = None
    elevation_max: float | None = None
    slope_min: float | None = None
    slope_max: float | None = None
    chunk_size: int | None = None
    time: datetime.datetime | None = None
