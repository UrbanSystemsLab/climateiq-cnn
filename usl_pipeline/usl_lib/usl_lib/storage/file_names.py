"""Useful constants for referring to file names consistently."""

ELEVATION_TIF = "elevation.tif"
HEADER_JSON = "header.json"
BOUNDARIES_TXT = "boundaries.txt"
BUILDINGS_TXT = "buildings.txt"
GREEN_AREAS_TXT = "green_areas.txt"
SOIL_CLASSES_TXT = "soil_classes.txt"

CITYCAT_ELEVATION_ASC = "Domain_DEM.asc"
CITYCAT_BUILDINGS_TXT = "Buildings.txt"
CITYCAT_GREEN_AREAS_TXT = "GreenAreas.txt"
CITYCAT_SPATIAL_GREEN_AREAS_TXT = "Spatial_GreenAreas.txt"

# Match only Domain 3 (500m) WPS files
WPS_DOMAIN3_NC_REGEX = r"met_em\.d03.*\.nc$"
# Match only Domain 3 (500m) WRF files (wrfout files are not appended
# with .nc extension)
WRF_DOMAIN3_NC_REGEX = r"wrfout\.d03.*$"
