"""Adds a new study area.

Runs the three stages that add an area to the flood pipeline

1. Extract city-clipped DEM, buildings, soil and green areas from the H3 source
   tiles (``scripts/preprocess_h3cells.py``).
2. Upload those to the study areas bucket and chunk them into the study area
   chunks bucket (``study_area_uploader``). This is what drives the cloud
   functions that build the ML feature matrices.
3. Generate NOAA Atlas 14 design storm rainfall scenarios for the city centroid
   and upload them to the flood simulation config bucket.

All three stages run in this process and share one local scratch directory.
"""

import argparse
import logging
import os
import pathlib
import re
import sys
import tempfile
import time
from typing import Sequence, Tuple

os.environ.setdefault("BUCKET_PREFIX", "test-")

from google.cloud import firestore  # noqa: E402
from google.cloud import storage  # noqa: E402

from usl_lib.storage import cloud_storage  # noqa: E402
from usl_lib.storage import metastore  # noqa: E402

from scripts import preprocess_h3cells  # noqa: E402

# How long to wait for the cloud functions to take a study area all the way to
# RESCALING_DONE before giving up and telling the caller where to look.
DEFAULT_WAIT_TIMEOUT_SECONDS = 3600
# The urban areas shapefile uses this where a feature has no state, which is every
# feature outside the US.
_NO_STATE = "XX"
_POLL_INTERVAL_SECONDS = 30

AREA_NAME_RE = re.compile(r"^[A-Z]{3}_[A-Z]{2}_[A-Za-z0-9]+$")

# Word separators within a city name. Other punctuation is dropped, so
# "St. Louis" becomes "StLouis" and "O'Fallon" becomes "OFallon".
_CITY_WORD_SEPARATORS = re.compile(r"[\s\-‐-―_/]+")
_NON_ALPHANUMERIC = re.compile(r"[^A-Za-z0-9]")

_UNSAFE_IN_AREA_NAME = re.compile(r"[/\s-]")
_RESERVED_AREA_NAMES = re.compile(r"^\.{1,2}$|^__.*__$")

logger = logging.getLogger(__name__)


def pascal_case_city(city: str) -> str:
    words = []
    for word in _CITY_WORD_SEPARATORS.split(city.strip()):
        kept = _NON_ALPHANUMERIC.sub("", word)
        if kept:
            words.append(kept[0].upper() + kept[1:])
    if not words:
        raise ValueError(f"City name {city!r} contains no letters or digits.")

    return "".join(words)


def validate_area_name_override(area_name: str) -> str:
    if not area_name:
        raise ValueError("--area-name cannot be empty.")

    unsafe = _UNSAFE_IN_AREA_NAME.search(area_name)
    if unsafe:
        raise ValueError(
            f"--area-name {area_name!r} contains {unsafe.group()!r}. A study area "
            "name is used as a GCS object prefix and a Firestore document id, so "
            "it cannot contain '/', whitespace, or '-'."
        )

    if _RESERVED_AREA_NAMES.match(area_name):
        raise ValueError(
            f"--area-name {area_name!r} is reserved by Firestore and cannot be "
            "used as a document id."
        )

    return area_name


def canonical_area_name(country_iso: str, state: str, city: str) -> str:
    country_iso = country_iso.strip().upper()
    state = state.strip().upper()
    name = f"{country_iso}_{state}_{pascal_case_city(city)}"

    if not AREA_NAME_RE.match(name):
        raise ValueError(
            f"Derived study area name {name!r} is not valid. Expected "
            "<ISO3>_<STATE>_<City> with a three letter country code, a two "
            "letter state code and a PascalCase city of letters and digits "
            f"(pattern {AREA_NAME_RE.pattern})."
        )

    return name


def derive_centroid_lat_lon(boundary_file: pathlib.Path) -> Tuple[float, float]:
    """Returns the (latitude, longitude) centroid of a city boundary.

    Args:
      boundary_file: Path to the city boundary shapefile.

    Returns:
      A (latitude, longitude) tuple in degrees.
    """
    import geopandas

    boundary = geopandas.read_file(boundary_file)
    if boundary.empty:
        raise ValueError(f"City boundary {boundary_file} contains no features.")

    # Project to an equal-area CRS so the centroid is not skewed by latitude,
    # then convert the single point back to WGS 84 for NOAA.
    equal_area = boundary.to_crs("EPSG:6933").geometry
    # geopandas >= 1.0 exposes union_all; older versions only unary_union.
    merged = (
        equal_area.union_all()
        if hasattr(equal_area, "union_all")
        else equal_area.unary_union
    )
    centroid = merged.centroid
    as_wgs84 = (
        geopandas.GeoSeries([centroid], crs="EPSG:6933").to_crs("EPSG:4326").iloc[0]
    )
    return as_wgs84.y, as_wgs84.x


def upload_rainfall_scenarios(
    rainfall_dir: pathlib.Path, config_group: str, storage_client: storage.Client
) -> list[str]:

    bucket = storage_client.bucket(cloud_storage.FLOOD_SIMULATION_CONFIG_BUCKET)
    rainfall_files = sorted(rainfall_dir.glob("Rainfall_Data_*.txt"))

    if not rainfall_files:
        raise ValueError(
            f"No Rainfall_Data_*.txt files were generated in {rainfall_dir}"
        )

    written = []
    for rainfall_file in rainfall_files:
        blob_name = f"{config_group}/{rainfall_file.name}"
        bucket.blob(blob_name).upload_from_filename(str(rainfall_file))
        logger.info("Uploaded gs://%s/%s", bucket.name, blob_name)
        written.append(blob_name)

    return written


def wait_for_study_area(
    db: firestore.Client, area_name: str, timeout_seconds: int
) -> str:

    deadline = time.monotonic() + timeout_seconds
    last_state = None

    while True:
        try:
            study_area = metastore.StudyArea.get(db, area_name)
        except ValueError:
            state = None
        else:
            state = study_area.state

        if state != last_state:
            logger.info("Study area %s state: %s", area_name, state)
            last_state = state

        if state == metastore.StudyAreaState.RESCALING_DONE:
            return str(state)

        errors = _chunk_errors(db, area_name)
        if errors:
            raise RuntimeError(
                f"Study area {area_name} has {len(errors)} chunk(s) in error. "
                f"First error: {errors[0]}"
            )

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Study area {area_name} did not reach "
                f"{metastore.StudyAreaState.RESCALING_DONE} within {timeout_seconds}s "
                f"(last state: {state}). Check cloud function logs for errors."
            )

        time.sleep(_POLL_INTERVAL_SECONDS)


def _chunk_errors(db: firestore.Client, area_name: str) -> list[str]:
    """Returns any error messages recorded against the study area's chunks."""
    errors: list[str] = []
    try:
        refs = metastore.StudyArea.list_all_chunk_refs(db, area_name)
    except ValueError:  # The study area document does not exist yet.
        return errors

    for ref in refs:
        snapshot = ref.get()
        error = (snapshot.to_dict() or {}).get("error")
        if error:
            errors.append(f"{ref.id}: {error}")

    return errors


def _build_uploader_argv(
    area_name: str, outputs: dict, args: argparse.Namespace
) -> list:
    """Builds the command line for the study area uploader.

    Args:
      area_name: Canonical study area name.
      outputs: Preprocessing output paths, keyed as process_city_data returns them.
      args: Parsed add_area arguments supplying the upload options.

    Returns:
      An argument list accepted by study_area_uploader.main.
    """
    argv = [
        f"--name={area_name}",
        f"--elevation-file={outputs['dem']}",
        f"--building-footprint-file={outputs['buildings']}",
        f"--green-areas-file={outputs['green_spaces_polygons']}",
        f"--soil-type-file={outputs['soil']}",
        f"--boundaries-file={outputs['city_boundary']}",
        f"--elevation-geotiff-band={args.elevation_geotiff_band}",
    ]

    if args.non_green_area_soil_classes:
        argv.append("--non-green-area-soil-classes")
        argv.extend(str(value) for value in args.non_green_area_soil_classes)
    if args.overwrite:
        argv.append("--overwrite")
    if args.export_to_citycat:
        argv.append("--export-to-citycat")
    if args.verbose:
        argv.append("--verbose")

    return argv


def resolve_identity(args: argparse.Namespace) -> tuple:
    """Resolves the canonical study area name from the urban areas shapefile.

    Args:
      args: Parsed add_area arguments identifying the city. Explicit
        --country-iso and --state override the shapefile values.

    Returns:
      A tuple of (area_name, selected_city). selected_city is the matched
      feature, passed back into preprocessing to avoid re-reading the shapefile.
    """
    urban_shp = preprocess_h3cells.join_path(args.working_dir, args.urban_areas_file)
    iso, state_abbr, city, selected_city = preprocess_h3cells.resolve_area_identity(
        urban_shp,
        args.country,
        args.city,
        state=args.state,
        smod_level=args.smod_level,
        allow_multiple_features=args.allow_multiple_features,
    )

    iso = args.country_iso or iso
    state_abbr = args.state or state_abbr

    if iso is None or state_abbr is None:
        raise ValueError(
            "Could not resolve the country code and state from "
            f"{args.urban_areas_file}. Pass --country-iso and --state explicitly."
        )

    if state_abbr.upper() == _NO_STATE:
        logger.warning(
            "%s has no state in the urban areas shapefile, so the study area name "
            "will use the %s placeholder. Pass --state to set one.",
            city,
            _NO_STATE,
        )

    return canonical_area_name(iso, state_abbr, city), selected_city


def add_area(args: argparse.Namespace) -> str:
    """Runs the full flow and returns the study area name."""
    if args.area_name:
        area_name, selected_city = validate_area_name_override(args.area_name), None
    elif args.country_iso and args.state:
        # Both supplied, so the name needs no lookup.
        area_name, selected_city = (
            canonical_area_name(args.country_iso, args.state, args.city),
            None,
        )
    else:
        area_name, selected_city = resolve_identity(args)

    config_group = f"{area_name}_config"

    logger.info("Study area name: %s", area_name)
    logger.info("Rainfall config group: %s", config_group)
    logger.info(
        "Target buckets: gs://%s/%s/ and gs://%s/%s/",
        cloud_storage.STUDY_AREA_BUCKET,
        area_name,
        cloud_storage.FLOOD_SIMULATION_CONFIG_BUCKET,
        config_group,
    )

    if args.dry_run:
        logger.info("Dry run: stopping before any processing or upload.")
        return area_name

    with _work_dir(args.work_dir) as work_dir:
        # Stage 1: clip the city out of the H3 source tiles.
        outputs = preprocess_h3cells.process_city_data(
            args.country,
            args.city,
            args.target_crs,
            args.working_dir,
            args.h3_root,
            work_dir,
            state=args.state,
            area_name=area_name,
            urban_areas_file=args.urban_areas_file,
            allow_multiple_features=args.allow_multiple_features,
            smod_level=args.smod_level,
            preselected_city=selected_city,
        )

        # Stage 2: upload and chunk. This triggers the feature matrix cloud
        # functions; the uploader itself waits for the study area metadata to be
        # registered before it chunks.
        import study_area_uploader.main as uploader_main

        uploader_main.main(_build_uploader_argv(area_name, outputs, args))

        # Stage 3: rainfall scenarios for the city centroid.
        if args.skip_rainfall:
            logger.info("Skipping rainfall scenario generation.")
        else:
            latitude, longitude = derive_centroid_lat_lon(outputs["city_boundary"])
            logger.info("City centroid: lat=%.5f lon=%.5f", latitude, longitude)

            from scripts import rainfall_scenario_generator as rainfall

            rainfall_dir = work_dir / "rainfall"
            atlas14_csv = rainfall.part1_download_atlas14(
                latitude, longitude, rainfall_dir
            )
            rainfall.part2_generate_rainfall_files(atlas14_csv, rainfall_dir)
            upload_rainfall_scenarios(rainfall_dir, config_group, storage.Client())

    if args.wait:
        wait_for_study_area(firestore.Client(), area_name, args.wait_timeout)
        logger.info("Study area %s is ready.", area_name)

    return area_name


class _work_dir:
    """Yields the scratch directory, creating a temporary one when unset."""

    def __init__(self, configured: str | None):
        self._configured = configured
        self._temp_dir: tempfile.TemporaryDirectory | None = None

    def __enter__(self) -> pathlib.Path:
        if self._configured:
            path = pathlib.Path(self._configured)
            path.mkdir(parents=True, exist_ok=True)
            return path

        self._temp_dir = tempfile.TemporaryDirectory()
        return pathlib.Path(self._temp_dir.name)

    def __exit__(self, *exc_info) -> None:
        if self._temp_dir is not None:
            self._temp_dir.cleanup()


def _get_args_parser() -> argparse.ArgumentParser:
    """Prepares the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Adds a new study area to the flood pipeline."
    )

    identity = parser.add_argument_group("study area identity")
    identity.add_argument(
        "--country-iso", default=None, help="Three letter country code, e.g. USA."
    )
    identity.add_argument(
        "--state", default=None, help="State or subdivision, e.g. MO."
    )
    identity.add_argument("--city", required=True, help="City name.")
    identity.add_argument("--country", required=True, help="Country name.")
    identity.add_argument(
        "--area-name", default=None, help="Override the derived study area name."
    )

    source = parser.add_argument_group("source data")
    source.add_argument(
        "--working-dir",
        default="gs://raw-data-h3index/Working_files",
        help="Directory holding the urban areas and H3 cell shapefiles.",
    )
    source.add_argument(
        "--h3-root",
        default="gs://raw-data-h3index/CONUS_Data_H3Index",
        help="Root holding one folder per H3 index.",
    )
    source.add_argument(
        "--urban-areas-file",
        default="01_urban_areas_simplified_with_state.shp",
        help="Urban areas shapefile within --working-dir.",
    )
    source.add_argument(
        "--target-crs", default="EPSG:3395", help="CRS to reproject outputs to."
    )
    source.add_argument(
        "--smod-level", default=None, help="Filter candidates by GHSL SMOD_LEVEL."
    )
    source.add_argument(
        "--allow-multiple-features",
        action="store_true",
        help="Permit a city selection matching several features.",
    )

    upload = parser.add_argument_group("upload")
    upload.add_argument(
        "--non-green-area-soil-classes",
        type=int,
        nargs="*",
        default=[],
        help="Soil classes to treat as impermeable.",
    )
    upload.add_argument(
        "--elevation-geotiff-band", type=int, default=1, help="Elevation band index."
    )
    upload.add_argument(
        "--export-to-citycat",
        action="store_true",
        help="Also generate CityCAT simulation inputs.",
    )
    upload.add_argument(
        "--overwrite", action="store_true", help="Delete existing objects first."
    )

    run_options = parser.add_argument_group("run options")
    run_options.add_argument(
        "--work-dir", default=None, help="Scratch directory for intermediate files."
    )
    run_options.add_argument(
        "--skip-rainfall", action="store_true", help="Skip rainfall scenarios."
    )
    run_options.add_argument(
        "--wait", action="store_true", help="Wait for processing to finish."
    )
    run_options.add_argument(
        "--wait-timeout",
        type=int,
        default=DEFAULT_WAIT_TIMEOUT_SECONDS,
        help="Seconds to wait when --wait is set. (default: %(default)s)",
    )
    run_options.add_argument(
        "--dry-run", action="store_true", help="Print the derived name, then stop."
    )
    run_options.add_argument("--verbose", action="store_true", help="Log progress.")

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point. Run as ``python usl_pipeline/add_area/main.py``."""
    args = _get_args_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    # The orchestrator's own progress should always be visible.
    logger.setLevel(logging.INFO)

    try:
        add_area(args)
    except Exception as error:
        logger.error("%s", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
