import argparse
import logging
import pathlib
import sys
import tempfile
import time

from google.cloud import firestore
from google.cloud import storage

from study_area_uploader.chunkers import study_area_chunkers
from study_area_uploader.transformers import study_area_transformers
from usl_lib.storage import cloud_storage
from usl_lib.storage import metastore


def main() -> None:
    """Breaks the input files into chunks and uploads them to GCS."""
    args = _parse_args()
    # Setting up logging:
    logging.getLogger("rasterio").setLevel(logging.WARNING)
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR)

    db = firestore.Client()
    storage_client = storage.Client()

    # Let's check and cleanup old files before uploading new ones.
    study_area_file_prefix = f"{args.name}/"
    overwrite = args.overwrite
    study_area_bucket = storage_client.bucket(cloud_storage.STUDY_AREA_BUCKET)
    if not _check_and_delete_storage_files_with_prefix(
        study_area_bucket, study_area_file_prefix, overwrite
    ):
        sys.exit(1)

    citycat_bucket = None
    if args.export_to_citycat:
        citycat_bucket = storage_client.bucket(
            cloud_storage.FLOOD_SIMULATION_INPUT_BUCKET
        )
        citycat_bucket_chunked = storage_client.bucket(
            cloud_storage.FLOOD_SIMULATION_INPUT_BUCKET_CHUNKED
        )
        if not _check_and_delete_storage_files_with_prefix(
            citycat_bucket, study_area_file_prefix, overwrite
        ):
            sys.exit(1)

    chunk_bucket = storage_client.bucket(cloud_storage.STUDY_AREA_CHUNKS_BUCKET)
    if not _check_and_delete_storage_files_with_prefix(
        chunk_bucket, study_area_file_prefix, overwrite
    ):
        sys.exit(1)

    if not _check_and_delete_storage_files_with_prefix(
        storage_client.bucket(cloud_storage.FEATURE_CHUNKS_BUCKET),
        study_area_file_prefix,
        overwrite,
    ):
        sys.exit(1)

    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = pathlib.Path(temp_dir)
        prepared_inputs = study_area_transformers.prepare_and_upload_study_area_files(
            args.name,
            args.elevation_file,
            args.boundaries_file,
            args.building_footprint_file,
            args.green_areas_file,
            args.soil_type_file,
            args.soil_type_mask_feature_property,
            work_dir,
            study_area_bucket,
            input_non_green_area_soil_classes=set(args.non_green_area_soil_classes),
        )

        if args.export_to_citycat:
            if citycat_bucket is None:
                raise ValueError("Internal error, CityCat bucket should be set earlier")
            study_area_transformers.prepare_and_upload_citycat_input_files(
                args.name,
                prepared_inputs,
                work_dir,
                citycat_bucket,
                elevation_geotiff_band=args.elevation_geotiff_band,
            )
            study_area_chunkers.build_and_upload_chunks_citycat(
                args.name,
                prepared_inputs,
                work_dir,
                citycat_bucket_chunked,
                1000,
                input_elevation_band=args.elevation_geotiff_band,
            )

        # Wait till study area metadata is registered by cloud_function triggered by
        # elevation file storing event in study_area_bucket.
        retries = 0
        while True:
            try:
                metastore.StudyArea.get(db, args.name)
            except ValueError:
                if retries > 2:
                    raise RuntimeError(
                        "Cloud function failed to create metastore entry on study area "
                        "upload. Check the error log for details: "
                        "https://pantheon.corp.google.com/errors"
                    )
                retries += 1
                time.sleep(5)
            else:
                break

        study_area_chunkers.build_and_upload_chunks(
            args.name,
            prepared_inputs,
            pathlib.Path(),
            chunk_bucket,
            1000,
            input_elevation_band=args.elevation_geotiff_band,
        )


def _check_and_delete_storage_files_with_prefix(
    bucket: storage.Bucket, prefix: str, overwrite: bool
) -> bool:
    """Checks if storage bucket folder has files, deletes them and returns True.

    The function returns False if files are found but overwrite is False (deletion is
    not allowed).
    """
    blobs = [blob for blob in bucket.list_blobs(prefix=f"{prefix}")]
    if len(blobs) == 0:
        # No files, nothing to check or delete
        return True
    if not overwrite:
        # Deletion is not allowed, stop the execution
        logging.error(
            f"{len(blobs)} file(s) found in gs://{bucket.name}/{prefix}, "
            "If you want to delete existing files, set --overwrite in order to clean "
            "them up before upload. Otherwise, use a different study area name."
        )
        return False
    logging.info("Deleting all files in gs://%s/%s*...", bucket.name, prefix)
    for blob in blobs:
        blob.delete()
    logging.info(" - %s files were deleted", len(blobs))
    return True


def _get_args_parser() -> argparse.ArgumentParser:
    """Prepares command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Uploads a set of files describing features of a geography into GCS. The "
            "geography is broken into smaller sub-geographic regions or 'chunks.' Each "
            "chunk is uploaded as an archive file containing the subset of the input "
            "files for that chunk."
        ),
    )
    parser.add_argument(
        "--name", help="Name to associate with the geography.", required=True
    )
    parser.add_argument(
        "--elevation-file", help="Tiff file containing elevation data.", required=True
    )
    parser.add_argument(
        "--elevation-geotiff-band",
        type=int,
        default=1,
        help=(
            "Optional band index of the layer for elevation in GeoTIFF file passed via "
            "the --elevation-file argument. Defaults to 1."
        ),
    )
    parser.add_argument(
        "--boundaries-file",
        help=(
            "Optional shape file defining a sub-area to crop to, "
            "removing data outside it."
        ),
    )
    parser.add_argument(
        "--green-areas-file",
        help="Shape file containing green area locations.",
        required=True,
    )
    parser.add_argument(
        "--non-green-area-soil-classes",
        type=int,
        default=[],
        help=(
            "Optional list of soil classes in the given green areas file that should "
            "not be treated as green areas. Areas with these soil classes will be "
            "treated as impermeable areas."
        ),
        nargs="*",
    )
    parser.add_argument(
        "--building-footprint-file",
        help="Shape file containing building footprints.",
        required=True,
    )
    parser.add_argument(
        "--soil-type-file",
        help="Shape file containing soil texture data.",
        required=True,
    )
    parser.add_argument(
        "--soil-type-mask-feature-property",
        default="soil_class",
        help=(
            "Optional name of the property in the soil type shape which provides "
            "soil classes."
        ),
    )
    parser.add_argument(
        "--export-to-citycat",
        type=bool,
        default=False,
        help=(
            "If set, additionally generates CityCAT configuration files and uploads "
            "them to the CityCAT inputs bucket. This is intended for geographies meant "
            "for training rather than prediction, as we won't be running simulations "
            "for the latter."
        ),
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help=(
            "If set, logs additional details for processing steps as the script runs. "
            "(Logs at the INFO level rather than ERROR.)"
        ),
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--overwrite",
        type=bool,
        default=False,
        help=(
            "If set, existing files for a study area of the same name will be deleted. "
            "If not set, the upload will halt to prevent deletion of data."
        ),
        action=argparse.BooleanOptionalAction,
    )
    return parser


def _parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = _get_args_parser()
    args = parser.parse_args()

    # Validation of CLI arguments
    if args.soil_type_file and not args.non_green_area_soil_classes:
        parser.error(
            "--non_green_area_soil_classes required if --soil_type_file present"
        )

    return args


if __name__ == "__main__":
    main()
