import argparse
import logging
import pathlib
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
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR)

    db = firestore.Client()
    storage_client = storage.Client()
    study_area_bucket = storage_client.bucket(cloud_storage.STUDY_AREA_BUCKET)
    study_area_chunk_bucket = storage_client.bucket(
        cloud_storage.STUDY_AREA_CHUNKS_BUCKET
    )
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
            export_to_city_cat(
                args,
                prepared_inputs,
                storage_client.bucket(cloud_storage.FLOOD_SIMULATION_INPUT_BUCKET),
                work_dir,
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

        build_chunks(args, prepared_inputs, study_area_chunk_bucket, work_dir)


def export_to_city_cat(
    args: argparse.Namespace,
    prepared_inputs: study_area_transformers.PreparedInputData,
    flood_simulation_input_bucket: storage.Bucket,
    work_dir: pathlib.Path,
):
    study_area_transformers.prepare_and_upload_citycat_input_files(
        args.name,
        prepared_inputs,
        work_dir,
        flood_simulation_input_bucket,
        elevation_geotiff_band=args.elevation_geotiff_band,
    )


def build_chunks(
    args: argparse.Namespace,
    prepared_inputs: study_area_transformers.PreparedInputData,
    study_area_chunk_bucket: storage.Bucket,
    work_dir: pathlib.Path,
):
    study_area_chunkers.build_and_upload_chunks(
        args.name, prepared_inputs, work_dir, study_area_chunk_bucket, args.chunk_length
    )


def parse_args() -> argparse.Namespace:
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
        "--boundaries-file",
        help="Shape file defining sub-area to crop around it and clear all the data"
        + " outside it",
    )
    parser.add_argument(
        "--green-areas-file", help="Shape file containing green area locations."
    )
    parser.add_argument(
        "--building-footprint-file", help="Shape file containing building footprints."
    )
    parser.add_argument(
        "--soil-type-file", help="Shape file containing soil texture data."
    )
    parser.add_argument(
        "--soil-type-mask-feature-property",
        default="soil_class",
        help="Feature property name in the soil type shape file providing soil classes",
    )
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=1000,
        help="Length of the sub-area chunk squares to upload.",
    )
    parser.add_argument(
        "--export-to-citycat",
        type=bool,
        default=False,
        help="Indicator of the execution mode where input files should be exported"
        + " to CityCat storage bucket.",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--elevation-geotiff-band",
        type=int,
        default=1,
        help="Band index in GeoTIFF file containing elevation data (default value is"
        + " 1)",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Indicates that more details of processing steps should be printed to the"
        + "console (INFO logging level instead of default ERROR one)",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--non-green-area-soil-classes",
        type=int,
        default=[],
        help="Optional list of soil classes that cannot be treated as green areas",
        nargs="*",
    )

    args = parser.parse_args()

    # Validation of CLI arguments
    if args.soil_type_file and not args.non_green_area_soil_classes:
        parser.error(
            "--non_green_area_soil_classes required if --soil_type_file present"
        )

    return args
