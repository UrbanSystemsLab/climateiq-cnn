import argparse
import io
import pathlib
import tarfile
import tempfile
import time

from google.cloud import firestore
from google.cloud import storage

from study_area_uploader.transformers import study_area_transformers
from usl_lib.storage import cloud_storage
from usl_lib.storage import file_names
from usl_lib.storage import metastore


def main() -> None:
    """Breaks the input files into chunks and uploads them to GCS."""
    args = parse_args()

    db = firestore.Client()
    storage_client = storage.Client()
    study_area_bucket = storage_client.bucket(cloud_storage.STUDY_AREA_BUCKET)
    study_area_chunk_bucket = storage_client.bucket(
        cloud_storage.STUDY_AREA_CHUNKS_BUCKET
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        prepared_inputs = study_area_transformers.prepare_and_upload_study_area_files(
            args.name,
            args.elevation_file,
            args.boundaries_file,
            args.building_footprint_file,
            args.green_areas_file,
            args.soil_type_file,
            args.soil_type_mask_feature_property,
            pathlib.Path(temp_dir),
            study_area_bucket,
        )

        if args.export_to_citycat:
            export_to_city_cat(
                args,
                prepared_inputs,
                storage_client.bucket(cloud_storage.FLOOD_SIMULATION_INPUT_BUCKET),
                pathlib.Path(temp_dir),
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

        for i, chunk in enumerate(build_chunks(prepared_inputs, temp_dir)):
            study_area_chunk_bucket.blob(f"{args.name}/chunk_{i}.tar").upload_from_file(
                chunk
            )


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
    prepared_inputs: study_area_transformers.PreparedInputData,
    work_dir: str,
):
    # Place-holder for the real chunking function.
    tar_fd = io.BytesIO()
    with tarfile.open(mode="w", fileobj=tar_fd) as tar:
        tar.add(
            str(prepared_inputs.elevation_file_path), arcname=file_names.ELEVATION_TIF
        )
    tar_fd.flush()
    tar_fd.seek(0)
    yield tar_fd


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
    )
    parser.add_argument(
        "--elevation-geotiff-band",
        type=int,
        default=1,
        help="Band index in GeoTIFF file containing elevation data (default value is"
        + " 1)",
    )

    return parser.parse_args()
