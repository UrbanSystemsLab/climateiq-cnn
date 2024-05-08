import argparse
import io
import tarfile
import time

from google.cloud import firestore
from google.cloud import storage

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

    study_area_bucket.blob(
        f"{args.name}/{file_names.ELEVATION_TIF}"
    ).upload_from_filename(args.elevation_file)
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

    for i, chunk in enumerate(build_chunks(args.elevation_file)):
        study_area_chunk_bucket.blob(f"{args.name}/chunk_{i}.tar").upload_from_file(
            chunk
        )


def build_chunks(elevation_file_path: str):
    # Place-holder for the real chunking function.
    tar_fd = io.BytesIO()
    with tarfile.open(mode="w", fileobj=tar_fd) as tar:
        tar.add(elevation_file_path, arcname=file_names.ELEVATION_TIF)
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
        "--green-areas-file", help="Shape file containing green area locations."
    )
    parser.add_argument(
        "--building-footprint-file", help="Shape file containing building footprints."
    )
    parser.add_argument(
        "--soil-type-file", help="Shape file containing soil texture data."
    )
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=1000,
        help="Length of the sub-area chunk squares to upload.",
    )

    return parser.parse_args()
