import argparse
import io
import tarfile

from google.cloud import firestore
from google.cloud import storage

from usl_lib.readers import elevation_readers
from usl_lib.storage import cloud_storage
from usl_lib.storage import metastore


def main() -> None:
    """Breaks the input files into chunks and uploads them to GCS."""
    args = parse_args()
    db = firestore.Client()

    with open(args.elevation_file, "rb") as elevation_fd:
        header = elevation_readers.read_from_geotiff(
            elevation_fd, header_only=True
        ).header

    study_area = metastore.StudyArea(
        name=args.name,
        col_count=header.col_count,
        row_count=header.row_count,
        x_ll_corner=header.x_ll_corner,
        y_ll_corner=header.y_ll_corner,
        cell_size=header.cell_size,
        crs=header.crs.to_string() if header.crs is not None else "",
    )
    study_area.create(db)

    storage_client = storage.Client()
    map_chunk_bucket = storage_client.bucket(cloud_storage.MAP_CHUNKS_BUCKET)
    for i, chunk in enumerate(build_chunks(args.elevation_file)):
        map_chunk_bucket.blob(f"{args.name}/chunk_{i}.tar").upload_from_file(chunk)


def build_chunks(elevation_file_path: str):
    # Place-holder for the real chunking function.
    tar_fd = io.BytesIO()
    with tarfile.open(mode="w", fileobj=tar_fd) as tar:
        tar.add(elevation_file_path, arcname="elevation.tif")
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
    parser.add_argument("--elevation-file", help="Tiff file containing elevation data.")
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
