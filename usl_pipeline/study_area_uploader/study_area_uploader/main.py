import argparse
from dataclasses import dataclass
import io
import pathlib
import tarfile
import tempfile
import time
from typing import Optional, Tuple

from google.cloud import firestore
from google.cloud import storage
from google.cloud.storage import bucket
from shapely import geometry

from study_area_uploader.readers import polygon_readers
from study_area_uploader.transformers import (
    elevation_transformers,
    study_area_transformers,
)
from usl_lib.readers import elevation_readers
from usl_lib.shared import geo_data
from usl_lib.storage import cloud_storage
from usl_lib.storage import file_names
from usl_lib.storage import metastore
from usl_lib.transformers import polygon_transformers
from usl_lib.writers import polygon_writers


@dataclass
class PreparedInputData:
    """Input data needed to run pipeline for flood scenarios."""

    elevation_file_path: pathlib.Path
    boundaries_polygons: Optional[list[Tuple[geometry.Polygon, int]]]
    buildings_polygons: Optional[list[Tuple[geometry.Polygon, int]]]
    green_areas_polygons: Optional[list[Tuple[geometry.Polygon, int]]]
    soil_classes_polygons: Optional[list[Tuple[geometry.Polygon, int]]]


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
        prepared_inputs = prepare_and_upload_study_area_files(
            args, pathlib.Path(temp_dir), study_area_bucket
        )

        if args.export_to_citycat:
            export_to_city_cat(
                prepared_inputs,
                storage_client.bucket(cloud_storage.FLOOD_SIMULATION_INPUT_BUCKET),
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


def prepare_and_upload_study_area_files(
    args: argparse.Namespace,
    work_dir: pathlib.Path,
    study_area_bucket: bucket.Bucket,
) -> PreparedInputData:
    """Prepares data needed to run pipeline for flood scenarios.

    Args:
        args: command-line arguments for study-area uploader script.
        work_dir: A folder that can be used for transforming files.
        study_area_bucket: Target cloud storage bucket to export study area files to.

    Returns:
        Prepared input data that can be exported to CityCat or be passed to chunker.
    """
    with open(args.elevation_file, "rb") as input_file:
        elevation_header = elevation_readers.read_from_geotiff(
            input_file, header_only=True
        ).header
    crs = elevation_header.crs.to_string()

    boundaries_polygons: Optional[list[Tuple[geometry.Polygon, int]]] = None
    sub_area_bounding_box: Optional[geo_data.BoundingBox] = None
    buildings_polygons: Optional[list[Tuple[geometry.Polygon, int]]] = None
    green_areas_polygons: Optional[list[Tuple[geometry.Polygon, int]]] = None
    soil_classes_polygons: Optional[list[Tuple[geometry.Polygon, int]]] = None
    if args.boundaries_file is None:
        elevation_file_path = args.elevation_file
    else:
        boundaries_polygons = polygon_readers.read_polygons_from_shape_file(
            args.boundaries_file, target_crs=crs
        )
        # Write boundaries to study area bucket
        with study_area_bucket.blob(f"{args.name}/{file_names.BOUNDARIES_TXT}").open(
            "w"
        ) as output_file:
            polygon_writers.write_polygons_to_text_file(
                boundaries_polygons, output_file
            )
        # Calculate bounding box rectangle for cropping sub-area
        sub_area_bounding_box = polygon_transformers.get_bounding_box_for_boundaries(
            p[0] for p in boundaries_polygons
        )
        # Crop elevation data
        elevation_file_path = str(work_dir / file_names.ELEVATION_TIF)
        elevation_transformers.crop_geotiff_to_sub_area(
            args.elevation_file, elevation_file_path, sub_area_bounding_box
        )

    # Write elevation to study area bucket
    study_area_bucket.blob(
        f"{args.name}/{file_names.ELEVATION_TIF}"
    ).upload_from_filename(elevation_file_path)

    # Read polygon files
    if args.building_footprint_file is not None:
        buildings_polygons = list(
            study_area_transformers.transform_shape_file(
                args.building_footprint_file, sub_area_bounding_box, crs
            )
        )
        # Write buildings to study area bucket
        with study_area_bucket.blob(f"{args.name}/{file_names.BUILDINGS_TXT}").open(
            "w"
        ) as output_file:
            polygon_writers.write_polygons_to_text_file(buildings_polygons, output_file)
    if args.green_areas_file is not None:
        green_areas_polygons = list(
            study_area_transformers.transform_shape_file(
                args.green_areas_file, sub_area_bounding_box, crs
            )
        )
        # Write green areas to study area bucket
        with study_area_bucket.blob(f"{args.name}/{file_names.GREEN_AREAS_TXT}").open(
            "w"
        ) as output_file:
            polygon_writers.write_polygons_to_text_file(
                green_areas_polygons, output_file
            )
        # Soil information is only used when green area data is defined.
        if args.soil_type_file is not None:
            soil_classes_polygons = list(
                study_area_transformers.transform_shape_file(
                    args.green_areas_file,
                    sub_area_bounding_box,
                    crs,
                    mask_value_feature_property=args.soil_type_mask_feature_property,
                )
            )
            # Write soil classes to study area bucket
            with study_area_bucket.blob(
                f"{args.name}/{file_names.SOIL_CLASSES_TXT}"
            ).open("w") as output_file:
                polygon_writers.write_polygons_to_text_file(
                    soil_classes_polygons, output_file, support_mask_values=True
                )

    return PreparedInputData(
        elevation_file_path,
        boundaries_polygons,
        buildings_polygons,
        green_areas_polygons,
        soil_classes_polygons,
    )


def export_to_city_cat(
    prepared_inputs: PreparedInputData,
    flood_simulation_input_bucket: bucket.Bucket,
):
    # Place-holder for exporting study area data as inputs for CityCat program
    pass


def build_chunks(
    prepared_inputs: PreparedInputData,
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

    return parser.parse_args()
