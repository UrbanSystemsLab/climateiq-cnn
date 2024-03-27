import argparse


def main() -> None:
    """Breaks the input files into chunks and uploads them to GCS."""
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
        "--green-areas-file",
        help="Shape file containing green area locations.",
        required=True,
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

    args = parser.parse_args()
    print(args)
