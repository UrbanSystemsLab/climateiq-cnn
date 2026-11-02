"""Builds a PMTiles archive with one vector layer per GeoJSON file.

The build runs in its own Python process (see start), because pyogrio bundles
a newer GDAL than fiona and rasterio:

    python -m usl_h3_tools.pmtiles_build OUT.pmtiles --name NAME \
        --layer LAYER:MINZOOM:MAXZOOM:PATH [--layer ...]
"""

import argparse
import contextlib
import dataclasses
import gzip
import heapq
import itertools
import logging
import operator
import os
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Sequence

from pmtiles import reader as pmtiles_reader
from pmtiles import tile as pmtiles_tile
from pmtiles import writer as pmtiles_writer

# Maximum compressed size of one layer's tile.
MAX_TILE_BYTES = 2_500_000

_PER_PART_METADATA = ("minzoom", "maxzoom", "bounds", "center")


@dataclasses.dataclass(frozen=True)
class LayerSource:
    name: str
    minzoom: int
    maxzoom: int
    path: str  # GeoJSON file holding the layer's features.

    def to_arg(self) -> str:
        """Returns the layer as a --layer argument."""
        return f"{self.name}:{self.minzoom}:{self.maxzoom}:{self.path}"

    @classmethod
    def from_arg(cls, arg: str) -> "LayerSource":
        """Parses a --layer argument."""
        name, minzoom, maxzoom, path = arg.split(":", 3)
        return cls(name, int(minzoom), int(maxzoom), path)


def start(
    out_path: str, name: str, description: str, layers: Sequence[LayerSource]
) -> subprocess.Popen:
    """Starts building out_path in a new Python process.

    The process exits non-zero on failure, and writes no file if no layer has
    features.
    """
    command = [
        sys.executable,
        "-m",
        __name__,
        out_path,
        "--name",
        name,
        "--description",
        description,
    ]
    for layer in layers:
        command += ["--layer", layer.to_arg()]
    package_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.Popen(command, cwd=package_parent)


def build(
    out_path: str, name: str, description: str, layers: Sequence[LayerSource]
) -> bool:
    """Builds out_path from the layers' GeoJSON files.

    Layers whose file is missing or empty are left out.

    Returns:
        False if no layer has features, in which case nothing is written.
    """
    out_dir = os.path.dirname(os.path.abspath(out_path))
    with tempfile.TemporaryDirectory(dir=out_dir) as temp_dir:
        parts = []
        for layer in layers:
            part = os.path.join(temp_dir, f"{layer.name}.pmtiles")
            if _write_layer(layer, part):
                parts.append(part)
        if not parts:
            logging.info("No features for %s", name)
            return False

        temp_path = os.path.join(temp_dir, "merged.pmtiles")
        merge(parts, temp_path, name, description)
        os.replace(temp_path, out_path)
    logging.info("Wrote %s with %d layers", out_path, len(parts))
    return True


def _write_layer(layer: LayerSource, out_path: str) -> bool:
    # Loaded only in the build process.
    import pyogrio
    from pyogrio import raw

    if not os.path.exists(layer.path):
        logging.info("Skipping layer %s: %s not found", layer.name, layer.path)
        return False
    if pyogrio.read_info(layer.path)["features"] == 0:
        logging.info("Skipping layer %s: %s has no features", layer.name, layer.path)
        return False

    with raw.open_arrow(layer.path) as (meta, stream):
        raw.write_arrow(
            stream,
            out_path,
            layer=layer.name,
            driver="PMTiles",
            geometry_name=meta["geometry_name"] or "wkb_geometry",
            geometry_type=meta["geometry_type"],
            crs=meta["crs"],
            dataset_options={
                "MINZOOM": layer.minzoom,
                "MAXZOOM": layer.maxzoom,
                "MAX_SIZE": MAX_TILE_BYTES,
            },
        )
    logging.info("Wrote layer %s from %s", layer.name, layer.path)
    return True


def merge(parts: Sequence[str], out_path: str, name: str, description: str) -> None:
    """Merges single-layer vector PMTiles archives into one archive.

    Tiles at the same z/x/y are combined into one tile holding every layer.

    Raises:
        ValueError: A part is not gzipped MVT.
    """
    with contextlib.ExitStack() as stack:
        sources = [
            pmtiles_reader.MmapSource(stack.enter_context(open(part, "rb")))
            for part in parts
        ]
        readers = [pmtiles_reader.Reader(source) for source in sources]
        headers = [reader.header() for reader in readers]
        for part, header in zip(parts, headers):
            if (
                header["tile_type"] != pmtiles_tile.TileType.MVT
                or header["tile_compression"] != pmtiles_tile.Compression.GZIP
            ):
                raise ValueError(f"{part} is not gzipped MVT")

        tiles = heapq.merge(
            *(_tiles_by_id(source) for source in sources), key=operator.itemgetter(0)
        )
        with open(out_path, "wb") as f:
            writer = pmtiles_writer.Writer(f)
            for tile_id, group in itertools.groupby(tiles, key=operator.itemgetter(0)):
                datas = [data for _, data in group]
                if len(datas) > 1:
                    datas = [gzip.compress(b"".join(map(gzip.decompress, datas)))]
                writer.write_tile(tile_id, datas[0])
            writer.finalize(
                _merged_header(headers),
                _merged_metadata(
                    [reader.metadata() for reader in readers], name, description
                ),
            )


def _tiles_by_id(source) -> Iterator[tuple[int, bytes]]:
    for (z, x, y), data in pmtiles_reader.all_tiles(source):
        yield pmtiles_tile.zxy_to_tileid(z, x, y), data


def _merged_header(headers: Sequence[dict[str, Any]]) -> dict[str, Any]:
    header = dict(headers[0])
    header["min_lon_e7"] = min(h["min_lon_e7"] for h in headers)
    header["min_lat_e7"] = min(h["min_lat_e7"] for h in headers)
    header["max_lon_e7"] = max(h["max_lon_e7"] for h in headers)
    header["max_lat_e7"] = max(h["max_lat_e7"] for h in headers)
    header["center_lon_e7"] = (header["min_lon_e7"] + header["max_lon_e7"]) // 2
    header["center_lat_e7"] = (header["min_lat_e7"] + header["max_lat_e7"]) // 2
    header["center_zoom"] = min(h["center_zoom"] for h in headers)
    return header


def _merged_metadata(
    metadatas: Sequence[dict[str, Any]], name: str, description: str
) -> dict[str, Any]:
    metadata = {
        key: value
        for key, value in metadatas[0].items()
        if key not in _PER_PART_METADATA
    }
    metadata["name"] = name
    metadata["description"] = description
    metadata["vector_layers"] = [
        layer for m in metadatas for layer in m.get("vector_layers", [])
    ]
    if all("tilestats" in m for m in metadatas):
        stats = [layer for m in metadatas for layer in m["tilestats"]["layers"]]
        metadata["tilestats"] = {"layerCount": len(stats), "layers": stats}
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_path")
    parser.add_argument("--name", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument(
        "--layer",
        action="append",
        required=True,
        type=LayerSource.from_arg,
        help="LAYER:MINZOOM:MAXZOOM:PATH",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    build(args.out_path, args.name, args.description, args.layer)


if __name__ == "__main__":
    main()
