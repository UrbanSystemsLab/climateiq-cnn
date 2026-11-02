from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional, TextIO, Tuple

import geopandas as gpd
import pandas as pd
from google.api_core import exceptions
from google.cloud import storage
from shapely.geometry import mapping

from .h3_utils import detect_h3_col, h3_to_polygon, parent_series
from .stats import stats_for_group

DEFAULT_CHUNK_SIZE = 16 * 1024 * 1024
MAX_FEATURE_SIZE = 256 * 1024 * 1024
SOURCE_GENERATION_KEY = "source_generation"
CONTENT_TYPE = "application/x-ndjson"

_FEATURES_ARRAY_START = re.compile(r'"features"\s*:\s*\[')
_SEPARATORS = " \t\r\n,"
_decoder = json.JSONDecoder()


def read_admin_geojson_preserve(
    path: str, name: Optional[str] = None
) -> Tuple[dict, gpd.GeoDataFrame]:
    """Load an admin boundary file (GeoJSON or Parquet) and return.

      - template_dict: a minimal GeoJSON-like header with 'type', 'name', and 'crs'
      - gdf: GeoDataFrame of admin polygons in EPSG:4326

    If the input is parquet, we synthesize a simple FeatureCollection-like template.
    """
    if path.lower().endswith((".geojson", ".json")):
        with open(path, "r") as f:
            data = json.load(f)

        # Extract CRS if present, otherwise default to WGS84 / CRS84 style
        crs = data.get(
            "crs",
            {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
            },
        )
        fc_name = data.get("name", name or "admin_boundaries")

        gdf = gpd.GeoDataFrame.from_features(data.get("features", []), crs="EPSG:4326")

        template = {"type": "FeatureCollection", "name": fc_name, "crs": crs}
        return template, gdf

    # Parquet (e.g. TIGER/ACS boundaries from Census)
    if path.lower().endswith(".parquet"):
        gdf = gpd.read_parquet(path)
        # Best-effort to ensure WGS84; if source has no CRS we assume EPSG:4326
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)
        else:
            gdf = gdf.to_crs("EPSG:4326")

        fc_name = name or "admin_boundaries"
        crs = {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        }
        template = {"type": "FeatureCollection", "name": fc_name, "crs": crs}
        return template, gdf

    raise ValueError(f"Unsupported admin file type for path: {path}")


def filter_admin_gdf(gdf: gpd.GeoDataFrame, expr: Optional[str]) -> gpd.GeoDataFrame:
    """Apply a pandas.query filter expression to the GeoDataFrame, if provided.

    We add a small safety wrapper so that if the expression references columns that
    don't exist, we raise a clear error message listing available columns.
    """
    if not expr:
        return gdf

    try:
        out = gdf.query(expr)
    except Exception as e:
        cols = list(gdf.columns)
        raise ValueError(
            f"Admin filter produced an error.\n"
            f"Expression: {expr!r}\n"
            f"Available columns: {cols}\n"
            f"Original error: {e}"
        )

    if out.empty:
        cols = list(gdf.columns)
        raise ValueError(
            f"Admin filter produced empty result.\n"
            f"Expression: {expr!r}\n"
            f"Available columns: {cols}"
        )

    return out


def write_geojson_like(template: dict, gdf: gpd.GeoDataFrame, path: str) -> None:
    """Write a GeoDataFrame to a GeoJSON file, preserving a template header.

    The header (type/name/crs) comes from an input file.
    The resulting file has the form:

    {
      "type": "FeatureCollection",
      "name": "...",
      "crs": {...},
      "features": [...]
    }
    """
    # Convert features to GeoJSON-like dicts
    features = json.loads(gdf.to_json())["features"]

    out = dict(template)
    out["features"] = features

    with open(path, "w") as f:
        json.dump(out, f)


def make_hex_level_geojson(
    h3_csv_path: str,
    value_cols: List[str],
    res: int,
    out_path: str,
    h3_col: Optional[str] = None,
):
    df = pd.read_csv(h3_csv_path)
    h3c = detect_h3_col(df, h3_col)
    df["h3_parent"] = parent_series(df, h3c, res)
    # Build polygons per parent
    parents = df["h3_parent"].dropna().astype(str).unique().tolist()
    gdf = gpd.GeoDataFrame(
        {"h3": parents, "geometry": [h3_to_polygon(h) for h in parents]},
        geometry="geometry",
        crs="EPSG:4326",
    )
    # Aggregate stats per parent
    feats = []
    for h in parents:
        sub = df[df["h3_parent"] == h]
        props: Dict = {"h3": h}
        for col in value_cols:
            if col in sub.columns:
                props.update(stats_for_group(sub[col], col))
        feats.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": mapping(gdf.loc[gdf["h3"] == h, "geometry"].values[0]),
            }
        )
    fc = {
        "type": "FeatureCollection",
        "name": f"h3_level_{res}",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": feats,
    }

    with open(out_path, "w") as f:
        json.dump(fc, f)


def to_geojson_seq(
    src: TextIO,
    dst: TextIO,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_feature_size: int = MAX_FEATURE_SIZE,
) -> int:
    """Streams the features of a FeatureCollection out one per line.

    Expected MapBox format. Each feature's original JSON text is written unchanged
    except that whitespace newlines become spaces, so it works whether the input
    holds one feature per line or the whole collection on a single line.
    """
    buf = ""

    def fill() -> bool:
        nonlocal buf
        chunk = src.read(chunk_size)
        buf += chunk
        return bool(chunk)

    while True:
        match = _FEATURES_ARRAY_START.search(buf)
        if match is not None:
            pos = match.end()
            break
        if len(buf) > max_feature_size or not fill():
            raise ValueError('No "features" array found in the GeoJSON input.')

    count = 0
    while True:
        while pos < len(buf) and buf[pos] in _SEPARATORS:
            pos += 1
        if pos == len(buf):
            buf, pos = "", 0
            if not fill():
                raise ValueError("GeoJSON input ends before its features array.")
            continue

        if buf[pos] == "]":
            return count

        try:
            feature, end = _decoder.raw_decode(buf, pos)
        except json.JSONDecodeError:
            buf, pos = buf[pos:], 0
            if len(buf) > max_feature_size:
                raise ValueError(
                    f"Feature {count} is not valid JSON or is larger than "
                    f"{max_feature_size} characters."
                ) from None
            if not fill():
                raise ValueError(
                    f"GeoJSON input ends inside feature {count}."
                ) from None
            continue

        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise ValueError(f"Item {count} of the features array is not a Feature.")

        dst.write(buf[pos:end].replace("\r", " ").replace("\n", " "))
        dst.write("\n")
        count += 1
        pos = end
        if pos > chunk_size:
            buf, pos = buf[pos:], 0


def convert_blob(
    bucket: storage.Bucket,
    src_name: str,
    dst_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> bool:
    """Writes a line-delimited copy of a GeoJSON object in the same bucket.

    Returns False if the copy was already up to date, True if it was written.
    """
    src = bucket.blob(src_name)
    src.reload()
    source_generation = str(src.generation)

    dst = bucket.get_blob(dst_name)
    if (
        dst is not None
        and (dst.metadata or {}).get(SOURCE_GENERATION_KEY) == source_generation
    ):
        logging.info("gs://%s/%s is up to date", bucket.name, dst_name)
        return False

    dst_generation = dst.generation if dst is not None else 0
    dst = bucket.blob(dst_name)
    dst.metadata = {SOURCE_GENERATION_KEY: source_generation}
    try:
        with src.open(
            "rt", encoding="utf-8", if_generation_match=src.generation
        ) as reader, dst.open(
            "wt",
            encoding="utf-8",
            content_type=CONTENT_TYPE,
            if_generation_match=dst_generation,
        ) as writer:
            count = to_geojson_seq(reader, writer, chunk_size=chunk_size)
    except exceptions.PreconditionFailed:
        raise RuntimeError(
            f"gs://{bucket.name}/{src_name} or gs://{bucket.name}/{dst_name} "
            "changed during conversion."
        ) from None

    logging.info(
        "Wrote %d features from gs://%s/%s to gs://%s/%s",
        count,
        bucket.name,
        src_name,
        bucket.name,
        dst_name,
    )
    return True
