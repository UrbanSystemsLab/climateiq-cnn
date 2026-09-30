from __future__ import annotations

import json
from typing import Optional, Tuple

import geopandas as gpd


def read_admin_geojson_preserve(
    path: str, name: Optional[str] = None
) -> Tuple[dict, gpd.GeoDataFrame]:
    """
    Load an admin boundary file (GeoJSON or Parquet) and return:

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

        gdf = gpd.GeoDataFrame.from_features(
            data.get("features", []), crs="EPSG:4326"
        )

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
    """
    Apply a pandas.query filter expression to the GeoDataFrame, if provided.

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
    """
    Write a GeoDataFrame to a GeoJSON file, preserving a template header
    (type/name/crs) from an input file.

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
