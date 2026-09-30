#!/usr/bin/env python3
"""
cli_add_admin_breadcrumbs.py

Adds parent breadcrumb fields to:
- admin_level_8 GeoJSON (parent = admin_level_6)
- admin_level_10 GeoJSON (parents = admin_level_8 + admin_level_6)
- H3 GeoJSONs in a directory (parents = admin_level_10 + (via that) admin_level_8 + admin_level_6)

Expected fields (kept simple + consistent):
- child keeps its own existing properties (incl. id / GEOID etc.)
- we ADD:
  admin_level_{P}_parent_boundary_id
  admin_level_{P}_parent_localname

Notes:
- We avoid centroid-in-geographic warnings by reprojecting to EPSG:3857 before centroids/joins.
- We do point-in-polygon joins (using centroids) for consistent 1:1 parent assignment.
"""

import argparse
import glob
import os
from typing import Optional, Tuple, List

import geopandas as gpd
import pandas as pd


# -----------------------------
# Helpers: columns + CRS
# -----------------------------

ID_CANDIDATES = ["boundary_id", "id", "GEOID", "geoid", "osm_id"]
NAME_CANDIDATES = ["localname", "NAME", "name"]


def _pick_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def resolve_id_col(df: pd.DataFrame) -> str:
    col = _pick_first_col(df, ID_CANDIDATES)
    if not col:
        raise ValueError(
            f"Could not find an ID column. Tried {ID_CANDIDATES}. Available: {list(df.columns)}"
        )
    return col


def resolve_name_col(df: pd.DataFrame) -> Optional[str]:
    # localname may not exist; NAME is common in TIGER-derived inputs
    return _pick_first_col(df, NAME_CANDIDATES)


def ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        # assume input GeoJSON is CRS84/WGS84 if missing
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def to_metric(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Web Mercator is good enough for centroid/containment at city/county scale.
    return gdf.to_crs("EPSG:3857")


def centroid_points_metric(gdf_wgs84: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    g = to_metric(gdf_wgs84).copy()
    g["__pt__"] = g.geometry.centroid
    g = g.set_geometry("__pt__")
    return g


# -----------------------------
# Core: parent assignment
# -----------------------------

def add_parent_fields_point_within(
    child: gpd.GeoDataFrame,
    parent: gpd.GeoDataFrame,
    parent_level: int,
) -> gpd.GeoDataFrame:
    """
    Assign each child a single parent via point-within-polygon:
    - child point = centroid (in EPSG:3857)
    - parent polygons in EPSG:3857
    """
    child = ensure_wgs84(child)
    parent = ensure_wgs84(parent)

    child_id = resolve_id_col(child)
    parent_id = resolve_id_col(parent)
    parent_name = resolve_name_col(parent)  # may be None

    child_pts = centroid_points_metric(child)
    parent_m = to_metric(parent)

    joined = gpd.sjoin(child_pts, parent_m, how="left", predicate="within")

    # joined includes columns from child (left) and parent (right)
    # If there are name collisions, GeoPandas will suffix with _left/_right.
    # We'll robustly read the parent columns from joined.
    def _join_col(base: str) -> str:
        if f"{base}_right" in joined.columns:
            return f"{base}_right"
        if base in joined.columns:
            # sometimes no collision -> no suffix
            return base
        if f"{base}_left" in joined.columns:
            # should not happen for parent fields, but keep safe
            return f"{base}_left"
        raise ValueError(
            f"Expected join column for '{base}' not found. Join cols: {list(joined.columns)}"
        )

    parent_id_join = _join_col(parent_id)
    parent_name_join = _join_col(parent_name) if parent_name else None

    out = child.copy()
    pid_series = joined[parent_id_join].astype(object).where(pd.notna(joined[parent_id_join]), None)
    out[f"admin_level_{parent_level}_parent_boundary_id"] = pid_series.values

    if parent_name_join:
        pname_series = joined[parent_name_join].astype(object).where(pd.notna(joined[parent_name_join]), None)
        out[f"admin_level_{parent_level}_parent_localname"] = pname_series.values
    else:
        out[f"admin_level_{parent_level}_parent_localname"] = [None] * len(out)

    return out


def add_h3_parents_from_admin10(
    h3_gdf: gpd.GeoDataFrame,
    admin10_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    For H3 polygons:
      - join H3 centroid point within admin10 polygons
      - fall back to sjoin_nearest for cells whose centroid misses all tracts
      - propagate ALL breadcrumb levels from admin10: 2 (country), 4 (state), 6, 8, 10
    """
    h3_gdf = ensure_wgs84(h3_gdf)
    admin10_gdf = ensure_wgs84(admin10_gdf)

    a10_id = resolve_id_col(admin10_gdf)
    a10_name = resolve_name_col(admin10_gdf)

    h3_pts = centroid_points_metric(h3_gdf)
    a10_m = to_metric(admin10_gdf)

    joined = gpd.sjoin(h3_pts, a10_m, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]

    unmatched_mask = joined["index_right"].isna()
    if unmatched_mask.any():
        unmatched_idx = joined.index[unmatched_mask]
        nearest = gpd.sjoin_nearest(h3_pts.loc[unmatched_idx], a10_m, how="left")
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        nearest = nearest.reindex(unmatched_idx)
        for col in nearest.columns:
            if col in joined.columns:
                joined.loc[unmatched_idx, col] = nearest[col]

    joined = joined.reindex(h3_gdf.index)

    def _join_col(base: str) -> Optional[str]:
        if base is None:
            return None
        if f"{base}_right" in joined.columns:
            return f"{base}_right"
        if base in joined.columns:
            return base
        return None

    a10_id_join = _join_col(a10_id)
    a10_name_join = _join_col(a10_name)

    out = h3_gdf.copy()

    def _safe_values(col_name):
        if col_name and col_name in joined.columns:
            s = joined[col_name].astype(object).where(pd.notna(joined[col_name]), None)
            return s.values
        return [None] * len(out)

    out["admin_level_10_parent_boundary_id"] = _safe_values(a10_id_join)
    out["admin_level_10_parent_localname"] = _safe_values(a10_name_join)

    for lvl in (8, 6, 4, 2):
        pid = f"admin_level_{lvl}_parent_boundary_id"
        pname = f"admin_level_{lvl}_parent_localname"

        pid_join = _join_col(pid) or f"{pid}_right"
        pname_join = _join_col(pname) or f"{pname}_right"

        out[pid] = _safe_values(pid_join)
        out[pname] = _safe_values(pname_join)

    return out


# -----------------------------
# IO / CLI
# -----------------------------

def load_geojson(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    return ensure_wgs84(gdf)


def write_geojson(gdf: gpd.GeoDataFrame, path: str) -> None:
    # Keep output clean + consistent
    gdf = ensure_wgs84(gdf)
    gdf.to_file(path, driver="GeoJSON")


def add_breadcrumbs(admin6_path: str, admin8_path: str, admin10_path: str, h3_dir: Optional[str]) -> None:
    print("[breadcrumbs] Loading admin GeoJSONs…")
    g6 = load_geojson(admin6_path)
    g8 = load_geojson(admin8_path)
    g10 = load_geojson(admin10_path)

    print("[breadcrumbs] Adding admin_level_6 parents to admin_level_8…")
    g8 = add_parent_fields_point_within(g8, g6, parent_level=6)

    print("[breadcrumbs] Adding admin_level_8 + admin_level_6 parents to admin_level_10…")
    g10 = add_parent_fields_point_within(g10, g8, parent_level=8)
    # Add admin6 as well (directly from g6)
    g10 = add_parent_fields_point_within(g10, g6, parent_level=6)

    print("[breadcrumbs] Writing updated admin8/admin10…")
    write_geojson(g8, admin8_path)
    write_geojson(g10, admin10_path)

    if not h3_dir:
        print("[breadcrumbs] No --h3_dir provided; done.")
        return

    h3_files = sorted(glob.glob(os.path.join(h3_dir, "*.geojson")))
    print(f"[breadcrumbs] Found {len(h3_files)} H3 GeoJSONs in {h3_dir}")

    for fp in h3_files:
        print(f"[breadcrumbs] Updating {os.path.basename(fp)} …")
        h = load_geojson(fp)
        h = add_h3_parents_from_admin10(h, g10)
        write_geojson(h, fp)

    print("[breadcrumbs] Done.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--admin6", required=True, help="Path to admin level 6 GeoJSON")
    p.add_argument("--admin8", required=True, help="Path to admin level 8 GeoJSON")
    p.add_argument("--admin10", required=True, help="Path to admin level 10 GeoJSON")
    p.add_argument("--h3_dir", required=False, default=None, help="Directory containing H3 GeoJSONs (admin_level_109..113...)")
    args = p.parse_args()

    add_breadcrumbs(args.admin6, args.admin8, args.admin10, args.h3_dir)


if __name__ == "__main__":
    main()
