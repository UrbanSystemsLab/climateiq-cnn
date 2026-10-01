from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import h3

from .util_io import (
    read_admin_geojson_preserve,
    filter_admin_gdf,
    write_geojson_like,
)

# ---------------------------------------------------------------------
# H3 helpers
# ---------------------------------------------------------------------


def h3_to_polygon(h: str) -> Polygon:
    """
    Return a Shapely Polygon for an H3 cell, with GeoJSON coord order [lon, lat].

    Supports both:
      - h3-py v4: h3.cell_to_boundary(h) -> [(lat, lon), ...]
      - h3-py v3: h3.h3_to_geo_boundary(h) -> [(lat, lon), ...]
    """
    # v4 style
    if hasattr(h3, "cell_to_boundary"):
        boundary = h3.cell_to_boundary(h)
    else:
        # v3 style
        boundary = h3.h3_to_geo_boundary(h)

    # Convert (lat, lon) -> (lon, lat) for GeoJSON / Shapely
    ring = [(lon, lat) for (lat, lon) in boundary]
    return Polygon(ring)


def h3_resolution(h: str) -> int:
    """
    Return the H3 resolution of a cell, compatible with h3 v3 and v4.
    Used by cli_h3_csv_to_hexjson.py.
    """
    if hasattr(h3, "get_resolution"):
        # h3-py v4
        return h3.get_resolution(h)
    if hasattr(h3, "h3_get_resolution"):
        # h3-py v3
        return h3.h3_get_resolution(h)
    raise AttributeError("Could not find an H3 resolution function on 'h3' module")


def df_with_geom_from_h3(csv_path: str, h3_col: str) -> gpd.GeoDataFrame:
    """
    Read a CSV with an H3 column and return a GeoDataFrame with polygons.
    """
    print(f"[admin_enrich] Loading H3 CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if h3_col not in df.columns:
        raise KeyError(
            f"H3 column {h3_col!r} not found in {csv_path}. "
            f"Columns: {list(df.columns)}"
        )

    print(f"[admin_enrich] Converting H3 cells in '{h3_col}' to polygons …")
    geom = df[h3_col].apply(h3_to_polygon)
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geom, crs="EPSG:4326")
    return gdf


# ---------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------


def _numeric_value_cols(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    """
    Auto-detect numeric columns to aggregate, excluding any in 'exclude'.
    """
    return [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


def _summarize(series: pd.Series) -> dict:
    """
    Compute summary stats for a numeric series.
    """
    arr = series.to_numpy(dtype=float)
    if arr.size == 0:
        return {
            "mean": None,
            "min": None,
            "max": None,
            "p90": None,
            "count": 0,
            "share_pos": 0.0,
        }

    pos_share = float(np.mean(arr > 0.0))
    return {
        "mean": float(np.nanmean(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "p90": float(np.nanpercentile(arr, 90)),
        "count": int(np.count_nonzero(~np.isnan(arr))),
        "share_pos": pos_share,
    }


# ---------------------------------------------------------------------
# Main admin enrichment
# ---------------------------------------------------------------------


def enrich_admin(
    admin_path: str,
    h3_csv: str,
    value_cols: Optional[List[str]],
    out_path: str,
    name: str,
    h3_col: str = "h3",
    level: int = 6,
    filter_expr: Optional[str] = None,
    id_col: Optional[str] = "id",
    geoid_col: Optional[str] = None,
    clip_bbox: Optional[Tuple[float, float, float, float]] = None,
    h3_resolution: Optional[int] = None,
    water_polygon_path: Optional[str] = None,
    filter_threshold: float = 0.001,
) -> str:
    """
    Spatially aggregate H3 CSV metrics into admin polygons and write a GeoJSON-like file.

    - admin_path:    path to admin boundaries (GeoJSON or parquet).
    - h3_csv:        path to H3 metrics CSV.
    - value_cols:    list of metric columns in the H3 CSV (e.g. ['flood_depth_1000y']).
                     If None/empty, numeric columns will be auto-detected.
    - out_path:      output GeoJSON path.
    - name:          FeatureCollection name (e.g. 'admin_level_6_climateiq_v1').
    - h3_col:        column in CSV holding H3 cell IDs (e.g. 'cell_code').
    - level:         admin_level to stamp onto the features (6/8/10/etc).
    - filter_expr:   pandas query string to subset the admin GeoDataFrame.
    - id_col:        preferred ID column name (default 'id').
    - geoid_col:     preferred GEOID column name (overrides id_col if present).
    - h3_resolution: if set, filter the H3 CSV to only this resolution before enrichment.
                     Use when the CSV contains multiple resolutions (e.g. 9-12 from
                     preprocessing) to avoid double-counting and improve performance.
    - water_polygon_path: path to a pickle file containing a Shapely geometry of coastal
                     water bodies (from JRC Global Surface Water). If provided, water
                     is subtracted from admin polygons instead of using the H3 data
                     footprint clip. Only waterfront polygons are modified; interior
                     polygons keep their exact original geometry.

    Returns:
        out_path
    """

    print("[admin_enrich] --------------------------------------------------")
    print(f"[admin_enrich] Admin path      : {admin_path}")
    print(f"[admin_enrich] H3 CSV          : {h3_csv}")
    print(f"[admin_enrich] H3 column       : {h3_col}")
    print(f"[admin_enrich] Output path     : {out_path}")
    print(f"[admin_enrich] Admin level     : {level}")
    print(f"[admin_enrich] Filter expr     : {filter_expr!r}")
    print(f"[admin_enrich] Requested value cols: {value_cols}")

    # ------------------------------------------------------------------
    # 1. Load admin geometries + template header
    # ------------------------------------------------------------------
    template_dict, admin_gdf = read_admin_geojson_preserve(admin_path, name=name)
    print(
        f"[admin_enrich] Loaded admin_gdf with {len(admin_gdf)} rows "
        f"and columns: {list(admin_gdf.columns)}"
    )

    # Apply user filter (e.g. single county / city / tract subset)
    admin_gdf = filter_admin_gdf(admin_gdf, filter_expr)
    print(
        f"[admin_enrich] After filter, {len(admin_gdf)} admin rows remain. "
        f"Index: {admin_gdf.index.tolist()}"
    )

    if len(admin_gdf) == 0:
        raise ValueError(
            f"[admin_enrich] No admin features selected after filter {filter_expr!r}"
        )

    # ------------------------------------------------------------------
    # 1b. Remove Census water-only tracts (level 10)
    # ------------------------------------------------------------------
    # Census tracts with tract codes starting with "99" (990xxx pattern) are
    # water-only areas. They survive the water clip safety threshold and must
    # be explicitly removed.
    # Early GEOID column detection for water tract filter
    _geoid_col_for_water = None
    if level == 10:
        _candidates = []
        if geoid_col:
            _candidates.append(geoid_col)
        if id_col:
            _candidates.append(id_col)
        _candidates.extend(["GEOID", "geoid", "GEOID10", "geoid10"])
        for _c in _candidates:
            if _c in admin_gdf.columns:
                _geoid_col_for_water = _c
                break
    if level == 10 and _geoid_col_for_water is not None:
        before_water_tract = len(admin_gdf)
        water_tract_mask = admin_gdf[_geoid_col_for_water].astype(str).str[-6:].str.startswith("99")
        if water_tract_mask.any():
            admin_gdf = admin_gdf[~water_tract_mask].reset_index(drop=True)
            print(
                f"[admin_enrich] Removed {before_water_tract - len(admin_gdf)} "
                f"Census water tracts (990xxx)"
            )

    # Clip admin polygon geometries to the tiff prediction extent.
    # This ensures output polygons reflect only the area covered by the ML model,
    # not the full census boundary which may extend well beyond the prediction area.
    # NOTE: The actual H3 data footprint clip happens after H3 cells are loaded
    #       (see "Clip to H3 data footprint" section below). clip_bbox is kept as
    #       a fast pre-filter to reduce the number of admin polygons before the
    #       expensive spatial join.
    if clip_bbox is not None:
        from shapely.geometry import box as shapely_box
        minx, miny, maxx, maxy = clip_bbox
        bbox_poly = shapely_box(minx, miny, maxx, maxy)
        admin_gdf = admin_gdf.copy()
        admin_gdf["geometry"] = admin_gdf.geometry.intersection(bbox_poly)
        # Drop any polygons that ended up empty after clipping (outside tiff extent)
        admin_gdf = admin_gdf[~admin_gdf.geometry.is_empty].reset_index(drop=True)
        print(f"[admin_enrich] Pre-filtered to tiff bbox {clip_bbox} -> {len(admin_gdf)} admin polygons remain")

        # ------------------------------------------------------------------
        # Remove detached polygon parts for level 6 (counties)
        # ------------------------------------------------------------------
        # Bbox clipping can strand polygon parts far from the main body (e.g.
        # Cook County extending across Lake Michigan). Remove any parts whose
        # centroid is >0.3° from the largest part's centroid.
        if level == 6:
            from shapely.geometry import MultiPolygon as _MP
            detached_removed = 0
            new_geoms = []
            for geom in admin_gdf.geometry:
                if geom.geom_type == "MultiPolygon" and len(geom.geoms) > 1:
                    parts = sorted(geom.geoms, key=lambda p: p.area, reverse=True)
                    main_centroid = parts[0].centroid
                    kept = [parts[0]]
                    for part in parts[1:]:
                        dx = abs(part.centroid.x - main_centroid.x)
                        dy = abs(part.centroid.y - main_centroid.y)
                        if dx <= 0.3 and dy <= 0.3:
                            kept.append(part)
                        else:
                            detached_removed += 1
                    new_geoms.append(_MP(kept) if len(kept) > 1 else kept[0])
                else:
                    new_geoms.append(geom)
            admin_gdf["geometry"] = new_geoms
            if detached_removed > 0:
                print(f"[admin_enrich] Removed {detached_removed} detached polygon parts (level 6)")

        # ------------------------------------------------------------------
        # Fill interior holes for level 8 (places)
        # ------------------------------------------------------------------
        # Census place polygons can have interior holes (enclaves, military
        # bases, unincorporated areas). Fill them so the polygon is solid.
        if level == 8:
            from shapely.geometry import Polygon as _Poly, MultiPolygon as _MPoly
            holes_filled = 0
            new_geoms = []
            for geom in admin_gdf.geometry:
                if geom.geom_type == "Polygon" and list(geom.interiors):
                    new_geoms.append(_Poly(geom.exterior))
                    holes_filled += 1
                elif geom.geom_type == "MultiPolygon":
                    filled_parts = []
                    for poly in geom.geoms:
                        if list(poly.interiors):
                            filled_parts.append(_Poly(poly.exterior))
                            holes_filled += 1
                        else:
                            filled_parts.append(poly)
                    new_geoms.append(_MPoly(filled_parts))
                else:
                    new_geoms.append(geom)
            admin_gdf["geometry"] = new_geoms
            if holes_filled > 0:
                print(f"[admin_enrich] Filled {holes_filled} interior holes (level 8)")

    # Decide what to use as 'id' (per Joe/Rajan: use GEOID for the API)
    candidate_id_cols: List[str] = []

    # Highest priority: explicit geoid_col argument
    if geoid_col:
        candidate_id_cols.append(geoid_col)

    # Next: explicit id_col argument (default "id")
    if id_col:
        candidate_id_cols.append(id_col)

    # Fallbacks: common GEOID variants from Census/TIGER
    candidate_id_cols.extend(["GEOID", "geoid", "GEOID10", "geoid10"])

    chosen_id_col: Optional[str] = None
    for col in candidate_id_cols:
        if col in admin_gdf.columns:
            chosen_id_col = col
            break

    if chosen_id_col is None:
        print(
            "[admin_enrich] WARNING: No id / GEOID-like column found in admin_gdf. "
            "Features will be written without an 'id' property."
        )
    else:
        print(f"[admin_enrich] Using '{chosen_id_col}' as the 'id' field")

    # ------------------------------------------------------------------
    # 2. Load H3 CSV with geometry
    # ------------------------------------------------------------------
    h3_gdf = df_with_geom_from_h3(h3_csv, h3_col=h3_col)

    # Filter to a single H3 resolution if requested.
    # Multi-resolution CSVs (from preprocessing) contain the same area at 4 resolutions;
    # using all would quadruple-count every cell in the aggregation statistics.
    if h3_resolution is not None and "h3_res" in h3_gdf.columns:
        before = len(h3_gdf)
        h3_gdf = h3_gdf[h3_gdf["h3_res"] == h3_resolution].reset_index(drop=True)
        print(
            f"[admin_enrich] Filtered to H3 resolution {h3_resolution}: "
            f"{len(h3_gdf)} cells (from {before})"
        )

    # Determine which value columns to aggregate
    if not value_cols:
        exclude = [h3_col, h3_gdf.geometry.name]
        value_cols = _numeric_value_cols(h3_gdf.drop(columns=exclude), exclude=[])
        print(f"[admin_enrich] Auto-detected numeric value columns: {value_cols}")
    else:
        # Only use columns that actually exist in this CSV
        # (cities may have fewer scenarios than the global config lists)
        available = [c for c in value_cols if c in h3_gdf.columns]
        missing = [c for c in value_cols if c not in h3_gdf.columns]
        if missing:
            print(f"[admin_enrich] Skipping {len(missing)} missing value columns: {missing}")
        value_cols = available
        print(f"[admin_enrich] Using value columns: {value_cols}")

    # Filter out H3 cells with infinitesimally small values (sparse grid effect)
    initial_count = len(h3_gdf)
    if value_cols:
        # Check if all value columns are below threshold
        mask = pd.Series([True] * len(h3_gdf), index=h3_gdf.index)
        for col in value_cols:
            if col in h3_gdf.columns:
                # Keep if value is >= threshold OR is exactly 0 OR is NaN (missing scenario data)
                mask = mask & ((h3_gdf[col].abs() >= filter_threshold) | (h3_gdf[col] == 0) | h3_gdf[col].isna())
        h3_gdf = h3_gdf[mask]
        filtered_count = initial_count - len(h3_gdf)
        if filtered_count > 0:
            print(f"[admin_enrich] Filtered {filtered_count} H3 cells with values < {filter_threshold} (sparse grid removal)")

    # ------------------------------------------------------------------
    # 2b. Clip water bodies from admin polygons
    # ------------------------------------------------------------------
    from shapely.ops import unary_union as _unary_union
    from shapely.validation import make_valid as _make_valid

    if water_polygon_path is not None:
        import pickle as _pickle
        from shapely import set_precision as _set_precision
        from shapely.geometry import Polygon as _Polygon, MultiPolygon as _MultiPolygon

        _GRID = 0.0000001

        print(f"[admin_enrich] Loading water polygons from {water_polygon_path}…")
        with open(water_polygon_path, "rb") as _wf:
            water_geom = _pickle.load(_wf)

        water_geom = water_geom.buffer(-0.0003)
        water_geom = _make_valid(water_geom)
        water_geom = _set_precision(water_geom, _GRID)

        admin_gdf = admin_gdf.copy()
        waterfront_count = 0
        kept_original = 0
        new_geoms = []
        for _, row in admin_gdf.iterrows():
            geom = _set_precision(row.geometry, _GRID)
            geom = _make_valid(geom)

            if geom.intersects(water_geom):
                clipped = _set_precision(geom.difference(water_geom), _GRID)
                clipped = _make_valid(clipped)
                waterfront_count += 1

                if clipped.geom_type == "GeometryCollection":
                    poly_parts = [g for g in clipped.geoms
                                  if isinstance(g, (_Polygon, _MultiPolygon))]
                    clipped = _unary_union(poly_parts) if poly_parts else geom

                if clipped.is_empty or clipped.area < geom.area * 0.01:
                    clipped = geom
                    kept_original += 1

                new_geoms.append(clipped)
            else:
                new_geoms.append(geom)
        admin_gdf["geometry"] = new_geoms

        before_clip = len(admin_gdf)
        admin_gdf = admin_gdf[~admin_gdf.geometry.is_empty].reset_index(drop=True)
        print(
            f"[admin_enrich] JRC water clip -> {waterfront_count} waterfront polygons clipped, "
            f"{kept_original} kept original (too small), "
            f"{before_clip - len(admin_gdf)} removed (entirely water)"
        )

    # Filter admin polygons to those overlapping the H3 data footprint (study area).
    # Geometries are NOT clipped — original census/water-clipped shapes are preserved.
    print("[admin_enrich] Filtering to H3 data footprint (study area)…")
    h3_data_footprint = _unary_union(h3_gdf.geometry)

    before_filter = len(admin_gdf)
    admin_gdf = admin_gdf[admin_gdf.intersects(h3_data_footprint)].reset_index(drop=True)
    print(
        f"[admin_enrich] Filtered to data footprint -> "
        f"{len(admin_gdf)} admin polygons remain (removed {before_filter - len(admin_gdf)} outside study area)"
    )

    # ------------------------------------------------------------------
    # 3. Spatial join via polygon intersection
    # ------------------------------------------------------------------
    print("[admin_enrich] Performing spatial join (H3 polygons intersecting admin polygons)…")
    joined = gpd.sjoin(h3_gdf, admin_gdf, how="inner", predicate="intersects")

    matched_admin_indices = set(joined["index_right"].unique()) if not joined.empty else set()

    print(
        f"[admin_enrich] Spatial join produced {len(joined)} H3–admin matches "
        f"over {len(admin_gdf)} admin polygons "
        f"({len(matched_admin_indices)} matched, {len(admin_gdf) - len(matched_admin_indices)} with no H3 overlap)."
    )

    # ------------------------------------------------------------------
    # 4. Aggregate per admin polygon (include ALL polygons, even those without H3 data)
    # ------------------------------------------------------------------
    out_rows: List[dict] = []

    def _build_admin_rec(admin_row, idx):
        """Build base record from admin polygon attributes."""
        rec = {}
        for k in admin_gdf.columns:
            if k != admin_gdf.geometry.name:
                if k == 'NAME':
                    value = admin_row[k]
                    rec['name'] = value.title() if isinstance(value, str) else value
                else:
                    rec[k] = admin_row[k]
        rec["admin_level"] = level
        if chosen_id_col is not None:
            boundary_id = str(admin_row[chosen_id_col])
            rec["id"] = boundary_id
            rec["boundary_id"] = boundary_id
        geom = admin_row.geometry
        rec["geometry"] = geom
        bounds = geom.bounds
        rec["bbox"] = [bounds[0], bounds[1], bounds[2], bounds[3]]
        return rec

    # Process matched admin polygons (have H3 data)
    for idx, sub in joined.groupby("index_right"):
        admin_row = admin_gdf.loc[idx]
        rec = _build_admin_rec(admin_row, idx)

        for col in value_cols:
            stats = _summarize(sub[col])
            rec[f"{col}"] = stats["mean"]
            rec[f"{col}_min"] = stats["min"]
            rec[f"{col}_max"] = stats["max"]
            rec[f"{col}_p90"] = stats["p90"]
            rec[f"{col}_count"] = stats["count"]
            rec[f"{col}_share_pos"] = stats["share_pos"]

        out_rows.append(rec)

    # Include unmatched admin polygons (no H3 overlap) with zero flood values
    for idx in admin_gdf.index:
        if idx not in matched_admin_indices:
            admin_row = admin_gdf.loc[idx]
            rec = _build_admin_rec(admin_row, idx)

            for col in value_cols:
                rec[f"{col}"] = 0.0
                rec[f"{col}_min"] = 0.0
                rec[f"{col}_max"] = 0.0
                rec[f"{col}_p90"] = 0.0
                rec[f"{col}_count"] = 0
                rec[f"{col}_share_pos"] = 0.0

            out_rows.append(rec)

    if not out_rows:
        raise ValueError(
            "[admin_enrich] Aggregation produced no rows. "
            "This usually means the join was empty."
        )

    out_gdf = gpd.GeoDataFrame(out_rows, geometry="geometry", crs="EPSG:4326")
    print(
        f"[admin_enrich] Built output GeoDataFrame with {len(out_gdf)} features "
        f"and columns: {list(out_gdf.columns)}"
    )

    # ------------------------------------------------------------------
    # 5. Write GeoJSON-like output preserving ClimateIQ header
    # ------------------------------------------------------------------
    write_geojson_like(template_dict, out_gdf, out_path)
    print(f"[admin_enrich] Wrote GeoJSON to {out_path}")
    print("[admin_enrich] --------------------------------------------------")
    return out_path
