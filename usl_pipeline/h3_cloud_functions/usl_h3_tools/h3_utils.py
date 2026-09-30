from typing import Optional
import pandas as pd
from shapely.geometry import Polygon
from shapely import wkt
from shapely.geometry import shape as shp_shape
import json
import h3

def detect_h3_col(df: pd.DataFrame, override: Optional[str] = None) -> str:
    if override and override in df.columns:
        return override
    for c in ['h3','hex','h3_index','h3_id']:
        if c in df.columns:
            return c
    raise ValueError("No H3 column found. Use --h3-col to specify it explicitly.")

def h3_to_polygon(h):
    coords = h3.h3_to_geo_boundary(h, geo_json=True)
    return Polygon(coords)

def build_hex_gdf_from_csv(df: pd.DataFrame, h3_col: str, has_geometry: bool = False):
    import geopandas as gpd
    if has_geometry and 'geometry' in df.columns:
        # geometry may be WKT or GeoJSON string
        geoms = []
        for val in df['geometry']:
            if val is None or (isinstance(val, float) and pd.isna(val)):
                geoms.append(None)
                continue
            s = str(val).strip()
            try:
                if s.startswith('{'):
                    geoms.append(shp_shape(json.loads(s)))
                else:
                    geoms.append(wkt.loads(s))
            except Exception:
                geoms.append(None)
        gdf = gpd.GeoDataFrame(df.copy(), geometry=geoms, crs='EPSG:4326')
        return gdf
    # Build polygons from H3
    gdf = pd.DataFrame({h3_col: df[h3_col].astype(str)}).dropna().drop_duplicates()
    gdf['geometry'] = gdf[h3_col].map(h3_to_polygon)
    import geopandas as gpd
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry', crs='EPSG:4326')
    return gdf

def parent_series(df: pd.DataFrame, h3_col: str, res: int) -> pd.Series:
    return df[h3_col].astype(str).map(lambda x: h3.h3_to_parent(x, res))
