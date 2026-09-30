from typing import List, Optional, Dict
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping
from .h3_utils import detect_h3_col, parent_series, h3_to_polygon
from .stats import stats_for_group

def make_hex_level_geojson(h3_csv_path: str,
                           value_cols: List[str],
                           res: int,
                           out_path: str,
                           h3_col: Optional[str] = None):
    df = pd.read_csv(h3_csv_path)
    h3c = detect_h3_col(df, h3_col)
    df['h3_parent'] = parent_series(df, h3c, res)
    # Build polygons per parent
    parents = df['h3_parent'].dropna().astype(str).unique().tolist()
    gdf = gpd.GeoDataFrame({'h3': parents,
                            'geometry': [h3_to_polygon(h) for h in parents]},
                            geometry='geometry', crs='EPSG:4326')
    # Aggregate stats per parent
    feats = []
    for h in parents:
        sub = df[df['h3_parent'] == h]
        props: Dict = {'h3': h}
        for col in value_cols:
            if col in sub.columns:
                props.update(stats_for_group(sub[col], col))
        feats.append({
            "type": "Feature",
            "properties": props,
            "geometry": mapping(gdf.loc[gdf['h3'] == h, 'geometry'].values[0])
        })
    fc = {"type": "FeatureCollection", "name": f"h3_level_{res}", "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}, "features": feats}
    import json
    with open(out_path, 'w') as f:
        json.dump(fc, f)
