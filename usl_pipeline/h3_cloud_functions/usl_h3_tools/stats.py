import numpy as np
import pandas as pd

def p90(x: pd.Series):
    if len(x) == 0:
        return None
    return float(np.nanpercentile(x.to_numpy(dtype=float), 90))

def share_pos(x: pd.Series):
    if len(x) == 0:
        return None
    arr = pd.to_numeric(x, errors='coerce').to_numpy()
    n = np.isfinite(arr).sum()
    if n == 0:
        return None
    return float((arr > 0).sum() / n)

def stats_for_group(s: pd.Series, base: str) -> dict:
    s = pd.to_numeric(s, errors='coerce')
    d = {}
    d[base] = float(np.nanmean(s)) if s.notna().any() else None
    d[f"{base}_min"] = float(np.nanmin(s)) if s.notna().any() else None
    d[f"{base}_max"] = float(np.nanmax(s)) if s.notna().any() else None
    d[f"{base}_p90"] = p90(s) if s.notna().any() else None
    d[f"{base}_count"] = int(s.notna().sum())
    d[f"{base}_share_pos"] = share_pos(s)
    return d
