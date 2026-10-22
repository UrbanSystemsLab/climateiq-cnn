"""Standalone worker: indexes a batch of sims' label chunks and prints JSON to
stdout. Run as a subprocess (one per batch) from train_floodroutingcnn_bigdata.py
so each batch's memory is fully reclaimed by the OS on exit -- large repeated
numpy allocations inside a single long-lived process fragment glibc's malloc
arena badly enough that RSS climbs into the tens of GB over ~1500 files even
though every array is properly `del`ed (verified: malloc_trim(0) only slows,
doesn't fix, the growth). A fresh process per batch sidesteps that entirely.

Args: FILECACHE_DIR K MIN_MAX_DEPTH MIN_FLOOD_FRACTION sim1 [sim2 ...]
Output: one JSON object per line to stdout, one per qualifying chunk:
  {"sim": ..., "geo_file": ..., "lab_file": ..., "max_depth": ...}
"""
import json
import pathlib
import sys

import numpy as np

FILECACHE = pathlib.Path(sys.argv[1])
K = int(sys.argv[2])
MIN_MAX_DEPTH = float(sys.argv[3])
MIN_FLOOD_FRACTION = float(sys.argv[4])
sims = sys.argv[5:]

for sim in sims:
    sim_dir = FILECACHE / sim
    geo_dir = sim_dir / "train" / "geospatial"
    lab_dir = sim_dir / "train" / "labels"
    if not geo_dir.exists() or not lab_dir.exists():
        continue
    geo_files = sorted(geo_dir.glob("*.npy"))
    lab_map = {f.stem: f for f in lab_dir.glob("*.npy")}
    for geo_file in geo_files:
        stem = geo_file.stem
        if stem not in lab_map:
            continue
        lab_file = lab_map[stem]
        labels_t = np.load(lab_file).astype(np.float32)
        labels = np.transpose(labels_t, (2, 0, 1))
        if labels.shape[0] < K + 1:
            continue
        labels_k = labels[:K]
        max_depth = float(labels_k.max())
        if max_depth < MIN_MAX_DEPTH:
            continue
        if (labels_k > 0).any(axis=0).mean() < MIN_FLOOD_FRACTION:
            continue
        print(json.dumps({
            "sim": sim,
            "geo_file": str(geo_file),
            "lab_file": str(lab_file),
            "max_depth": max_depth,
        }))
