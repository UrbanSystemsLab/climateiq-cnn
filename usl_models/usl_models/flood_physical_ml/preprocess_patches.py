"""Pre-slice CityCAT chunks into ready-to-load patches for flood_physical_ml.

All expensive work (12-channel geo features incl. whitebox flow, label
transpose, temporal features) is done once here. Training then only needs
``np.load`` + an index lookup per sample, so samples can be drawn in fully
random order across chunks at no extra cost.

Output layout (``out_dir`` should encode patch/stride, e.g. .../p256_s128):

    out_dir/
      meta.json                      build parameters
      index_{split}.csv              one row per patch (see _INDEX_FIELDS)
      <sim_name>/
        temporal.npy                 raw rainfall vector (T_rain,)
        temporal_v2.npy              _build_temporal_tensor_v2 output (T_rain, 6)
        <split>/geospatial/<stem>__y<y0>_x<x0>.npy   (P, P, GEO_FEATURES)
        <split>/labels/<stem>__y<y0>_x<x0>.npy       (T, P, P)

``valid`` in the index is a 0/1 string: character t is 1 iff
``labels[t:t+n_future_steps]`` passes the flood filter. It depends on
n_future_steps / thresholds; rerun with ``--index-only`` after changing them
(reuses the patch files, skips the geo feature computation).

Usage:
    python -m usl_models.flood_physical_ml.preprocess_patches \
        --filecache-dir /scratch/hw4402/climateiq_filecache_us \
        --out-dir /scratch/hw4402/climateiq_patches_us/p256_s128 \
        --patch-size 256 --stride 128 --n-future-steps 6 --workers 8
"""

import argparse
import csv
import json
import multiprocessing as mp
import pathlib
import shutil

import numpy as np

from usl_models.flood_physical_ml import constants
from usl_models.flood_physical_ml.dataset import (
    FEATURE_DIRNAME,
    LABEL_DIRNAME,
    TEMPORAL_FILENAME,
    _load_geo_with_sink,
)

TEMPORAL_V2_FILENAME = "temporal_v2.npy"
_INDEX_FIELDS = ["sim", "stem", "y0", "x0", "T", "valid"]


def patch_name(stem: str, y0: int, x0: int) -> str:
    return f"{stem}__y{y0}_x{x0}.npy"


def _is_valid_patch(labels_patch, min_max_depth, min_flood_fraction) -> bool:
    """Same filter as load_dataset_windowed_patches._is_valid_patch."""
    if labels_patch.max() < min_max_depth:
        return False
    return (labels_patch > 0).any(axis=0).mean() >= min_flood_fraction


def _valid_string(labels_patch, n_future, min_max_depth, min_flood_fraction) -> str:
    n_starts = labels_patch.shape[0] - n_future + 1
    return "".join(
        "1"
        if _is_valid_patch(
            labels_patch[t : t + n_future], min_max_depth, min_flood_fraction
        )
        else "0"
        for t in range(max(n_starts, 0))
    )


def _process_chunk(job: dict) -> list[dict]:
    """Slice one (sim, split, stem) chunk into patches. Runs in a worker."""
    src = pathlib.Path(job["src_sim_dir"]) / job["split"]
    dst = pathlib.Path(job["dst_sim_dir"]) / job["split"]
    stem, p, stride = job["stem"], job["patch_size"], job["stride"]
    geo_dir, label_dir = dst / FEATURE_DIRNAME, dst / LABEL_DIRNAME
    geo_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    labels = np.transpose(np.load(src / LABEL_DIRNAME / f"{stem}.npy"), (2, 0, 1))
    T, H, W = labels.shape

    geo = None
    if not job["index_only"]:
        geo = _load_geo_with_sink(src / FEATURE_DIRNAME / f"{stem}.npy")
        assert geo.shape[-1] == constants.GEO_FEATURES, geo.shape

    rows = []
    for y0 in range(0, H - p + 1, stride):
        for x0 in range(0, W - p + 1, stride):
            name = patch_name(stem, y0, x0)
            labels_patch = labels[:, y0 : y0 + p, x0 : x0 + p]
            if geo is not None:
                np.save(
                    geo_dir / name,
                    geo[y0 : y0 + p, x0 : x0 + p].astype(job["dtype"]),
                )
                np.save(label_dir / name, labels_patch.astype(job["dtype"]))
            rows.append(
                dict(
                    sim=job["sim"],
                    stem=stem,
                    y0=y0,
                    x0=x0,
                    T=T,
                    valid=_valid_string(
                        labels_patch,
                        job["n_future_steps"],
                        job["min_max_depth"],
                        job["min_flood_fraction"],
                    ),
                )
            )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--filecache-dir", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    ap.add_argument("--sims", nargs="*", default=None,
                    help="Sim names (default: every <city>/<rainfall> in filecache)")
    ap.add_argument("--splits", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--n-future-steps", type=int, default=6)
    ap.add_argument("--min-max-depth", type=float, default=0.05)
    ap.add_argument("--min-flood-fraction", type=float, default=0.12)
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32",
                    help="Storage dtype; float16 halves disk usage")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--index-only", action="store_true",
                    help="Only rebuild index_*.csv (patch files must already exist)")
    ap.add_argument("--allow-missing-flow", action="store_true",
                    help="Proceed without whitebox/rasterio (flow channels = 0)")
    args = ap.parse_args()

    if not args.index_only and not args.allow_missing_flow:
        try:
            import rasterio  # noqa: F401
            import whitebox  # noqa: F401
        except ImportError as e:
            raise SystemExit(
                f"{e}. compute_flow_features would silently write zero flow "
                "channels. Install whitebox + rasterio, or pass --allow-missing-flow."
            )

    sims = args.sims or sorted(
        str(d.relative_to(args.filecache_dir))
        for d in args.filecache_dir.glob("*/*")
        if (d / TEMPORAL_FILENAME).exists()
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs_by_split: dict[str, list[dict]] = {s: [] for s in args.splits}
    for sim in sims:
        src_sim, dst_sim = args.filecache_dir / sim, args.out_dir / sim
        if not (src_sim / TEMPORAL_FILENAME).exists():
            print(f"skip {sim}: no {TEMPORAL_FILENAME}")
            continue
        dst_sim.mkdir(parents=True, exist_ok=True)
        if not args.index_only:
            from usl_models.flood_physical_ml.dataset import _build_temporal_tensor_v2

            temporal_vec = np.load(src_sim / TEMPORAL_FILENAME)
            shutil.copy(src_sim / TEMPORAL_FILENAME, dst_sim / TEMPORAL_FILENAME)
            np.save(
                dst_sim / TEMPORAL_V2_FILENAME,
                _build_temporal_tensor_v2(temporal_vec).numpy(),
            )
        for split in args.splits:
            feat_dir = src_sim / split / FEATURE_DIRNAME
            label_dir = src_sim / split / LABEL_DIRNAME
            if not feat_dir.exists() or not label_dir.exists():
                continue
            stems = sorted(
                {f.stem for f in feat_dir.glob("*.npy")}
                & {f.stem for f in label_dir.glob("*.npy")}
            )
            for stem in stems:
                jobs_by_split[split].append(
                    dict(
                        sim=sim, split=split, stem=stem,
                        src_sim_dir=str(src_sim), dst_sim_dir=str(dst_sim),
                        patch_size=args.patch_size, stride=args.stride,
                        n_future_steps=args.n_future_steps,
                        min_max_depth=args.min_max_depth,
                        min_flood_fraction=args.min_flood_fraction,
                        dtype=args.dtype, index_only=args.index_only,
                    )
                )

    with mp.get_context("spawn").Pool(args.workers) as pool:
        for split, jobs in jobs_by_split.items():
            if not jobs:
                continue
            index_path = args.out_dir / f"index_{split}.csv"
            with open(index_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_INDEX_FIELDS)
                writer.writeheader()
                for i, rows in enumerate(pool.imap_unordered(_process_chunk, jobs)):
                    writer.writerows(rows)
                    print(f"[{split}] {i + 1}/{len(jobs)} chunks", flush=True)
            print(f"wrote {index_path}")

    meta = {k: (str(v) if isinstance(v, pathlib.Path) else v)
            for k, v in vars(args).items()}
    meta["geo_features"] = constants.GEO_FEATURES
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
