"""FloodRoutingCNN retrain on the FULL available training pool -- ~306
usable simulations across 11 cities (2 of them, CapeTown_Watershed_2 and
Phoenix_Central, never used at all before), vs. the ~38 hand-picked sims
every checkpoint so far actually trained on. Limited training diversity
was the clearest structural explanation for why the model doesn't
generalise well to unseen terrain (the runaway-pixel bug found on real
Philadelphia production data was exactly this kind of out-of-distribution
failure). This is the most direct test of that hypothesis.

Warm-started from the `downhill` checkpoint (routing-constraint fix,
currently one half of the production ensemble) -- keeps that fix, only
changes data scale from here.

MEMORY: the existing training scripts preload every chunk's full arrays
into a Python list up front, which was fine for ~38 sims (~200 chunks,
~20GB) but would be ~140GB+ for the full pool (~1500 chunks) -- not
something to naively scale up and hope fits. This script instead builds a
lightweight INDEX (file paths + elevation range + temporal vector + a
depth/flood-fraction filter check, all computed once) and loads each
chunk's actual arrays from disk fresh, on demand, inside sample_chunk() --
bounded memory regardless of dataset size, at the cost of repeated disk
I/O and feature recomputation per step (the same per-chunk cost every
evaluation script in this project already pays routinely).

Manhattan-config_v1 sims (13 of them) are excluded -- confirmed broken
during download (feature/label chunk indices don't line up), a real
upstream data issue, not something a retry fixes.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import ctypes
# Pin glibc's mmap threshold so large (chunk-sized, ~50MB+) numpy allocations
# always go through mmap/munmap and get returned to the OS immediately on
# free. Without this, glibc adaptively raises the threshold after repeated
# large frees and starts serving them from the process's brk arena instead,
# where freed-but-fragmented memory isn't returned to the OS -- verified
# this alone was driving RSS unbounded (60GB+ by step 100) purely from the
# training loop's repeated random cache-miss chunk loads against the
# ~1300-chunk pool, even with a small bounded LRU cache. Confirmed fixed:
# RSS plateaus and stays flat under this fix across hundreds of loads.
ctypes.CDLL("libc.so.6").mallopt(ctypes.c_int(-3), ctypes.c_int(128 * 1024))

import collections, hashlib, json, pathlib, subprocess, sys, random, time
import numpy as np
import tensorflow as tf
import keras

sys.path.insert(0, str(pathlib.Path(__file__).parent / "usl_models"))
from usl_models.flood_ml.model import FloodRoutingCNN, FloodModel, FloodConvLSTM
from usl_models.flood_ml.dataset import _compute_all_geo_features
from usl_models.flood_ml import customloss
from usl_models.flood_ml import constants

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

FILECACHE = pathlib.Path("/scratch/rmj7591/climateiq/train_data")
OUTPUT_DIR = pathlib.Path("/scratch/rmj7591/climateiq/train_output")
CHUNK_SIZE = 1000  # full resolution -- no cropping
K = 11             # up from 4 -- matches T_STRIP, the real evaluation horizon
TRAIN_STEPS = 20000
LOG_EVERY = 50
CKPT_EVERY = 300
MIN_MAX_DEPTH = 0.1
MIN_FLOOD_FRACTION = 0.01
DEEP_WATER_BIAS_PROB = 0.5
DEEP_WATER_MIN_DEPTH = 1.0
WARM_CKPT = "/scratch/rmj7591/climateiq/train_output/floodroutingcnn_bigdata_20261013-160742/weights_latest.npz"

timestamp = time.strftime("%Y%m%d-%H%M%S")
RUN_DIR = OUTPUT_DIR / f"floodroutingcnn_bigdata_{timestamp}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"Run dir: {RUN_DIR}")

_sim_list_path = pathlib.Path(__file__).parent / "all_available_sims_clean.txt"
TRAIN_SIMS = [line.strip() for line in open(_sim_list_path) if line.strip()]
TRAIN_SIMS = [s for s in TRAIN_SIMS if (FILECACHE / s).exists()]
print(f"Train sims: {len(TRAIN_SIMS)} (from {_sim_list_path.name})")

ELEV_RANGES = json.load(open(FILECACHE / "study_area_elevation_ranges.json"))


def study_id_of(sim):
    return sim.split("/")[0].split("-")[0]


def build_12ch_geo(geo_9ch):
    try:
        return _compute_all_geo_features(geo_9ch)
    except Exception as e:
        print(f"  [WARN] _compute_all_geo_features failed ({e})")
        h, w = geo_9ch.shape[:2]
        return np.concatenate([geo_9ch, np.zeros((h, w, 3), np.float32)], axis=-1)


def build_full_temporal(temporal_vec, m):
    v = temporal_vec.astype(np.float32)
    T = len(v)
    cumsum = np.cumsum(v)
    total = cumsum[-1] if cumsum[-1] > 0 else 1.0
    run_max = np.maximum.accumulate(v)
    delta_rate = np.concatenate([[0.0], np.diff(v)])
    feat_2d = np.stack([
        v, cumsum / total, delta_rate, np.log1p(cumsum), run_max,
        np.arange(T, dtype=np.float32) / max(T - 1, 1),
    ], axis=1).astype(np.float32)
    T_MAX = constants.MAX_RAINFALL_DURATION
    if T >= T_MAX:
        return feat_2d[:T_MAX]
    pad = np.zeros((T_MAX - T, m), np.float32)
    return np.concatenate([feat_2d, pad], axis=0)


# Build a LIGHTWEIGHT INDEX only -- (geo_file, lab_file, temporal_full,
# elev_range, max_depth) per chunk that passes the filters. temporal_full
# is tiny (~20KB) so it's fine to keep; geo/label arrays (~92MB/chunk) are
# NOT retained here -- sample_chunk() loads them fresh from disk on every
# call. See module docstring for why (memory).
#
# The per-chunk filter scan itself (reading every label file once to check
# max-depth/flood-fraction) touches ~78GB across the full ~1500-chunk pool.
# Doing that in-process, even with each array properly `del`ed, drives RSS
# into the tens of GB: glibc's malloc arena doesn't return freed same-sized
# large allocations to the OS reliably (verified -- periodic malloc_trim(0)
# only slowed, didn't stop, the growth), so it OOM'd two consecutive job
# launches at 32G and 64G. index_worker.py runs the scan in short-lived
# subprocesses (one per batch of sims) instead, so the OS fully reclaims
# memory on each exit regardless of allocator fragmentation -- verified
# bounded (~7GB worst-case batch, flat parent RSS) scanning the full pool
# inside a real 8G SLURM allocation. The result is cached to disk so this
# scan only ever runs once per (sim list, filter params) combination.
INDEX_CACHE_DIR = pathlib.Path(__file__).parent / "index_cache"
INDEX_CACHE_DIR.mkdir(exist_ok=True)
INDEX_BATCH = 20

sim_meta = {}
valid_sims = []
skipped = 0
for sim in TRAIN_SIMS:
    sim_dir = FILECACHE / sim
    tmp_path = sim_dir / "temporal.npy"
    geo_dir = sim_dir / "train" / "geospatial"
    if not tmp_path.exists() or not geo_dir.exists():
        skipped += 1
        continue
    r = ELEV_RANGES.get(study_id_of(sim))
    if r is None:
        skipped += 1
        continue
    temporal_vec = np.load(tmp_path)
    temporal_full = build_full_temporal(temporal_vec, 6)
    elev_range = r["elevation_max"] - r["elevation_min"]
    sim_meta[sim] = (temporal_full, elev_range)
    valid_sims.append(sim)

cache_key = hashlib.sha256(
    json.dumps(
        {"sims": valid_sims, "K": K, "min_max_depth": MIN_MAX_DEPTH,
         "min_flood_fraction": MIN_FLOOD_FRACTION},
        sort_keys=True,
    ).encode()
).hexdigest()[:16]
cache_path = INDEX_CACHE_DIR / f"chunk_index_{cache_key}.json"

if cache_path.exists():
    print(f"Loading cached chunk index: {cache_path}")
    chunk_records = json.load(open(cache_path))
else:
    print(f"No cached index ({cache_path.name}) -- scanning {len(valid_sims)} sims "
          f"via batched subprocesses (bounded memory, ~1 min per 20 sims)...")
    chunk_records = []
    worker_path = pathlib.Path(__file__).parent / "index_worker.py"
    for i in range(0, len(valid_sims), INDEX_BATCH):
        batch = valid_sims[i:i + INDEX_BATCH]
        p = subprocess.run(
            [sys.executable, str(worker_path), str(FILECACHE), str(K),
             str(MIN_MAX_DEPTH), str(MIN_FLOOD_FRACTION), *batch],
            capture_output=True, text=True,
        )
        if p.returncode != 0:
            raise RuntimeError(f"index_worker batch {i} failed: {p.stderr[-2000:]}")
        for line in p.stdout.splitlines():
            if line.strip():
                chunk_records.append(json.loads(line))
        print(f"  indexed sims {i}-{i + len(batch)}/{len(valid_sims)} "
              f"({len(chunk_records)} chunks so far)", flush=True)
    json.dump(chunk_records, open(cache_path, "w"))
    print(f"Cached index to {cache_path}")

sim_data = []
deep_sim_data = []
for rec in chunk_records:
    temporal_full, elev_range = sim_meta[rec["sim"]]
    entry = (
        pathlib.Path(rec["geo_file"]), pathlib.Path(rec["lab_file"]),
        temporal_full, elev_range, rec["max_depth"],
    )
    sim_data.append(entry)
    if rec["max_depth"] >= DEEP_WATER_MIN_DEPTH:
        deep_sim_data.append(entry)

print(f"Indexed {len(sim_data)} full chunks ({len(deep_sim_data)} with genuine deep water, "
      f"{skipped} sims skipped)")
if not sim_data:
    raise RuntimeError("No usable training examples found")

if "--build-index-only" in sys.argv:
    # Lets this same script pre-warm the on-disk index cache from a cheap
    # CPU-only allocation, so the (GPU-partition, more expensive) training
    # job just loads the cache instantly instead of spending ~16 minutes of
    # GPU-node time on the scan itself.
    print("--build-index-only: cache is populated, exiting before model setup.")
    sys.exit(0)


_CHUNK_CACHE = collections.OrderedDict()
# Cache the WHOLE pool (not just a small LRU slice): once every chunk has been
# seen once, every later sample is a cache hit and the expensive whitebox D8
# recomputation (~1.5s/chunk) never runs again -- this is what actually fixes
# GPU utilization (job 18377175 was auto-killed by HPC at 49.69% util because a
# small cache meant most steps were cache misses stuck waiting on CPU-bound
# disk+whitebox work).
#
# mallopt + posix_fadvise above are NECESSARY but NOT SUFFICIENT at this
# scale -- confirmed again on 2026-10-14 (job 19738527): even capped at 900
# chunks (~83GB of genuinely-retained Python arrays, well under budget),
# MaxRSS still climbed to exactly 200GB and got OOM-killed at step 650,
# matching the same "page cache not reclaimed fast enough under concurrent
# GPU-training memory pressure" pattern documented below for the earlier
# uncapped attempt (190GB by step 550, also still climbing). The retained
# cache itself was never the dominant growth source at either cap size --
# so shrinking it further mostly trades OOM risk for the OTHER documented
# failure mode (GPU-util auto-kill from too many cache misses) without
# fixing the real cause. A bigger --mem request was tried instead but isn't
# actually available: the node has 1.5TB free, but rtx6000_humanities /
# this account's QOS rejects ANY --mem above 200G outright at submission
# time (450G, 300G, even 210G all refused immediately, not queued -- a hard
# per-job ceiling, separate from the GrpTRES=1400G aggregate cap another
# user was independently consuming ~1112G of at the same time). So 200G is
# the real, fixed budget this script has to live within -- the only
# remaining lever is shrinking this cache further to leave more slack for
# the unresolved page-cache growth. Cut from 900 to 400 (~36.8GB retained
# vs ~83GB) on 2026-10-14, trading some GPU utilization (more cache misses
# -> more repeated ~1.5s whitebox D8 recomputation) for a real chance of
# running past step 650 at all.
_CHUNK_CACHE_MAX = min(len(sim_data), 400)


def _load_npy_dontneed(path):
    """np.load(), then tell the kernel to drop this file's pages from cache.

    Job 18371918 (200G budget) reached 187.6GB by step 600 and was still
    climbing, not plateauing -- confirmed via isolated tests (load_chunk_
    isolation_test.py) that this is real Linux page cache, not an
    anonymous-memory leak: reading many distinct large .npy files across a
    ~1300-chunk / ~117GB pool populates page cache per file, and it doesn't
    get reclaimed fast enough under concurrent GPU-training memory pressure
    to stay bounded even with a huge memory grant. POSIX_FADV_DONTNEED
    explicitly evicts this file's pages the moment we're done with them,
    capping page-cache growth at a small, predictable size regardless of
    how many distinct chunks get sampled over the full 20000-step run --
    a surgical fix instead of relying on a large memory grant and a hoped-
    for natural plateau.
    """
    with open(path, "rb") as f:
        arr = np.load(f)
        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return arr


def load_chunk(entry):
    """Load one indexed chunk's actual arrays, LRU-cached and bounded."""
    geo_file, lab_file, temporal_full, elev_range, max_depth = entry
    cached = _CHUNK_CACHE.get(geo_file)
    if cached is not None:
        _CHUNK_CACHE.move_to_end(geo_file)
        geo_12ch, labels_k = cached
    else:
        geo_9ch = _load_npy_dontneed(geo_file).astype(np.float32)
        geo_12ch = build_12ch_geo(geo_9ch)
        labels_t = _load_npy_dontneed(lab_file).astype(np.float32)
        labels_k = np.transpose(labels_t, (2, 0, 1))[:K]
        _CHUNK_CACHE[geo_file] = (geo_12ch, labels_k)
        if len(_CHUNK_CACHE) > _CHUNK_CACHE_MAX:
            _CHUNK_CACHE.popitem(last=False)
    return geo_12ch, labels_k, temporal_full, elev_range, max_depth

params = FloodModel.Params(
    loss_version="v11_fa2",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4, global_clipnorm=1.0),
)
model = FloodRoutingCNN(params=params, spatial_dims=(CHUNK_SIZE, CHUNK_SIZE))
# v10 (this run chain's original loss) was deleted from customloss.py during
# an unrelated dead-code cleanup elsewhere in the project, before this
# script was checked as a caller -- confirmed unrecoverable (absent from
# every git branch/blob/stash, since it was only ever an uncommitted local
# edit). v11_fa2 is the disclosed replacement: it's what the production
# FloodConvLSTM checkpoint actually trains with, and its depth_weight term
# (1 + 4*sigmoid(30*(depth-0.05)) + 12*sigmoid(5*(depth-1.0))) up-weights
# ~1m+ water by up to 12x -- a plausible, evidence-based answer to the
# near-zero dangerous-depth CSI the production FloodRoutingCNN ensemble
# showed in a full true-closed-loop-AR evaluation (2026-10-13). This is a
# real methodology change partway through this warm-started checkpoint
# chain, not a silent swap -- flagging here so it's not mistaken for v10.
loss_fn = customloss.make_hybrid_loss_v11_fa2
optimizer = params.optimizer

_dummy_depth = tf.zeros((1, CHUNK_SIZE, CHUNK_SIZE, 1))
_dummy_geo = tf.zeros((1, CHUNK_SIZE, CHUNK_SIZE, 12))
_ = model._route_step(_dummy_depth, _dummy_geo, tf.zeros((1,)), tf.zeros((1,)), tf.zeros((1, CHUNK_SIZE, CHUNK_SIZE, 1)), tf.zeros((1,)))
_d = np.load(WARM_CKPT)
_warm_weights = [_d[f"w{i:02d}"] for i in range(len(_d.files))]
model.set_weights(_warm_weights)
print(f"Warm-started from bigdata run's own checkpoint (resumed after low-GPU-util "
      f"warning, ~step 3200, {WARM_CKPT}): {len(_warm_weights)} weights")


@tf.function(reduce_retracing=True)
def _train_step_fn(x, y):
    with tf.GradientTape() as tape:
        depth = tf.zeros((1, CHUNK_SIZE, CHUNK_SIZE, 1), dtype=tf.float32)
        cumul_F = tf.zeros((1, CHUNK_SIZE, CHUNK_SIZE, 1), dtype=tf.float32)
        total_loss = 0.0
        for t in range(K):
            rain_rate = x["temporal"][:, t, 0]
            rain_cum = x["temporal"][:, t, 1]
            depth, cumul_F = model._route_step(
                depth, x["geospatial"], rain_rate, rain_cum, cumul_F, x["elevation_range_m"]
            )
            total_loss += loss_fn(y[:, t], depth)
        total_loss = total_loss / K
    grads = tape.gradient(total_loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, 1.0)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss


def sample_chunk():
    pool = deep_sim_data if (random.random() < DEEP_WATER_BIAS_PROB and deep_sim_data) else sim_data
    entry = random.choice(pool)
    return load_chunk(entry)


def save_checkpoint(name):
    ws = model.get_weights()
    np.savez(str(RUN_DIR / name), **{f"w{i:02d}": w for i, w in enumerate(ws)})


print(f"\nTraining {TRAIN_STEPS} steps, K={K}, FULL RESOLUTION ({CHUNK_SIZE}x{CHUNK_SIZE}, no cropping), "
      f"loss=v11_fa2, true-slope=ON, downhill-bias=ON, deep-water-bias={DEEP_WATER_BIAS_PROB}\n")
loss_history = []
best_loss = float("inf")
t0 = time.time()
for step in range(1, TRAIN_STEPS + 1):
    geo_12ch, labels_k, temporal_full, elev_range, max_depth = sample_chunk()
    temporal_full_tf = tf.constant(temporal_full[np.newaxis], dtype=tf.float32)
    temporal_windows = tf.stack(
        [FloodConvLSTM._get_temporal_window(temporal_full_tf, t, 1)[:, 0] for t in range(1, K + 1)],
        axis=1,
    )
    x = {
        "geospatial": tf.constant(geo_12ch[np.newaxis], dtype=tf.float32),
        "temporal": temporal_windows,
        "elevation_range_m": tf.constant([elev_range], dtype=tf.float32),
    }
    y = tf.constant(labels_k[np.newaxis], dtype=tf.float32)

    loss_val = float(_train_step_fn(x, y))
    loss_history.append(loss_val)

    if step % LOG_EVERY == 0:
        recent = np.mean(loss_history[-LOG_EVERY:])
        elapsed_min = (time.time() - t0) / 60.0
        print(f"  step {step}/{TRAIN_STEPS}  loss={recent:.4f}  elapsed={elapsed_min:.1f}min")
        if np.isnan(recent):
            print("FAIL: NaN loss -- stopping")
            break

    if step % CKPT_EVERY == 0:
        save_checkpoint("weights_latest.npz")
        recent = np.mean(loss_history[-CKPT_EVERY:])
        if recent < best_loss:
            best_loss = recent
            save_checkpoint("weights_best.npz")
        print(f"  [checkpoint] step {step}: saved (best={best_loss:.4f})")

save_checkpoint("weights_final.npz")
print(f"\nDone training. Final checkpoint: {RUN_DIR / 'weights_final.npz'}")
print(f"Total time: {(time.time() - t0) / 60:.1f} min")
