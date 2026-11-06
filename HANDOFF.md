# FloodML Handoff

## What this is

A ConvLSTM-based flood depth predictor for urban areas, trained per-chunk
and run autoregressively over a rainfall storm. Inputs: geospatial features
(DEM, soil, etc.), rainfall vector, and a rolling window of prior depth
maps. Output: per-pixel flood depth in meters.

## Production checkpoint

**Branch:** `arch-v3-experiment` (this branch). Promote to `main` via PR.

**Best weights:** `train_output/run_20260425-172505/weights_ordered.npz`
(not in git — generated locally; see "Train" below).

| Metric                              | Value     | Target |
| ----------------------------------- | --------- | ------ |
| Metric A — % flooded ≤10cm tol      | **66.66%** | ≥90%   |
| Metric C — peak-depth agreement     | 25.4%     | ≥90%   |
| Metric D — spatial correlation      | 0.74      | ≥0.90  |
| False-positive rate                 | 5.1%      | ≤2%    |

**By city (Metric A):**
- Phoenix_SM, Phoenix_PV: 78–86% (production-ready)
- Atlanta, NYC, Manhattan: 50–66% (deep urban canyon flooding under-predicted; see "Known limit")

## Train

```bash
# From repo root, on a machine with 2 GPUs and the FloodML dataset under
# /home/shared/climateiq/filecache (adjust FILECACHE_DIR in run_quick_train.py
# if different on HPC).
export CUDA_VISIBLE_DEVICES=0,1
python run_quick_train.py
```

Configuration is at the top of `run_quick_train.py`. The production config:

- `EPOCHS=2`, `STEPS_PER_EPOCH=2000`, `BATCH_SIZE=4`, `PATCH_SIZE=256`
- `WARM_START=True` from the V3 R3 checkpoint
- `USE_RAIN_BROADCAST=True`, `USE_DEEP_DECODER=False`, `USE_GROUP_NORM=False`
- `loss_version="v1"`, peak LR `2e-5`, cosine decay
- Walltime: ~50 min per 2-epoch run on 2× A100

Output: `train_output/run_<timestamp>/weights_ordered.npz`.

## Eval

```bash
python eval_production.py <path/to/weights_ordered.npz> <output.json>
```

Auto-detects which architecture flags were used by inspecting weight tensor
shapes. Walks all simulations under `FILECACHE`, runs single-step
GT-context prediction at peak rainfall, reports per-chunk metrics A/C/D
plus IoU and false-positive rate. ~1 min walltime.

## Architecture flags

All in `FloodModel.Params` in
`usl_models/usl_models/flood_ml/model.py`. All default `False`, preserving
the V1 baseline behaviour — flip to opt in.

| Flag                  | What it does                                                                 | Status                |
| --------------------- | ---------------------------------------------------------------------------- | --------------------- |
| `use_rain_broadcast`  | Tiles rain rate + cumulative rain as extra spatiotemporal input channels     | **Production: True**  |
| `use_dilated_geo`     | Dilated convs before `geo_cnn`                                               | Off (untested in v3)  |
| `use_storm_embed`     | Global storm-intensity FC → spatial bias                                     | Off                   |
| `use_deep_decoder`    | Extra refinement convs at H/2 + H resolution                                 | Off (regressed in R4) |
| `use_group_norm`      | Replace decoder BatchNorm with GroupNorm                                     | Off — see "Known limit" |
| `loss_version`        | `"v1"` log-depth hybrid (proven) or `"v3"` depth-weighted linear MSE         | `"v1"` in production  |

## Known limit — pred caps at ~2.5m on inference

Cause: the two decoder BatchNorm layers
(`flood_conv_lstm/batch_normalization_1` and `_2`) accumulated running
statistics with `running_var` ~29 on the second one during the original
30+ epochs of training under the old 2.5m feedback clip. At inference
those running stats compress activations such that single-step output
caps at ~2.5m, even on Atlanta/Manhattan chunks where GT is 4m.

What we tried that did not fix it:
- Reset BN running stats and continue training — stats re-accumulated
- Per-city BN refresh at inference — unlocked deep predictions but
  destroyed shallow accuracy (Metric A went 66% → 61%)
- Switch to V3 loss — small temporary gain, regressed by epoch 3

What works on synthetic / single-batch tests but needs a longer real
training run to fully realise:
- `use_group_norm=True` — eliminates the BN train/eval gap entirely.
  After 200 force-overfit steps, single-step inference reached 3.92m on
  a 4m GT sample. A 2-epoch warm-start on real data hit Metric A 47.95%
  with FP rate 20% (model unleashed but not calibrated yet). Plan for v2:
  warm-start V3 R3, flip `use_group_norm=True`, train 8–15 epochs at
  LR 5e-6 → expect Metric A breakthrough + acceptable FP rate.

## Layout

```
run_quick_train.py           # main training entry point
eval_production.py           # production eval (auto-detects flags)
run_predict_city.py          # batch prediction over a city's chunks
run_predict_fullchunk.py     # single-chunk prediction with stitching

usl_models/usl_models/flood_ml/
    model.py                 # FloodModel + FloodConvLSTM
    customloss.py            # V1 + V3 loss functions
    dataset.py               # tf.data pipeline + feature builders
    constants.py
```

Local diagnostic / experiment scripts (`test_*.py`, `diagnose_*.py`,
`compare_*.py`, `pretrain_validate.py`, etc.) are gitignored — they were
single-use investigations and aren't meant to ship.

## Pointers for v2

1. Implement `use_group_norm` end-to-end:
   - `model.py` already has the flag and constructs `GroupNormalization`
     when set; warm-start logic in `run_quick_train.py` already handles
     BN→GN tensor count change.
   - Run `python test_groupnorm_validate.py` (gitignored, ask Rohan for
     the file) to confirm 4m unlock at single-step inference before
     committing GPU time.
   - Train 8–15 epochs warm-started from the production R6 checkpoint at
     LR 5e-6.
2. If GN still calibration-noisy, try LayerNorm or InstanceNorm in the
   decoder — same train/eval consistency, different normalisation grouping.
3. The autoregressive `call_n` loop in `model.py` ALSO has the 4m clip
   (line ~1019); both train and inference clips are now consistent.
