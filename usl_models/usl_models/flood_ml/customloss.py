"""Loss functions and physics-aware auxiliary losses for FloodML.

Primary training loss
---------------------
make_hybrid_loss  – flood-aware weighted MSE (used by model.compile).

Auxiliary losses (used in a custom training loop alongside make_hybrid_loss)
---------------------------------------------------------------------------
arrival_time_loss        – differentiable timing loss (requires multi-step preds).
storage_consistency_loss – soft mass-balance regulariser (requires multi-step preds).

Both auxiliary losses need the *full predicted sequence* from ``call_n``, so they
cannot be plugged directly into ``model.fit``.  See the docstrings below for a
minimal ``tf.GradientTape`` snippet showing how to combine them.
"""

import tensorflow as tf
from keras.saving import register_keras_serializable


@register_keras_serializable(package="Custom", name="flood_weighted_mse")
def flood_weighted_mse(
    y_true,
    y_pred,
    flood_thresh=0.05,
    flood_weight=30.0,
    alpha=10.0,
    delta=0.5,
):
    """Huber-based flood loss with adaptive flood weighting and underprediction penalty.

    Improvements over plain MSE:
    1. Huber loss — less sensitive to outlier depths than pure MSE.
    2. Sigmoid flood weighting — smoothly upweights flooded pixels (depth > thresh).
    3. Underprediction penalty — flood models that predict too shallow are worse
       than slightly over-predicting (safety-critical).

    Args:
        y_true: Ground-truth flood depth [B, H, W] or [B, H, W, 1].
        y_pred: Predicted flood depth     [B, H, W] or [B, H, W, 1].
        flood_thresh: Depth threshold for "flooded" sigmoid transition.
        flood_weight: Max extra weight for flooded pixels.
        alpha: Sigmoid steepness.
        delta: Huber transition point.

    Returns:
        Scalar loss value.
    """
    err = y_pred - y_true
    abs_err = tf.abs(err)

    huber = tf.where(
        abs_err < delta,
        0.5 * tf.square(err),
        delta * (abs_err - 0.5 * delta),
    )

    w = 1.0 + flood_weight * tf.sigmoid(alpha * (y_true - flood_thresh))

    # Penalise underprediction more (safety-critical for flood warning)
    under_pred = tf.nn.relu(y_true - y_pred)
    under_penalty = 0.3 * w * tf.square(under_pred)

    return tf.reduce_mean(w * huber + under_penalty)


# Maximum flood depth considered physically meaningful for training.
# Depths above this threshold are treated as simulation outliers and capped.
# 4 m: covers Atlanta (peak ~4.4 m) and Phoenix (peak ~1.6 m) events;
# Manhattan 5-7 m ponding in closed canyons above this are outliers.
DEPTH_CAP_M = 4.0


def _flood_focal_loss(
    y_true: tf.Tensor,
    y_pred_relu: tf.Tensor,
    threshold: float = 0.05,
    gamma: float = 2.0,
    flooded_weight: float = 10.0,
    false_alarm_weight: float = 1.0,
) -> tf.Tensor:
    """Binary flood-detection focal loss.

    Directly penalises predicting flooding in the wrong spatial locations —
    the key failure mode identified: model predicts correct depth magnitude
    but in wrong pixels (Pearson corr ≈ 0 at flooded pixels).

    A pixel is 'flooded' if GT depth > threshold (default 5 cm).
    flooded_weight: asymmetric weight on missed floods (false negatives)
    vs false alarms (false positives).  10× weight on missed floods is
    appropriate because producing no flood where flooding exists is far
    worse than a small false alarm.

    false_alarm_weight: explicit weight on false alarms (predicting wet
    where GT is dry). Defaults to 1.0 (v6-v10's implicit, never-tuned
    value) -- kept for exact backward compatibility. This is the ONLY term
    in make_hybrid_loss_v6..v10 that penalises false positives FAR from any
    real flooding at all: the weighted_mse term in those losses is masked
    to a 5-pixel dilation of real flooding (`effective_mask`), so far-field
    false positives are invisible to it entirely, and this focal term is
    itself only weighted 0.3x in the total loss. That combination -- a
    false-alarm-rate that stayed at 0.7-0.85 across every v6/v8/v9/v10
    variant and every architecture change tried in one session (all of
    which only ever adjusted flooded_weight down, never false_alarm_weight
    up) -- is the direct, mechanistic explanation for why over-prediction
    never improved: nothing in the loss was ever asked to care.

    Args:
        y_true:       [B, H, W] GT depth, already capped at DEPTH_CAP_M.
        y_pred_relu:  [B, H, W] predicted depth, relu'd, uncapped.
        threshold:    Flood detection threshold in metres.
        gamma:        Focal loss concentration parameter.
        flooded_weight: Extra weight on missed floods (false negatives).
        false_alarm_weight: Extra weight on false alarms (false positives).

    Returns:
        Scalar focal loss.
    """
    flood_gt = tf.cast(y_true > threshold, tf.float32)  # [B, H, W]

    # Smooth flood probability centred at threshold
    # Sharpness 20 → 90% confident by ±0.1 m from threshold
    flood_prob = tf.sigmoid(20.0 * (y_pred_relu - threshold))  # [B, H, W]

    eps = 1e-7
    bce = -(
        flooded_weight * flood_gt * tf.math.log(flood_prob + eps)
        + false_alarm_weight * (1.0 - flood_gt) * tf.math.log(1.0 - flood_prob + eps)
    )

    # Focal weight: hard examples (wrong) get higher weight
    p_t = flood_gt * flood_prob + (1.0 - flood_gt) * (1.0 - flood_prob)
    focal = tf.pow(1.0 - p_t, gamma) * bce

    return tf.reduce_mean(focal)


def _abedice_loss(
    y_true: tf.Tensor,
    y_pred_relu: tf.Tensor,
    h_min: float = 0.02,
    sharpness: float = 40.0,
    boundary_weight: float = 5.0,
    fp_weight: float = 0.8,
    fn_weight: float = 1.4,
    eps: float = 1e-6,
) -> tf.Tensor:
    """Adaptive Boundary-enhanced Dice loss for flood spatial extent.

    Boundary pixels receive 5× higher weight via morphological gradient
    (dilation - erosion of the GT wet mask), directly targeting the failure
    mode where spatial correlation is poor because the model misplaces the
    flood edge rather than misestimating depth.

    Asymmetric FN > FP (1.4 vs 0.8): missing a flooded pixel is penalised
    more than a false alarm, appropriate for flood-safety applications.

    Args:
        y_true:      [B, H, W] GT depth, already capped and NaN-masked.
        y_pred_relu: [B, H, W] predicted depth, relu'd.
        h_min:       Wet/dry threshold in metres (default 2 cm).
        sharpness:   Sigmoid steepness for soft wet mask.
        boundary_weight: Extra weight multiplier on boundary pixels.
        fp_weight:   FP weight in the Dice denominator.
        fn_weight:   FN weight in the Dice denominator.

    Reference: https://www.sciencedirect.com/science/article/pii/S1746809425002526
    """
    # Soft predicted wet mask [B, H, W]
    pred_wet = tf.sigmoid(sharpness * (y_pred_relu - h_min))

    # Hard GT wet mask [B, H, W]
    true_wet = tf.cast(y_true > h_min, tf.float32)

    # Boundary via morphological gradient: dilation(mask) - erosion(mask)
    tw_4d = true_wet[..., tf.newaxis]  # [B, H, W, 1] for max_pool2d
    dilation = tf.nn.max_pool2d(tw_4d, ksize=3, strides=1, padding="SAME")
    erosion = 1.0 - tf.nn.max_pool2d(1.0 - tw_4d, ksize=3, strides=1, padding="SAME")
    boundary = tf.squeeze(
        tf.clip_by_value(dilation - erosion, 0.0, 1.0), axis=-1
    )  # [B, H, W]

    # Adaptive weight: base + wet region + boundary emphasis
    weight = 1.0 + true_wet + boundary_weight * boundary

    # Batch-level TP, FP, FN (summed across entire batch × spatial grid)
    TP = tf.reduce_sum(weight * pred_wet * true_wet)
    FP = tf.reduce_sum(weight * pred_wet * (1.0 - true_wet))
    FN = tf.reduce_sum(weight * (1.0 - pred_wet) * true_wet)

    score = (2.0 * TP + eps) / (2.0 * TP + fp_weight * FP + fn_weight * FN + eps)
    return 1.0 - score


@register_keras_serializable(package="Custom", name="make_hybrid_loss")
def make_hybrid_loss(y_true, y_pred):
    """Flood-aware log-depth loss + spatial flood-detection losses.

    Loss terms:
    1. weighted_log_mse  — log-depth MSE with flood-depth weighting and 5×5
       spatial-activity mask.  Teaches depth magnitude.
    2. peak_penalty (×0.5) — penalises underestimating the spatial maximum
       depth per sample in log space.
    3. focal (×1.0) — binary cross-entropy on flooded/dry pixels with 10×
       asymmetric weight on missed floods and focal down-weighting of easy
       pixels.  Teaches WHERE flooding occurs.
    4. abedice (×0.3) — Adaptive Boundary-enhanced Dice loss: 5× boundary
       weighting via morphological gradient, asymmetric FN>FP.  Directly
       targets the spatial-correlation failure mode where the model places
       flooding in the wrong locations.

    Args:
        y_true: Ground-truth flood depth  [B, H, W] or [B, H, W, 1].
        y_pred: Predicted flood depth     [B, H, W] or [B, H, W, 1].

    Returns:
        Scalar loss value.
    """
    # ── normalise to [B, H, W] ──────────────────────────────────────
    if y_true.shape.ndims == 4 and y_true.shape[-1] == 1:
        y_true = y_true[..., 0]
    if y_pred.shape.ndims == 4 and y_pred.shape[-1] == 1:
        y_pred = y_pred[..., 0]

    # ── NaN mask ────────────────────────────────────────────────────
    valid = tf.math.logical_not(tf.math.is_nan(y_true))
    y_true = tf.where(valid, y_true, tf.zeros_like(y_true))
    y_pred = tf.where(valid, y_pred, tf.zeros_like(y_pred))
    valid_float = tf.cast(valid, tf.float32)

    # ── depth cap — GT capped at 4 m; pred left uncapped ────────────
    y_true = tf.minimum(y_true, DEPTH_CAP_M)
    y_pred_relu = tf.nn.relu(y_pred)  # non-negative, uncapped

    # ── 5×5 spatial-activity mask ───────────────────────────────────
    flood_bin = tf.cast(y_true > 0.0, tf.float32)[..., tf.newaxis]  # [B,H,W,1]
    active_4d = tf.nn.max_pool2d(
        flood_bin, ksize=5, strides=1, padding="SAME"
    )  # [B,H,W,1]
    active = tf.squeeze(active_4d, axis=-1)  # [B,H,W]

    # Fallback: fully-dry frames use full frame so model still learns zeros.
    any_active = tf.reduce_any(active > 0.5, axis=[1, 2], keepdims=True)
    active = tf.where(any_active, active, tf.ones_like(active))
    effective_mask = valid_float * active

    # ── flood-aware log-depth weighted loss ─────────────────────────
    flood_weight = 1.0 + 10.0 * y_true  # ≥ 1, linear with GT depth

    log_true = tf.math.log1p(y_true)
    log_pred = tf.math.log1p(y_pred_relu)
    log_squared_error = tf.square(log_true - log_pred)

    weighted_log_mse = tf.reduce_sum(
        flood_weight * log_squared_error * effective_mask
    ) / (tf.reduce_sum(effective_mask) + 1e-8)

    # ── peak-depth penalty (log space, weight 2.0) ───────────────────
    valid_f = tf.cast(valid, tf.float32)
    log_pred_max = tf.reduce_max(log_pred * valid_f, axis=[1, 2])
    log_true_max = tf.reduce_max(log_true * valid_f, axis=[1, 2])
    peak_penalty = tf.reduce_mean(tf.square(log_pred_max - log_true_max))

    # ── ABeDice: boundary-aware spatial extent loss (replaces focal) ─
    # ABeDice is a strict superset of focal — it adds boundary weighting on top
    # of the same wet/dry classification objective, so there is no reason to run both.
    abedice = _abedice_loss(y_true, y_pred_relu)

    return weighted_log_mse + 0.5 * peak_penalty + 1.0 * abedice


@register_keras_serializable(package="Custom", name="make_hybrid_loss_v6")
def make_hybrid_loss_v6(y_true, y_pred):
    """V6 loss -- v5 with deep-water emphasis restored.

    15-epoch smoke test of v5 (fixed shallow-water weighting) showed a real
    trade-off: shallow water (0-0.1m) improved ~3x and moderate water
    (0.1-0.5m) improved too, but deep water (1-2m, 2-4m) got WORSE than the
    pretrain baseline in every single one of 15 epochs. Root cause: v5's two
    ramps only differ by ~1.4x between "shallow, already weighted" (0.1m,
    weight~9.2) and "deep" (4m, weight~13) -- far less relative pressure to
    prioritize deep water over shallow water than v4 had (which jumped ~8x
    between 0.4m and 1m+, just starting from an unfairly low baseline for
    shallow water).

    V6 fix: shrink the shallow ramp's ceiling (8->4) and steepen + grow the
    deep ramp's ceiling (4->12, centred later and steeper: sigmoid(5*(y-1))
    instead of sigmoid(3*(y-1))) so there's a clear three-tier structure --
    dry (~1.1) < shallow/moderate (~3-6) < deep (~11-17) -- instead of
    shallow and deep sitting too close together. Shallow water still gets
    meaningfully more weight than dry (fixing the original v4 bug), while
    deep water gets even more emphasis than v4 originally had.
    """
    DEPTH_CAP_M_V6 = 4.0
    if y_true.shape.ndims == 4 and y_true.shape[-1] == 1:
        y_true = y_true[..., 0]
    if y_pred.shape.ndims == 4 and y_pred.shape[-1] == 1:
        y_pred = y_pred[..., 0]

    valid = tf.math.logical_not(tf.math.is_nan(y_true))
    y_true = tf.where(valid, y_true, tf.zeros_like(y_true))
    y_pred = tf.where(valid, y_pred, tf.zeros_like(y_pred))
    valid_float = tf.cast(valid, tf.float32)

    y_true = tf.minimum(y_true, DEPTH_CAP_M_V6)
    y_pred_relu = tf.nn.relu(y_pred)

    flood_bin = tf.cast(y_true > 0.0, tf.float32)[..., tf.newaxis]
    active = tf.squeeze(
        tf.nn.max_pool2d(flood_bin, ksize=5, strides=1, padding="SAME"), axis=-1
    )
    any_active = tf.reduce_any(active > 0.5, axis=[1, 2], keepdims=True)
    active = tf.where(any_active, active, tf.ones_like(active))
    effective_mask = valid_float * active

    depth_weight = (
        1.0
        + 4.0 * tf.sigmoid(100.0 * (y_true - 0.05))
        + 12.0 * tf.sigmoid(5.0 * (y_true - 1.0))
    )
    squared_error = tf.square(y_pred_relu - y_true)
    weighted_mse = tf.reduce_sum(depth_weight * squared_error * effective_mask) / (
        tf.reduce_sum(effective_mask) + 1e-8
    )

    valid_f = tf.cast(valid, tf.float32)
    pred_max = tf.reduce_max(y_pred_relu * valid_f, axis=[1, 2])
    true_max = tf.reduce_max(y_true * valid_f, axis=[1, 2])
    peak_penalty = tf.reduce_mean(tf.square(pred_max - true_max))

    focal = _flood_focal_loss(y_true, y_pred_relu)

    return weighted_mse + 0.2 * peak_penalty + 0.3 * focal


def _v11_variant_body(y_true, y_pred, false_alarm_weight):
    """Shared body for the v11 false_alarm_weight calibration sweep.

    v11 itself (false_alarm_weight=8.0) proved the mechanism works -- real-
    AR false-alarm rate dropped substantially on both cities (Atlanta
    ~0.70->0.48, Manhattan ~0.63->0.41) -- but overshot: CSI and dangerous-
    CSI both fell hard (Manhattan CSI 0.175->0.060). These variants sweep
    false_alarm_weight between v10's untuned 1.0 and v11's 8.0 to find a
    point that keeps more of the false-alarm win without giving back this
    much detection accuracy -- same calibration pattern as the v8/v9/v10
    flooded_weight sweep and the gravity_loss_weight 1/3/5 sweep.
    """
    DEPTH_CAP_M_V11 = 4.0
    if y_true.shape.ndims == 4 and y_true.shape[-1] == 1:
        y_true = y_true[..., 0]
    if y_pred.shape.ndims == 4 and y_pred.shape[-1] == 1:
        y_pred = y_pred[..., 0]

    valid = tf.math.logical_not(tf.math.is_nan(y_true))
    y_true = tf.where(valid, y_true, tf.zeros_like(y_true))
    y_pred = tf.where(valid, y_pred, tf.zeros_like(y_pred))
    valid_float = tf.cast(valid, tf.float32)

    y_true = tf.minimum(y_true, DEPTH_CAP_M_V11)
    y_pred_relu = tf.nn.relu(y_pred)

    flood_bin = tf.cast(y_true > 0.0, tf.float32)[..., tf.newaxis]
    active = tf.squeeze(
        tf.nn.max_pool2d(flood_bin, ksize=5, strides=1, padding="SAME"), axis=-1
    )
    any_active = tf.reduce_any(active > 0.5, axis=[1, 2], keepdims=True)
    active = tf.where(any_active, active, tf.ones_like(active))
    effective_mask = valid_float * active

    depth_weight = (
        1.0
        + 4.0 * tf.sigmoid(30.0 * (y_true - 0.05))
        + 12.0 * tf.sigmoid(5.0 * (y_true - 1.0))
    )
    squared_error = tf.square(y_pred_relu - y_true)
    weighted_mse = tf.reduce_sum(depth_weight * squared_error * effective_mask) / (
        tf.reduce_sum(effective_mask) + 1e-8
    )

    valid_f = tf.cast(valid, tf.float32)
    pred_max = tf.reduce_max(y_pred_relu * valid_f, axis=[1, 2])
    true_max = tf.reduce_max(y_true * valid_f, axis=[1, 2])
    peak_penalty = tf.reduce_mean(tf.square(pred_max - true_max))

    focal = _flood_focal_loss(
        y_true, y_pred_relu, flooded_weight=5.0, false_alarm_weight=false_alarm_weight
    )

    return weighted_mse + 0.2 * peak_penalty + 0.6 * focal


@register_keras_serializable(package="Custom", name="make_hybrid_loss_v11_fa2")
def make_hybrid_loss_v11_fa2(y_true, y_pred):
    """V11 sweep point: false_alarm_weight=2.0 (was 8.0 in v11, 1.0 in v10)."""
    return _v11_variant_body(y_true, y_pred, false_alarm_weight=2.0)


def _soft_band_csi_loss(y_true, y_pred, bands, sharpness=40.0):
    """Differentiable surrogate for CSI within each (lo, hi) depth band.

    MSE-style losses have no notion of "landed in the right depth band" --
    they only see continuous error magnitude. This term does, via a soft
    (sigmoid-product) relaxation of band membership instead of a hard
    threshold, so it's differentiable and can be optimized directly, the
    same way soft-IoU/soft-Dice losses are used in segmentation when the
    eval metric itself is a thresholded overlap score. Returns 1 - mean(soft
    CSI) across `bands`, i.e. 0 when every true/predicted pixel pair agrees
    on which band it's in.

    y_true/y_pred must already be same-shape, NaN-free, and depth (not
    logits) -- caller's job, so this stays a pure, reusable building block.
    """
    def soft_band(depth, lo, hi):
        lower = tf.sigmoid(sharpness * (depth - lo))
        return lower if hi is None else lower * tf.sigmoid(sharpness * (hi - depth))

    losses = []
    for lo, hi in bands:
        pred_in = soft_band(y_pred, lo, hi)
        true_in = soft_band(y_true, lo, hi)
        tp = tf.reduce_sum(pred_in * true_in)
        fp = tf.reduce_sum(pred_in * (1.0 - true_in))
        fn = tf.reduce_sum((1.0 - pred_in) * true_in)
        soft_csi = tp / (tp + fp + fn + 1e-6)
        losses.append(1.0 - soft_csi)
    return tf.reduce_mean(tf.stack(losses))


@register_keras_serializable(package="Custom", name="make_hybrid_loss_v12_bandcsi")
def make_hybrid_loss_v12_bandcsi(y_true, y_pred):
    """v11_fa2 base + an explicit soft-CSI surrogate for the low (0.15-0.30m)
    and med (0.30-0.60m) depth bands -- the two evaluation metrics v11_fa2's
    depth_weight term doesn't specifically target (it upweights ANY flooded
    pixel and >1m pixels, but nothing rewards correct band classification in
    between). Added 2026-10-21 after diagnosing that low_csi/med_csi (7-17%
    across every model tried on this project) are bottlenecked by depth
    PRECISION, not training amount: typical MAE (0.1-0.26m) is comparable to
    or larger than these bands' own width (0.15m, 0.30m), so a close-but-
    imprecise prediction lands in the wrong bin and scores as a miss on both
    sides. This is an UNTESTED hypothesis -- smoke-test the actual training
    effect before trusting it changes anything real.
    """
    y_true_sq = y_true[..., 0] if y_true.shape.ndims == 4 and y_true.shape[-1] == 1 else y_true
    y_pred_sq = y_pred[..., 0] if y_pred.shape.ndims == 4 and y_pred.shape[-1] == 1 else y_pred

    valid = tf.math.logical_not(tf.math.is_nan(y_true_sq))
    y_true_safe = tf.where(valid, y_true_sq, tf.zeros_like(y_true_sq))
    y_pred_relu = tf.nn.relu(y_pred_sq)
    # Invalid pixels get their prediction zeroed too -- otherwise a stray
    # high prediction on a masked-out pixel could still fall in-band and
    # pollute the soft-CSI sums despite contributing nothing to `base`.
    y_pred_safe = tf.where(valid, y_pred_relu, tf.zeros_like(y_pred_relu))

    base = _v11_variant_body(y_true, y_pred, false_alarm_weight=2.0)
    band_csi_loss = _soft_band_csi_loss(
        y_true_safe, y_pred_safe, bands=[(0.15, 0.30), (0.30, 0.60)]
    )
    return base + 2.0 * band_csi_loss


def gravity_consistency_loss(pred: tf.Tensor, geospatial: tf.Tensor) -> tf.Tensor:
    """Penalise isolated water-surface-elevation peaks unsupported by terrain.

    water_surface_elevation (WSE) = terrain elevation (ch0) + predicted depth.
    Physically, an isolated local MAXIMUM in WSE (higher than every one of
    its 8 neighbours) with no real terrain depression underneath it is
    "water appearing from nowhere" -- exactly the failure mode this loss
    targets. Real depressions (ch9, the DEM-sink channel already computed by
    the data pipeline) are exempted, since ponding in a genuine local low
    point is physically correct, not a hallucination.

    Args:
        pred:       (B, H, W, 1) or (B, H, W) predicted flood depth (m),
            AFTER Green-Ampt/flow-routing corrections -- penalise what
            actually gets fed forward and evaluated, same as mass_loss.
        geospatial: (B, H, W, 12) static features.

    Returns:
        Scalar penalty (>= 0).
    """
    if pred.shape.ndims == 3:
        pred = pred[..., tf.newaxis]
    depth = tf.nn.relu(pred)
    elev = geospatial[:, :, :, 0:1]
    dem_sink = geospatial[:, :, :, 9:10]  # [0,1], already normalised

    wse = elev + depth

    def _shift(t, dy, dx):
        padded = tf.pad(t, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="SYMMETRIC")
        h = tf.shape(t)[1]
        w = tf.shape(t)[2]
        return padded[:, 1 + dy:1 + dy + h, 1 + dx:1 + dx + w, :]

    offsets = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (1, 1), (1, -1)]
    neighbor_wse_max = tf.reduce_max(
        tf.concat([_shift(wse, dy, dx) for dy, dx in offsets], axis=-1),
        axis=-1, keepdims=True,
    )

    # Isolated peak: this pixel's water surface sits above every neighbour's.
    excess = tf.nn.relu(wse - neighbor_wse_max)
    not_sink = 1.0 - dem_sink
    penalty = excess * not_sink

    return tf.reduce_mean(tf.square(penalty))


def infiltration_consistency_loss(
    y_true: tf.Tensor, pred: tf.Tensor, geospatial: tf.Tensor
) -> tf.Tensor:
    """Penalises false-positive flood predictions on high-infiltration soil.

    A false-positive-channel analysis (comparing the geospatial-channel
    values at false-positive vs. true-positive vs. true-negative pixels)
    found that false positives sit on measurably higher hydraulic
    conductivity (ch4) and porosity (ch6) than true positives -- the model
    is over-predicting flooding onto ground that should be absorbing water,
    despite that data already being in its input. Nothing previously
    penalised ignoring it. This loss does NOT penalise true positives in
    high-infiltration areas (infiltration capacity makes flooding less
    likely, not impossible) -- only predicted-wet-but-actually-dry pixels.

    Args:
        y_true:     (B, H, W, 1) or (B, H, W) ground-truth depth (m) for
            this step.
        pred:       (B, H, W, 1) or (B, H, W) predicted flood depth (m),
            AFTER Green-Ampt/flow-routing corrections -- same convention as
            gravity_consistency_loss.
        geospatial: (B, H, W, 12) static features.

    Returns:
        Scalar penalty (>= 0).
    """
    if pred.shape.ndims == 3:
        pred = pred[..., tf.newaxis]
    if y_true.shape.ndims == 3:
        y_true = y_true[..., tf.newaxis]
    depth = tf.nn.relu(pred)

    hydraulic_conductivity = geospatial[:, :, :, 4:5]  # already normalised [0, 1]
    porosity = geospatial[:, :, :, 6:7]  # already normalised [0, 1]
    infiltration_capacity = (hydraulic_conductivity + porosity) / 2.0

    is_false_positive = tf.cast(
        tf.logical_and(depth > 0.05, y_true <= 0.05), tf.float32
    )
    penalty = depth * infiltration_capacity * is_false_positive

    return tf.reduce_mean(penalty)


# ---------------------------------------------------------------------------
# Soft arrival-time head
# ---------------------------------------------------------------------------
# Why: training only on per-frame depth does not directly optimise *when*
# flooding first reaches each pixel.  Two predictions with the same average
# depth error can have very different arrival-time errors.  This auxiliary loss
# directly penalises timing mistakes.
#
# How (one-sentence version for the team):
#   We turn "first time depth > threshold" into a smooth, differentiable
#   quantity using a sigmoid + cumulative-product trick, so we can
#   backpropagate through it.
# ---------------------------------------------------------------------------


def _compute_soft_arrival_time(
    depth_sequence: tf.Tensor,
    threshold: float = 0.05,
    alpha: float = 50.0,
) -> tf.Tensor:
    """Differentiable soft arrival-time map from a predicted depth sequence.

    Formulation (per-pixel, subscript t = time):
        p_t  = sigmoid(α (h_t − threshold))       # soft "wet" probability
        q_t  = p_t · ∏_{k<t} (1 − p_k)           # soft first-arrival PMF
        τ    = Σ_t  t · q_t                        # expected arrival time

    Pixels that never exceed the threshold are assigned τ = T.

    Args:
        depth_sequence: [B, T, H, W] (or [B, T, H, W, 1]) predicted depth.
        threshold: Depth in metres counted as "flooded" (default 5 cm).
        alpha: Sigmoid sharpness.  α=50 gives a transition width of ~0.04 m.

    Returns:
        [B, H, W] soft arrival-time map in timestep units.
    """
    if len(depth_sequence.shape) == 5:
        depth_sequence = tf.squeeze(depth_sequence, axis=-1)

    T = depth_sequence.shape[1]

    # Soft wet indicator at every timestep  [B, T, H, W]
    p = tf.sigmoid(alpha * (depth_sequence - threshold))

    # Survival function: ∏_{k<t}(1-p_k), computed in log-space for stability
    log_survival = tf.cumsum(tf.math.log1p(-p + 1e-8), axis=1, exclusive=True)
    survival = tf.exp(log_survival)

    # Soft first-arrival PMF
    q = p * survival  # [B, T, H, W]
    q_sum = tf.reduce_sum(q, axis=1)  # [B, H, W]

    # Expected arrival time
    t_idx = tf.cast(tf.range(T), tf.float32)[
        tf.newaxis, :, tf.newaxis, tf.newaxis
    ]  # [1, T, 1, 1]
    tau = tf.reduce_sum(t_idx * q, axis=1) / (q_sum + 1e-8)

    # Never-wet pixels → τ = T
    tau = tf.where(
        q_sum < 1e-6,
        tf.cast(T, tf.float32) * tf.ones_like(tau),
        tau,
    )
    return tau


def _ground_truth_arrival_time(
    depth_labels: tf.Tensor,
    threshold: float = 0.05,
) -> tf.Tensor:
    """Hard (non-differentiable) arrival-time map from ground-truth labels.

    Args:
        depth_labels: [B, T, H, W] ground-truth depth sequence.
        threshold: Same depth threshold used in the soft version.

    Returns:
        [B, H, W] arrival time (timestep index).  T where pixel never floods.
    """
    T = depth_labels.shape[1]
    wet = tf.cast(depth_labels > threshold, tf.float32)
    ever_wet = tf.reduce_any(wet > 0.5, axis=1)  # [B, H, W]
    first_wet = tf.cast(tf.argmax(wet, axis=1), tf.float32)  # [B, H, W]
    return tf.where(
        ever_wet, first_wet, tf.cast(T, tf.float32) * tf.ones_like(first_wet)
    )


def arrival_time_loss(
    depth_pred: tf.Tensor,
    depth_true: tf.Tensor,
    threshold: float = 0.05,
    alpha: float = 50.0,
) -> tf.Tensor:
    """Differentiable arrival-time auxiliary loss.

    Compares the soft predicted arrival time against the hard ground-truth
    arrival time using element-wise Huber loss (δ = 2 timesteps).  Only
    pixels that actually flood in the ground truth contribute to the loss.

    Typical usage inside a ``tf.GradientTape`` block::

        with tf.GradientTape() as tape:
            depth_seq = model.call_n(inputs, n=T)         # [B, T, H, W]
            depth_loss = make_hybrid_loss(labels, depth_seq[:, -1])
            arr_loss   = arrival_time_loss(depth_seq, labels_full_seq)
            total      = depth_loss + 0.1 * arr_loss      # weight as needed

    Args:
        depth_pred: [B, T, H, W] predicted depth sequence (from ``call_n``).
        depth_true: [B, T, H, W] ground-truth depth sequence.
        threshold: Flood-arrival threshold (m).
        alpha: Sigmoid sharpness for the soft arrival computation.

    Returns:
        Scalar Huber loss over flooded pixels.
    """
    tau_pred = _compute_soft_arrival_time(depth_pred, threshold, alpha)
    tau_true = _ground_truth_arrival_time(depth_true, threshold)

    # Restrict to pixels that actually flood in ground truth
    floods = tf.reduce_any(depth_true > threshold, axis=1)  # [B, H, W]
    mask = tf.cast(floods, tf.float32)
    n_floods = tf.reduce_sum(mask) + 1e-8

    # Element-wise Huber (δ = 2 timesteps)
    diff = tf.abs(tau_pred - tau_true)
    delta = 2.0
    huber = tf.where(
        diff <= delta,
        0.5 * tf.square(diff),
        delta * (diff - 0.5 * delta),
    )
    return tf.reduce_sum(huber * mask) / n_floods


# ---------------------------------------------------------------------------
# Storage-consistency regulariser
# ---------------------------------------------------------------------------
# Why: in autoregressive rollout, small per-step errors compound.  A common
# failure mode is "teleporting water" — depth appearing at pixels with no
# corresponding rainfall.  This regulariser catches that.
#
# How (one sentence for the team):
#   We check that the total water in the domain doesn't increase faster
#   than rainfall can fill it; violations are softly penalised.
# ---------------------------------------------------------------------------


def storage_consistency_loss(
    depth_pred: tf.Tensor,
    rainfall_per_step: tf.Tensor,
) -> tf.Tensor:
    """Soft physics regulariser: storage increase must not exceed rainfall.

    Constraint (one-sided):
        S_{t+1} − S_t  ≤  R_t        (per batch element)
    where S_t = mean depth across the spatial domain and R_t = rainfall depth
    at step t.  Water *leaving* the domain (infiltration, runoff exit) is
    allowed; only water *appearing from nowhere* is penalised.

    This is intentionally coarse — it operates on domain-mean storage, not
    per-cell fluxes — so it is cheap and does not require knowledge of the
    drainage network.

    Args:
        depth_pred: [B, T, H, W] predicted depth sequence (from ``call_n``).
        rainfall_per_step: [B, T] raw rainfall depth (m) per timestep.
            This is column 0 of the engineered temporal tensor.

    Returns:
        Scalar regularisation loss (≥ 0).
    """
    # Domain-mean depth at each timestep  [B, T]
    mean_depth = tf.reduce_mean(depth_pred, axis=(2, 3))

    # Storage change  [B, T-1]
    delta_S = mean_depth[:, 1:] - mean_depth[:, :-1]

    # Rainfall input at the *current* step  [B, T-1]
    rain = rainfall_per_step[:, :-1]

    # Penalise only positive violations (water from nowhere)
    violation = tf.nn.relu(delta_S - rain)
    return tf.reduce_mean(violation)
