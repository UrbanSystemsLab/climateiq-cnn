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

    Args:
        y_true:       [B, H, W] GT depth, already capped at DEPTH_CAP_M.
        y_pred_relu:  [B, H, W] predicted depth, relu'd, uncapped.
        threshold:    Flood detection threshold in metres.
        gamma:        Focal loss concentration parameter.
        flooded_weight: Extra weight on missed floods (false negatives).

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
        + (1.0 - flood_gt) * tf.math.log(1.0 - flood_prob + eps)
    )

    # Focal weight: hard examples (wrong) get higher weight
    p_t = flood_gt * flood_prob + (1.0 - flood_gt) * (1.0 - flood_prob)
    focal = tf.pow(1.0 - p_t, gamma) * bce

    return tf.reduce_mean(focal)


@register_keras_serializable(package="Custom", name="make_hybrid_loss")
def make_hybrid_loss(y_true, y_pred):
    """Flood-aware log-depth loss + spatial flood-detection focal loss.

    Key design decisions:

    1. Depth cap at 4 m — raised from 2.5 m to cover Atlanta/Phoenix peak
       events.  Manhattan outliers >4 m are still excluded.

    2. Log-depth error — error is computed as
           (log(1+pred) − log(1+true))²
       instead of (pred − true)².  In log space a 0.03 m vs 0.10 m error
       gets equal gradient weight as a 0.5 m vs 1.6 m error, fixing the
       training-distribution bias where 99 % of pixels are <0.2 m.

    3. Flood weight — w = 1 + 10·y_true on raw (pre-log) depths keeps deeper
       pixels upweighted relative to shallow ones.

    4. 5×5 spatial-activity mask — excludes far-interior dry pixels while
       preserving a boundary ring of dry context around every flood edge.
       Fully-dry frames fall back to the full frame.

    5. Peak-depth penalty (weight 2.0) — directly penalises underestimating
       the spatial maximum depth per sample, in log space.

    6. Flood-detection focal loss (weight 0.5) — binary cross-entropy on
       flooded/dry classification with focal weighting and 10× asymmetric
       weight on missed floods.  Directly penalises predicting flooding in
       the wrong spatial locations — the dominant failure mode where corr≈0
       despite low MAE (dry pixels dominate the depth error metric).

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

    # ── flood-detection focal loss (weight 0.5) ──────────────────────
    focal = _flood_focal_loss(y_true, y_pred_relu)

    return weighted_log_mse + 0.5 * peak_penalty + 0.5 * focal


@register_keras_serializable(package="Custom", name="make_hybrid_loss_v3")
def make_hybrid_loss_v3(y_true, y_pred):
    """V3 loss — LINEAR MSE weighted by sqrt(depth), NOT log-depth.

    The v1/v2 losses used log(1+depth)² which *penalises shallow errors more
    than deep errors* — e.g. 5cm error on 10cm depth scores worse than 1m error
    on 3m depth. This causes the model to systematically underpredict deep floods.

    V3 design:
    - Linear MSE in metres (not log-space) — error of 1m on 3m depth now dominates.
    - Weight = √(y_true + 1) + 10·(y_true > 0.5) — deeper pixels get stronger
      gradients, so the model is forced to learn full depth range.
    - Peak penalty in linear space — penalises pred_max < GT_max directly.
    - Focal at 0.3 weight — still reduces false positives but lets MSE dominate.

    Expected impact: pred_max should track GT_max (instead of capping at ~2.4m),
    closing the 26-pt gap on Metric A by matching deep-water predictions.
    """
    DEPTH_CAP_M_V3 = 4.0
    if y_true.shape.ndims == 4 and y_true.shape[-1] == 1:
        y_true = y_true[..., 0]
    if y_pred.shape.ndims == 4 and y_pred.shape[-1] == 1:
        y_pred = y_pred[..., 0]

    valid = tf.math.logical_not(tf.math.is_nan(y_true))
    y_true = tf.where(valid, y_true, tf.zeros_like(y_true))
    y_pred = tf.where(valid, y_pred, tf.zeros_like(y_pred))
    valid_float = tf.cast(valid, tf.float32)

    y_true = tf.minimum(y_true, DEPTH_CAP_M_V3)
    y_pred_relu = tf.nn.relu(y_pred)

    # 5×5 spatial-activity mask around flooded pixels (keep dry context ring)
    flood_bin = tf.cast(y_true > 0.0, tf.float32)[..., tf.newaxis]
    active = tf.squeeze(
        tf.nn.max_pool2d(flood_bin, ksize=5, strides=1, padding="SAME"), axis=-1
    )
    any_active = tf.reduce_any(active > 0.5, axis=[1, 2], keepdims=True)
    active = tf.where(any_active, active, tf.ones_like(active))
    effective_mask = valid_float * active

    # Depth-weighted linear MSE: deep pixels contribute MORE, not less
    depth_weight = tf.sqrt(y_true + 1.0) + 10.0 * tf.cast(y_true > 0.5, tf.float32)
    squared_error = tf.square(y_pred_relu - y_true)
    weighted_mse = tf.reduce_sum(
        depth_weight * squared_error * effective_mask
    ) / (tf.reduce_sum(effective_mask) + 1e-8)

    # Peak penalty in LINEAR space (not log) — forces pred_max → GT_max
    valid_f = tf.cast(valid, tf.float32)
    pred_max = tf.reduce_max(y_pred_relu * valid_f, axis=[1, 2])
    true_max = tf.reduce_max(y_true * valid_f, axis=[1, 2])
    peak_penalty = tf.reduce_mean(tf.square(pred_max - true_max))

    # Focal loss for flood/dry classification (lower weight vs v1: 0.3 instead of 0.5)
    focal = _flood_focal_loss(y_true, y_pred_relu)

    return weighted_mse + 1.0 * peak_penalty + 0.3 * focal


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
