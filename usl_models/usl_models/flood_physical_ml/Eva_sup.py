#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
tf.random.set_seed(42)

MAX_FLOOD = 4

def masked_mae(prediction: tf.Tensor, label: tf.Tensor, mask_thred: list) -> tf.Tensor:
    """Calculates masked spatial MAE
    Args:
        prediction: [*, H, W] prediction tensor.
        label: [*, H, W] label tensor.
        mask_thred: [lower, upper], if upper == False, the CSI of 
        all flooding areas will be calculated

    Returns:
        a float of the masked MAE.
    """
    assert prediction.shape == label.shape
    
    prediction = prediction if tf.is_tensor(prediction) else tf.convert_to_tensor(prediction)
    label = label if tf.is_tensor(label) else tf.convert_to_tensor(label)

    lower = mask_thred[0]
    upper = mask_thred[1]

    if not upper:
        mask = label > lower
    else:
        mask = tf.math.logical_and(label > lower, label <= upper)

    if not tf.reduce_any(mask):
        return None

    absolute_error = tf.abs(prediction[mask] - label[mask])
    masked_mae = tf.reduce_mean(absolute_error)

    return float(masked_mae.numpy())

def masked_CSI(prediction: tf.Tensor, label: tf.Tensor, mask_thred: list) -> float:
    """Caculates masked Critical Success Index (CSI)
    Args:
        prediction: [*, H, W] prediction tensor.
        label: [*, H, W] label tensor.
        mask_thred: [lower, upper], if upper == False, the CSI of 
        all flooding areas will be calculated

    Returns:
        a float of the masked CSI.
    """
    assert prediction.shape == label.shape
    prediction = prediction if tf.is_tensor(prediction) else tf.convert_to_tensor(prediction)
    label = label if tf.is_tensor(label) else tf.convert_to_tensor(label)

    # convert all the prediction and labels into [n, H, W]
    H = label.shape[-2]
    W = label.shape[-1]
    prediction = tf.reshape(prediction, [-1, H, W])
    label = tf.reshape(label, [-1, H, W])

    lower = mask_thred[0]
    upper = mask_thred[1]

    # get masks for prediction and label
    if not upper:
        prediction_mask = prediction > lower
        label_mask = label > lower
    else:
        prediction_mask = tf.math.logical_and(prediction > lower, prediction <= upper)
        label_mask = tf.math.logical_and(label > lower, label <= upper)
    correct_mask = tf.math.logical_and(prediction_mask, label_mask)

    # calculate CSI = TP / (TP + FP + FN) for each map
    prediction_sum = tf.reduce_sum(tf.cast(prediction_mask, tf.int32), axis=[-1, -2])
    label_sum = tf.reduce_sum(tf.cast(label_mask, tf.int32), axis=[-1, -2])
    correct_sum = tf.reduce_sum(tf.cast(correct_mask, tf.int32), axis=[-1, -2])

    # remove maps with not prediction and labels in the mask thred
    csi_mask = (prediction_sum + label_sum) > 0
    if not tf.reduce_any(csi_mask):
        return None
    CSI = correct_sum[csi_mask] / (prediction_sum[csi_mask] + label_sum[csi_mask] - correct_sum[csi_mask])

    # calculate mean CSI for all predicted maps
    mean_CSI = tf.reduce_mean(CSI)

    return float(mean_CSI.numpy())

def masked_NSE(prediction: tf.Tensor, label: tf.Tensor, mask_thred: list) -> float:
    """Caculates masked Critical Success Index (CSI)
    Args:
        prediction: [*, H, W] prediction tensor.
        label: [*, H, W] label tensor.
        mask_thred: [lower, upper], if upper == False, the CSI of 
        all flooding areas will be calculated

    Returns:
        a float of the masked NSE.
    """
    eps = 1e-7
    assert prediction.shape == label.shape
    prediction = prediction if tf.is_tensor(prediction) else tf.convert_to_tensor(prediction)
    label = label if tf.is_tensor(label) else tf.convert_to_tensor(label)
    
    # convert all the prediction and labels into [n, H, W]
    H = label.shape[-2]
    W = label.shape[-1]
    prediction = tf.reshape(prediction, [-1, H, W])
    label = tf.reshape(label, [-1, H, W])

    lower = mask_thred[0]
    upper = mask_thred[1]

    if not upper:
        mask = label > lower
    else:
        mask = tf.math.logical_and(label > lower, label <= upper)
        
    # if no masked areas, return None
    if not tf.reduce_any(mask):
        return None
    
    mask = tf.cast(mask, label.dtype)

    obs_mean = tf.reduce_sum(label * mask, axis=(-1, -2), keepdims=True) / (tf.reduce_sum(mask, axis=(-1, -2), keepdims=True) + eps)

    mse = tf.reduce_sum((label - prediction) ** 2 * mask, axis=(-1, -2)) # (n,)
    ssd = tf.reduce_sum((label - obs_mean) ** 2 * mask, axis=(-1, -2)) # (n,)

    # just calculate NSE for maps with masked areas
    NSE_mask = tf.reduce_sum(mask, axis=(-1, -2)) > 0 # (n,)

    NSE = 1 - mse[NSE_mask]/(ssd[NSE_mask] + eps)

    mean_NSE = tf.reduce_mean(NSE)

    return float(mean_NSE.numpy())

def percentail_peak(prediction: tf.Tensor, label: tf.Tensor, percentail: int):
    """Get top k mean value pairs
    Args:
        prediction: [*, H, W] prediction tensor.
        label: [*, H, W] label tensor.
        mask_thred: [lower, upper], if upper == False, the CSI of 
        all flooding areas will be calculated

    Returns:
        a tensor of the mean value of top k prediction pixels (*), e.g., (B * T,) or (B,).
        a tensor of the mean value of top k label pixels (*), e.g., (B * T,) or (B,).
    """
    assert prediction.shape == label.shape
    prediction = prediction if tf.is_tensor(prediction) else tf.convert_to_tensor(prediction)
    label = label if tf.is_tensor(label) else tf.convert_to_tensor(label)

    H = label.shape[-2]
    W = label.shape[-1]

    prediction = tf.reshape(prediction, [-1, H * W])
    label = tf.reshape(label, [-1, H * W])

    k = int(H * W * percentail / 100)

    top_k_prediction = tf.math.top_k(prediction, k=k).values
    top_k_label = tf.math.top_k(label, k=k).values

    top_k_prediction_mean = tf.reduce_mean(top_k_prediction, axis = -1)
    top_k_label_mean = tf.reduce_mean(top_k_label, axis = -1)

    return top_k_prediction_mean.numpy(), top_k_label_mean.numpy()

def masked_SSIM(prediction: tf.Tensor, label: tf.Tensor, mask_thred: list) -> tf.Tensor:
    """Caculates masked Critical Success Index (CSI)
    Args:
        prediction: [*, H, W] prediction tensor.
        label: [*, H, W] label tensor.
        mask_thred: [lower, upper], if upper == False, the CSI of 
        all flooding areas will be calculated

    Returns:
        a float of the masked SSIM.
    """
    eps = 1e-7
    assert prediction.shape == label.shape
    prediction = prediction if tf.is_tensor(prediction) else tf.convert_to_tensor(prediction)
    label = label if tf.is_tensor(label) else tf.convert_to_tensor(label)

    H = label.shape[-2]
    W = label.shape[-1]

    lower = mask_thred[0]
    upper = mask_thred[1]

    if not upper:
        mask = label > lower
    else:
        mask = tf.math.logical_and(label > lower, label <= upper)
    # return None if no masked areas
    if not tf.reduce_any(mask):
        return None

    mask = tf.cast(mask, label.dtype)

    prediction = tf.reshape(prediction, [-1, H, W, 1])
    label = tf.reshape(label, [-1, H, W, 1])
    mask = tf.reshape(mask, [-1, H, W])

    filter_size = 11
    padding_size = filter_size // 2

    paddings = [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]]

    padded_prediction = tf.pad(prediction, paddings, mode="REFLECT")
    padded_label = tf.pad(label, paddings, mode="REFLECT")

    ssim_map = tf.image.ssim(padded_prediction, 
                             padded_label, 
                             max_val=MAX_FLOOD, 
                             filter_size = filter_size, 
                             return_index_map=True) # (n, H, W)

    masked_ssim = tf.reduce_sum(ssim_map * mask, axis=(-1, -2)) / (tf.reduce_sum(mask, axis=(-1, -2)) + eps) # (n, )
    ssim_masks = tf.reduce_sum(mask, axis=(-1, -2)) > 0
    masked_ssim = masked_ssim[ssim_masks]

    return float(tf.reduce_mean(masked_ssim).numpy())