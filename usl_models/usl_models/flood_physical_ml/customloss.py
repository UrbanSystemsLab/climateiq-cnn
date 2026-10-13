import os
import numpy as np

import tensorflow as tf
from keras.saving import register_keras_serializable
from usl_models.flood_physical_ml.constants import *
from keras import layers
from usl_models.flood_physical_ml.customloss_old import make_hybrid_loss

def _compute_advective_acceleration(padded_v_sequence):
    u = padded_v_sequence[..., 0:1] # [B, V_M_STEPS+2, H+2, W+2, 1]
    v = padded_v_sequence[..., 1:2] # [B, V_M_STEPS+2, H+2, W+2, 1]
    d_u_d_x = (u[:, :, 1:-1, 2:] - u[:, :, 1:-1, 0:-2])/(2*RESOLUTION) #[B, V_M_STEPS+2, H, W, 1]
    d_u_d_y = (u[:, :, 2:, 1:-1] - u[:, :, 0:-2, 1:-1])/(2*RESOLUTION) #[B, V_M_STEPS+2, H, W, 1]
    d_v_d_x = (v[:, :, 1:-1, 2:] - v[:, :, 1:-1, 0:-2])/(2*RESOLUTION) #[B, V_M_STEPS+2, H, W, 1]
    d_v_d_y = (v[:, :, 2:, 1:-1] - v[:, :, 0:-2, 1:-1])/(2*RESOLUTION) #[B, V_M_STEPS+2, H, W, 1]

    acceleration_u = u[:, :, 1:-1, 1:-1] * d_u_d_x + v[:, :, 1:-1, 1:-1] * d_u_d_y #[B, V_M_STEPS+2, H, W, 1]
    acceleration_v = u[:, :, 1:-1, 1:-1] * d_v_d_x + v[:, :, 1:-1, 1:-1] * d_v_d_y #[B, V_M_STEPS+2, H, W, 1]

    return tf.concat([acceleration_u, acceleration_v], axis=-1) # [B, V_M_STEPS+2, H, W, 1]

def momentum_loss(
    velocity_sequence: tf.Tensor,
    depth_sequence: tf.Tensor,
    A: tf.Tensor,
    B: tf.Tensor,
    DEM: tf.Tensor,
    Building: tf.Tensor,
    N: tf.Tensor,
    Valid_Mask = None
    ):
    """
    Args:
        velocity_sequence: Velocity output of the model [B, V_M_STEPS+2, H, W, 2], with [:, 0] as the predicted initial velocity, [:, 1: V_M_STEPS+1] as the predicted middle velocities, and [:, V_M_STEPS+1] as the predicted velocity of next time step;
        depth_sequence: The flood depth output of the model [B, V_M_STEPS+2, H, W, 1], with [:, 0:V_M_STEPS] as the calculated flooding depths at middle steps, and [:, -1] as the predicted flooding depth of next time step;
        A: [V_M_STEPS, V_M_STEPS]
        B: [1, V_M_STEPS]
        DEM_map: The DEM [B, H, W, 1]
        Building: the Bulding heights [B, H, W, 1]
        N: [B, H, W, 1]
        Valid_Mask: [B, H, W, 1]
    """
    nan_mask = tf.math.is_nan(velocity_sequence)
    # tf.print("Total number of predicted nan is: ", tf.reduce_sum(tf.cast(nan_mask, tf.int32)))

    nan_mask = tf.math.is_nan(DEM)
    # tf.print("Total number of DEM nan is: ", tf.reduce_sum(tf.cast(nan_mask, tf.int32)))

    velocity_0 = velocity_sequence[:, 0][:, tf.newaxis] # [B, 1, H, W, 2], predicted initial velocity
    velocity_1 = velocity_sequence[:, -1][:, tf.newaxis] # [B, 1, H, W, 2], predicted velocity of next time step

    elevation = DEM + Building # [B, H, W, 1]
    elevation = elevation[:, tf.newaxis] # [B, 1, H, W, 1]

    water_level = depth_sequence + elevation # [B, V_M_STEPS+2, H, W, 1]

    # tf.print("water depth ave is: ", tf.reduce_mean(depth_sequence))

    ############################################################
    #####################Calculate Momentum#####################
    ############################################################

    #########################Water Level Diff###################
    # x,  x, x
    # x, -1, 1
    # x,  x, x

    # x,  x, x
    # x, -1, x
    # x,  1, x
    water_level_padded = tf.pad(water_level, [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]], mode = "REFLECT")

    water_surface_differences_u = water_level_padded[:, :, 1:-1, 2:] - water_level_padded[:, :, 1:-1, 1:-1] # [B, V_M_STEPS+2, H, W, 1] <
    water_surface_differences_v = water_level_padded[:, :, 2:, 1:-1] - water_level_padded[:, :, 1:-1, 1:-1] # [B, W_M_STEPS+2, H, W, 1] ^
    water_surface_differences = tf.concat([water_surface_differences_u, water_surface_differences_v], axis=-1) # [B, V_M_STEPS+2, H, W, 2] two directions < and ^

    water_surface_gradient = water_surface_differences/RESOLUTION # [B, V_M_STEPS+2, H, W, 2]

    water_surface_momentum = GRAVITY * water_surface_gradient # [B, V_M_STEPS+2, H, W, 2]

    # tf.print("water surface momentum ave is: ", tf.reduce_mean(water_surface_momentum))

    #################################Fraction momentum####################
    N = N[:, tf.newaxis] # [B, 1, H, W, 1]
    velocity_norm = tf.reduce_sum(tf.square(velocity_sequence), axis=-1, keepdims=True)
    velocity_norm = tf.sqrt(velocity_norm + EPS) #[B, V_M_STEPS+2, H, W, 1]

    friction = N**2*GRAVITY/depth_sequence**(4/3)*velocity_norm*velocity_sequence #[B, V_M_STEPS+2, H, W, 2]
    friction = tf.where(depth_sequence > EPS_WATER, friction, 0.0) #[B, V_M_STEPS+2, H, W, 2]
    #################################total momentum#######################
    momentum = -1 * (water_surface_momentum + friction) #[B, V_M_STEPS+2, H, W, 2]

    ######################################################################
    ########################Local Acceleration####################
    ######################################################################
    #steps
    acceleration = (velocity_sequence[:, 1:-1] - velocity_sequence[:, 0:-2])/STEP_LENGTH # [B, V_M_STEPS, H, W, 2]
    step_acceleration = tf.einsum('bvhwd, vk->bkhwd', acceleration, tf.transpose(A)) # [B, V_M_STEPS, H, W, 2]

    # final
    final_acceleration = tf.einsum('bvhwd, vk->bkhwd', acceleration, tf.transpose(B)) # [B, 1, H, W, 2]

    # all
    local_acceleration = tf.concat([step_acceleration, final_acceleration], axis=1) # [B, V_M_STEPS+1, H, W, 2]

    ######################################################################
    ###########################advective acceleration#####################
    ######################################################################
    padded_v_sequence = tf.pad(velocity_sequence[:, 1:], [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]], mode = "REFLECT") # [B, V_M_STEPS+1, H+2, W+2, 2]

    advective_acceleration = _compute_advective_acceleration(padded_v_sequence) # [B, V_M_STEPS+1, H, W, 2]


    ##########################Calculate momentum loss#####################
    ######################################################################
    ######################################################################
    overall_loss = local_acceleration + advective_acceleration + (momentum[:, 1:]) # [B, V_M_STEPS+1, H, W, 2]

    depth_mask = depth_sequence[:, 1:]
    depth_mask = tf.cast(depth_mask>0, dtype=tf.float32) # [B, V_M_STEPS+1, H, W, 2]
    overall_loss = overall_loss * depth_mask
    overall_loss = overall_loss * Valid_Mask[:, tf.newaxis, ...]

    total_mean = tf.reduce_mean(tf.abs(overall_loss))
    
    return total_mean

def mass_loss(
    depth_sequence: tf.Tensor,
    Valid_Mask = None
    ):
    """
    punish flooding depth predictions under zero
    Valid_Mask: [B, H, W, 1]
    """
    # [B, V_M_STEPS+1, H, W, 1]
    vio_physical_prediction = tf.maximum(0.0, -1*depth_sequence[:, 1:])
    mse = tf.reduce_mean(tf.square(vio_physical_prediction * Valid_Mask[:, tf.newaxis]))
    return mse

def depth_loss(
    predicted_depth: tf.Tensor,
    label: tf.Tensor,
    Valid_Mask = None
    ):
    """
    Args:
        predicted_depth: [B, H, W, 1]
        label: [B, H, W, 1]
    """
    loss = make_hybrid_loss(label, predicted_depth)
    
    return loss

def velocity_loss(
    last_final_velocity: tf.Tensor,
    predicted_init_velocity: tf.Tensor,
    Valid_Mask = None
    ):
    
    return tf.reduce_mean(tf.square(last_final_velocity - predicted_init_velocity))
    
@register_keras_serializable(package="Custom", name="physical_loss")
def physical_loss(
    velocity_sequence: tf.Tensor,
    depth_sequence: tf.Tensor,
    A: tf.Tensor,
    B: tf.Tensor,
    DEM: tf.Tensor,
    Building: tf.Tensor,
    N: tf.Tensor,
    label: tf.Tensor,
    last_final_velocity = None,
    Valid_Mask = None
    ):
    momentum_l = momentum_loss(
        velocity_sequence,
        depth_sequence,
        A,
        B,
        DEM,
        Building,
        N,
        Valid_Mask
    )

    mass_l = mass_loss(depth_sequence, Valid_Mask)

    depth_l = depth_loss(depth_sequence[:, -1], label, Valid_Mask)

    # momentum_l = 0

    physical_l = momentum_l + mass_l + depth_l

    return physical_l, momentum_l, mass_l, depth_l


if __name__ == "__main__":
    velocity_sequence = tf.random.normal((2, 12, 1000, 1000, 2))
    depth_sequence = tf.random.normal((2, 11, 1000, 1000, 1)) * 10
    A = tf.random.normal((10, 10))
    B = tf.random.normal((1, 10))
    DEM = tf.random.normal((2, 1000, 1000, 1))
    Building = tf.random.normal((2, 1000, 1000, 1))
    N = tf.random.normal((2, 1000, 1000, 1))
    label = tf.ones((2, 1000, 1000, 1)) * 10
    last_final_velocity = None

    # loss = momentum_loss(
    # velocity_sequence,
    # depth_sequence,
    # A,
    # B,
    # DEM_map,
    # Building,
    # t_b,
    # )
    # predict = tf.random.normal((2, 100, 100, 1)) * 10
    
    # loss = depth_loss(predict, label)

    # loss = velocity_loss(velocity_sequence[:, 0], velocity_sequence[:, 1])
    loss = physical_loss(
        velocity_sequence,
        depth_sequence,
        A,
        B,
        DEM,
        Building,
        N,
        label,
        last_final_velocity
        )

    print(loss)