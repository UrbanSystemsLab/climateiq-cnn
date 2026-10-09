import os
import numpy as np

import tensorflow as tf
from keras.saving import register_keras_serializable
from usl_models.flood_physical_ml.constants import *
from keras import layers
from usl_models.flood_physical_ml.customloss_old import make_hybrid_loss

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
        depth_sequence: The flood depth output of the model [B, V_M_STEPS+1, H, W, 1], with [:, 0:V_M_STEPS] as the calculated flooding depths at middle steps, and [:, -1] as the predicted flooding depth of next time step;
        A: [V_M_STEPS, V_M_STEPS]
        B: [1, V_M_STEPS]
        DEM_map: The DEM [B, H, W, 1]
        Building: the Bulding heights [B, H, W, 1]
        N: [B, H, W, 1]
    """

    depth_sequence = tf.nn.relu(depth_sequence[:, 0:-1]) # [B, V_M_STEPS, H, W, 1]

    velocity_0 = velocity_sequence[:, 0][:, tf.newaxis] # [B, 1, H, W, 2], predicted initial velocity
    velocity_1 = velocity_sequence[:, -1][:, tf.newaxis] # [B, 1, H, W, 2], predicted velocity of next time step

    elevation = DEM + Building # [B, H, W, 1]
    elevation = elevation[:, tf.newaxis] # [B, 1, H, W, 1]

    water_level = depth_sequence + elevation # [B, V_M_STEPS, H, W, 1]

    ############################################################
    #####################Calculate Momentum#####################
    ############################################################

    #########################Water Level Diff###################

    # # index map
    # # x, x, x
    # # x, x, 0
    # # x, 1, x
    # level_core = np.zeros((3, 3, 1, 2), dtype=float)
    # level_core[1, 1, 0, 0] = -1
    # level_core[1, 1, 0, 1] = -1
    # level_core[1, 2, 0, 0] = 1
    # level_core[2, 1, 0, 1] = 1 

    # core_layer = layers.Conv2D(filters=2, kernel_size=(3,3), padding='valid', use_bias=False, trainable=False)
    # core_layer.build(input_shape=(None, None, None, 1))
    # core_layer.set_weights([level_core])

    

    # x,  x, x
    # x, -1, 1
    # x,  x, x

    # x,  x, x
    # x, -1, x
    # x,  1, x
    water_level_padded = tf.pad(water_level, [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]], mode = "REFLECT")

    water_surface_differences_u = water_level_padded[:, :, 1:-1, 2:] - water_level_padded[:, :, 1:-1, 1:-1]
    water_surface_differences_v = water_level_padded[:, :, 2:, 1:-1] - water_level_padded[:, :, 1:-1, 1:-1]
    water_surface_differences = tf.concat([water_surface_differences_u, water_surface_differences_v], axis=-1) # [B, V_M_STEPS, H, W, 2] two directions < and ^

    # water_surface_differences = core_layer(water_level_padded) # [B, V_M_STEPS, H, W, 2] two directions < and ^
    water_surface_gradient = water_surface_differences/RESOLUTION # [B, V_M_STEPS, H, W, 2]

    # no water momentum for dry areas
    water_surface_gradient = tf.where(depth_sequence > EPS_WATER, water_surface_gradient, 0.0)

    water_surface_momentum = GRAVITY * water_surface_gradient # [B, V_M_STEPS, H, W, 2]

    #################################Fraction momentum####################
    N = N[:, tf.newaxis] # [B, 1, H, W, 1]
    # velocity_norm = tf.linalg.norm(velocity_sequence[:, 1:-1], axis=-1, keepdims=True) #[B, V_M_STEPS, H, W, 1]
    velocity_norm = tf.reduce_sum(tf.square(velocity_sequence[:, 1:-1]), axis=-1, keepdims=True)
    velocity_norm = tf.sqrt(velocity_norm + EPS) #[B, V_M_STEPS, H, W, 1]

    friction = N**2*GRAVITY/tf.maximum(depth_sequence, EPS_WATER)**(4/3)*velocity_norm*velocity_sequence[:, 1:-1] #[B, V_M_STEPS, H, W, 2]
    friction = tf.where(depth_sequence > EPS_WATER, friction, 0.0)
    #################################total momentum#######################
    momentum = -1 * (water_surface_momentum + friction) #[B, V_M_STEPS, H, W, 2]

    ######################################################################
    ########################Acceleration for each step####################
    ######################################################################
    step_acceleration = tf.einsum('bvhwd, vk->bkhwd', momentum, tf.transpose(A)) # [B, V_M_STEPS, H, W, 2]

    ######################################################################
    ###########################overall acceleration#######################
    ######################################################################
    overall_acceleration = tf.einsum('bvhwd, vk->bkhwd', momentum, tf.transpose(B)) # [B, 1, H, W, 2]

    ##########################Calculate momentum loss#####################
    ######################################################################
    ######################################################################
    calculated_mid_velocity = velocity_0 + STEP_LENGTH * step_acceleration #[B, V_M_STEPS, H, W, 2]
    calculated_next_velocity = velocity_0 + STEP_LENGTH * overall_acceleration # [B, 1, H, W, 2]

    loss_mid_step = tf.abs(velocity_sequence[:, 1:-1] - calculated_mid_velocity) # [B, V_M_STEPS, H, W, 2]
    loss_next_step = tf.abs(velocity_1 - calculated_next_velocity) # [B, 1, H, W, 2]

    mean_loss_mid_step = tf.reduce_mean(loss_mid_step)
    mean_loss_next_step = tf.reduce_mean(loss_next_step)

    total_mean = mean_loss_mid_step + mean_loss_next_step
    # print(mean_loss_next_step)

    

    return total_mean

def mass_loss(
    depth_sequence: tf.Tensor,
    Valid_Mask = None
    ):
    """
    punish flooding depth predictions under zero
    """
    # [B, V_M_STEPS+1, H, W, 1]
    vio_physical_prediction = tf.maximum(0.0, -1*depth_sequence)
    mse = tf.reduce_mean(tf.square(vio_physical_prediction))
    
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

    # if last_final_velocity is not None:
    #     velocity_l = velocity_loss(last_final_velocity, velocity_sequence[:, 0], Valid_Mask)
    # else:
    #     velocity_l = 0
    # tf.print("momentum loss is: ", momentum_l, "| depth loss is: ", depth_l, " | velocity loss is: ", velocity_l, " | mass loss is: ", mass_l)

    # physical_l = momentum_l + mass_l + depth_l + velocity_l

    # tf.print("momentum loss is: ", momentum_l, "| depth loss is: ", depth_l, " | mass loss is: ", mass_l)
    physical_l = 100 * momentum_l + mass_l + 200 * depth_l

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