"""Constant definitions for Flood CNN."""

# Geospatial constants
GEO_FEATURES = 12  # 9 raw + DEM sink (ch9) + flow curvature (ch10) + TWI (ch11)
MAP_HEIGHT = 1000
MAP_WIDTH = 1000
RESOLUTION = 2
GRAVITY = 9.8

# Temporal parameters. May be tuned.
N_FLOOD_MAPS = 5
M_RAINFALL = 6
MAX_RAINFALL_DURATION = 864  # (60 / 5) * 24 * 3
STEP_LENGTH = 300

# Model output parameters.
V_M_STEPS = 10
W_DENSITY = 1

RANDOM_SEED = 42

MID_STEPS = 10

EPS = 1e-7
EPS_WATER = 1e-3

###################################
###################################
###################################
# Model Structure
###################################
###################################
###################################

#################################################
#### water + building + dem (water_level) #######
#################################################


# water_level_feature, the output feature dimension
# outdim = (B, T, H, W, W_L_FEATURE_DIM)
W_L_CONV1_PAD_SIZE = (1, 1)
W_L_CONV1_FEATURE_DIM = 16
W_L_CONV1_FEATUER_KERNEL_SIZE = (3, 3)
W_L_CONV1_FEATUER_KERNEL_STRIDE = (1, 1)

W_L_CONV2_PAD_SIZE = (1, 1)
W_L_CONV2_HIDDEN_DIM = 16
W_L_CONV2_KERNEL_SIZE = (3, 3)
W_L_CONV2_KERNEL_STRIDE = (1, 1)
W_L_CONV2_ACTIVATION = "relu"

#################################################
#### Friction: Water + friction factor #######
#################################################
# input [B, T, H, W, 2]
# output [B, T, H, W, FRICTION_CONV2_KERNEL_STRIDE]
# friction_encoder_conv
FRICTION_CONV1_PAD_SIZE = (1, 1)
FRICTION_CONV1_HIDDEN_DIM = 16
FRICTION_CONV1_KERNEL_SIZE = (3, 3)
FRICTION_CONV1_KERNEL_STRIDE = (1, 1)

FRICTION_CONV2_PAD_SIZE = (1, 1)
FRICTION_CONV2_HIDDEN_DIM = 16
FRICTION_CONV2_KERNEL_SIZE = (3, 3)
FRICTION_CONV2_KERNEL_STRIDE = (1, 1)
FRICTION_CONV2_ACTIVATION = "relu"

#################################################
#### infiltration: water + infiltration factors #######
#################################################
# input [B, T, H, W, 5]
# output [B, T, H, W, INFILTRATION_CONV2_HIDDEN_DIM]
# friction_encoder_conv
INFILTRATION_CONV1_PAD_SIZE = (1, 1)
INFILTRATION_CONV1_HIDDEN_DIM = 16
INFILTRATION_CONV1_KERNEL_SIZE = (3, 3)
INFILTRATION_CONV1_KERNEL_STRIDE = (1, 1)

INFILTRATION_CONV2_PAD_SIZE = (1, 1)
INFILTRATION_CONV2_HIDDEN_DIM = 16
INFILTRATION_CONV2_KERNEL_SIZE = (3, 3)
INFILTRATION_CONV2_KERNEL_STRIDE = (1, 1)
INFILTRATION_CONV2_ACTIVATION = "relu"

#################################################
############## rainfall features ################
#################################################
# input [B, 5, M]
# output [B, 5, dim]
RAINFALL_DENSE1_HIDDEN_DIM = 64
RAINFALL_DENSE1_ACTIVATION = "relu"
RAINFALL_DENSE2_HIDDEN_DIM = 16
RAINFALL_DENSE2_ACTIVATION = "relu"

#################################################
#### Merge #######
#################################################
MERGE_CONV1_PAD_SIZE = (1, 1)
MERGE_CONV1_HIDDEN_DIM = 16
MERGE_CONV1_KERNEL_SIZE = (3, 3)
MERGE_CONV1_KERNEL_STRIDE = (1, 1)
MERGE_CONV1_ACTIVATION = "relu"
#################################################
#### UNet #######
#################################################

####################encoder#######################

# unet_encoder_1 !!!! Make sure the output size is hight//4
# outdim = (B, T, H//4, W//4, UNET_ENCODER_1_HIDDEN_DIM)
UNET_ENCODER_1_PAD_SIZE = (1, 1)
UNET_ENCODER_1_KERNEL_SIZE = (3, 3)
UNET_ENCODER_1_KERNEL_STRIDE = (1, 1)
UNET_ENCODER_1_HIDDEN_DIM = 32
UNET_ENCODER_1_ACTIVATION = "relu"
UNET_ENCODER_1_AVEPOOLING_SIZE = 2
UNET_ENCODER_1_AVEPOOLING_STRIDE = (2, 2)

# unet_encoder_2 !!!! Make sure the output size is hight//16
# outdim = (B, T, H//16, W//16, UNET_ENCODER_2_HIDDEN_DIM)
UNET_ENCODER_2_PAD_SIZE = (2, 2)
UNET_ENCODER_2_KERNEL_SIZE = (5, 5)
UNET_ENCODER_2_KERNEL_STRIDE = (1, 1)
UNET_ENCODER_2_HIDDEN_DIM = 64
UNET_ENCODER_2_ACTIVATION = "relu"
UNET_ENCODER_2_AVEPOOLING_SIZE = 2
UNET_ENCODER_2_AVEPOOLING_STRIDE = (2, 2)

# unet_encoder_3 !!!! Make sure the output size is hight//64
# outdim = (B, T, H//64, W//64, UNET_ENCODER_3_HIDDEN_DIM)
UNET_ENCODER_3_PAD_SIZE = (2, 2)
UNET_ENCODER_3_KERNEL_SIZE = (5, 5)
UNET_ENCODER_3_KERNEL_STRIDE = (1, 1)
UNET_ENCODER_3_HIDDEN_DIM = 128
UNET_ENCODER_3_ACTIVATION = "relu"
UNET_ENCODER_3_AVEPOOLING_SIZE = 2
UNET_ENCODER_3_AVEPOOLING_STRIDE = (2, 2)


####################upsampling and decoder#######################
# water_level_decoder1_up
UNET_DECODER_1_UP_SIZE = 2 # value is set for correct shape
UNET_DECODER_1_UP_INTERP = "bilinear"

# water_level_decoder1_conv
UNET_DECODER_1_CONV_PAD_SIZE = (1, 1)
UNET_DECODER_1_CONV_HIDDEN_DIM = 64
UNET_DECODER_1_CONV_KERNEL_SIZE = (3, 3)
UNET_DECODER_1_CONV_KERNEL_STRIDE = (1, 1)
UNET_DECODER_1_CONV_ACTIVATION = "relu"

# water_level_decoder2_up
UNET_DECODER_2_UP_SIZE = 2 # value is set for correct shape
UNET_DECODER_2_UP_INTERP = "bilinear"

# water_level_decoder2_conv
UNET_DECODER_2_CONV_PAD_SIZE = (1, 1)
UNET_DECODER_2_CONV_HIDDEN_DIM = 32
UNET_DECODER_2_CONV_KERNEL_SIZE = (3, 3)
UNET_DECODER_2_CONV_KERNEL_STRIDE = (1, 1)
UNET_DECODER_2_CONV_ACTIVATION = "relu"

# water_level_decoder3_up
UNET_DECODER_3_UP_SIZE = 2 # value is set for correct shape
UNET_DECODER_3_UP_INTERP = "bilinear"

# water_level_decoder3_conv
UNET_DECODER_3_CONV_PAD_SIZE = (1, 1)
UNET_DECODER_3_CONV_HIDDEN_DIM = 16
UNET_DECODER_3_CONV_KERNEL_SIZE = (3, 3)
UNET_DECODER_3_CONV_KERNEL_STRIDE = (1, 1)
UNET_DECODER_3_CONV_ACTIVATION = "relu"


#################################################
################# ConvLSTM ######################
#################################################
CONVLSTM_1_LSTM_UNITS = 32
CONVLSTM_1_KERNEL_SIZE = (5, 5)
CONVLSTM_1_STRIDES = 1
CONVLSTM_1_LSTM_DROPOUT = 0.0
CONVLSTM_1_RECURRENT_DROPOUT = 0.0 
CONVLSTM_1_ACTIVATION = "relu"
CONVLSTM_1_PADDING = "same"

CONVLSTM_2_LSTM_UNITS = 32
CONVLSTM_2_KERNEL_SIZE = (5, 5)
CONVLSTM_2_STRIDES = 1
CONVLSTM_2_LSTM_DROPOUT = 0.0
CONVLSTM_2_RECURRENT_DROPOUT = 0.0 
CONVLSTM_2_ACTIVATION = "relu"
CONVLSTM_2_PADDING = "same"

#################################################
################# Multihead Prediction ##########
#################################################
# infiltration prediction
INFILTRATION_PRED_CONV1_PAD_SIZE = (1, 1)
INFILTRATION_PRED_CONV1_HIDDEN_DIM = 16
INFILTRATION_PRED_CONV1_KERNEL_SIZE = (3, 3)
INFILTRATION_PRED_CONV1_STRIDE = 1
INFILTRATION_PRED_CONV1_ACTIVATION = "relu"

INFILTRATION_PRED_CONV2_OUTPUT_DIM = 1
INFILTRATION_PRED_CONV2_KERNEL_SIZE = (1, 1)
INFILTRATION_PRED_CONV2_STRIDE = 1
INFILTRATION_PRED_CONV2_ACTIVATION = "relu"

# speed prediction
# u prediction
U_PRED_CONV1_PAD_SIZE = (1, 1)
U_PRED_CONV1_HIDDEN_DIM = 32
U_PRED_CONV1_KERNEL_SIZE = (3, 3)
U_PRED_CONV1_STRIDE = 1
U_PRED_CONV1_ACTIVATION = "relu"

U_PRED_CONV2_OUTPUT_DIM = MID_STEPS + 2
U_PRED_CONV2_KERNEL_SIZE = (1, 1)
U_PRED_CONV2_STRIDE = 1
U_PRED_CONV2_ACTIVATION = None

# v prediction
V_PRED_CONV1_PAD_SIZE = (1, 1)
V_PRED_CONV1_HIDDEN_DIM = 32
V_PRED_CONV1_KERNEL_SIZE = (3, 3)
V_PRED_CONV1_STRIDE = 1
V_PRED_CONV1_ACTIVATION = "relu"

V_PRED_CONV2_OUTPUT_DIM = MID_STEPS + 2
V_PRED_CONV2_KERNEL_SIZE = (1, 1)
V_PRED_CONV2_STRIDE = 1
V_PRED_CONV2_ACTIVATION = None









