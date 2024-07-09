# USL Models

## Model memory experiments

Training the ConvLSTM is expensive in VRAM on GPU. We trained the model using an A100 with 40GB of RAM and were only able to get a batch size of 4.

Some strategies to reduce memory requirement:

1) Smaller context window (`constants.N_FLOOD_MAPS`)
2) Smaller spatial chunks (currently using 1000 X 1000)
3) Migrating to JAX instead of TF
4) Using half-precision floats (`dtype=tf.float16`)
