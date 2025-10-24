# usl_models/atmo_ml/losses.py
import tensorflow as tf
import keras as keras

def create_balanced_loss(num_tasks, task_weights, log_vars):
    def balanced_loss(y_true, y_pred):
        task_losses = []
        for i in range(num_tasks):
            t_true = y_true[..., i : i + 1]
            t_pred = y_pred[..., i : i + 1]
            mse = tf.reduce_mean(tf.square(t_true - t_pred))
            task_losses.append(mse)
        
        task_losses = tf.stack(task_losses)
        
        # Check if we're in MSE mode (log_vars not trainable means zeros)
        if not log_vars.trainable:
            # Pure MSE mode
            weighted = task_losses
        else:
            # Uncertainty weighting mode
            precision = tf.exp(-log_vars)
            weighted = 0.5 * precision * task_losses + 0.5 * tf.abs(log_vars)
        
        weighted = weighted * tf.constant(task_weights, dtype=tf.float32)
        return tf.reduce_sum(weighted)
    
    return balanced_loss
