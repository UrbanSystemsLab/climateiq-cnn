"""Data spatial window functions for AtmoML model."""

import tensorflow as tf
from usl_models.atmo_ml import constants


def create_input_output_sequences(
    inputs, labels, time_steps_per_day=constants.TOTAL_TIME_STEPS, debug=False
):
    """Full input sequence (4 time steps).

    [(X_{d-1}^{18}, X^0, X^6), (X^0, X^6, X^{12}),
    (X^6, X^{12}, X^{18}), (X^{12}, X^{18}, X_{d+1}^0)]
    Full output sequence (4 time steps):
    [(Y^0, Y^3), (Y^6, Y^9), (Y^{12}, Y^{15}), (Y^{18}, Y^{21})]
    Args:
        inputs: [X_{d-1}^{18}, X^0, X^6, X^{12}, X^{18}), X_{d+1}^0)]
        labels : [Y^0, Y^3, Y^6, Y^9, Y^{12}, Y^{15}, Y^{18}, Y^{21}]
        time_steps_per_day (int, optional): _description_. Defaults to 4.
        debug (bool, optional): Whether to print debug information. Defaults to False.

    Yields:
        _type_: _description_
    """
    num_days = inputs.shape[1] // time_steps_per_day

    for day in range(num_days):
        daily_input_sequences = []
        daily_output_sequences = []

        # Special case: First day (no previous data)
        if day == 0:
            # Use the first day's data itself for the initial sequence
            input_seq = tf.stack(
                [
                    inputs[
                        :, day * time_steps_per_day + 0
                    ],  # X_{d-1}^{18} (treated as X^0)
                    inputs[:, day * time_steps_per_day + 0],  # X^0
                    inputs[:, day * time_steps_per_day + 1],  # X^6
                ],
                axis=-1,
            )
            daily_input_sequences.append(input_seq)
        else:
            # Standard input sequence construction for day d
            input_seq = tf.stack(
                [
                    inputs[
                        :, (day - 1) * time_steps_per_day + (time_steps_per_day - 1)
                    ],  # X_{d-1}^{18}
                    inputs[:, day * time_steps_per_day + 0],  # X^0
                    inputs[:, day * time_steps_per_day + 1],  # X^6
                ],
                axis=-1,
            )
            daily_input_sequences.append(input_seq)

        # Construct input sequences for the current day
        for t in range(1, time_steps_per_day - 1):
            input_seq = tf.stack(
                [
                    inputs[:, day * time_steps_per_day + (t - 1)],  # X_{t-1}
                    inputs[:, day * time_steps_per_day + t],  # X^t
                    inputs[:, day * time_steps_per_day + (t + 1)],  # X^{t+1}
                ],
                axis=-1,
            )
            daily_input_sequences.append(input_seq)

        # Handle the case for the last sequence in the day
        if day < num_days - 1:
            input_seq = tf.stack(
                [
                    inputs[
                        :, day * time_steps_per_day + (time_steps_per_day - 2)
                    ],  # X_{t-1}
                    inputs[
                        :, day * time_steps_per_day + (time_steps_per_day - 1)
                    ],  # X^t
                    inputs[:, (day + 1) * time_steps_per_day + 0],  # X_{d+1}^0
                ],
                axis=-1,
            )
        else:
            # For the last day, repeat the last available data point
            input_seq = tf.stack(
                [
                    inputs[
                        :, day * time_steps_per_day + (time_steps_per_day - 2)
                    ],  # X_{t-1}
                    inputs[
                        :, day * time_steps_per_day + (time_steps_per_day - 1)
                    ],  # X^t
                    inputs[
                        :, day * time_steps_per_day + (time_steps_per_day - 1)
                    ],  # X^t (repeated for no next day)
                ],
                axis=-1,
            )
        daily_input_sequences.append(input_seq)

        # Construct output sequences for the current day
        for t in range(0, time_steps_per_day - 1, 2):
            output_seq = tf.stack(
                [
                    labels[:, day * time_steps_per_day + t],  # Y^t
                    labels[:, day * time_steps_per_day + (t + 1)],  # Y^{t+1}
                ],
                axis=-1,
            )
            daily_output_sequences.append(output_seq)

        # Instead of stacking, we process sequences directly (e.g., pass to model)
        for input_seq, output_seq in zip(daily_input_sequences, daily_output_sequences):
            if debug:
                tf.print("Input sequence:", input_seq.numpy().flatten().tolist())
                tf.print("Output sequence:", output_seq.numpy().flatten().tolist())
            yield input_seq, output_seq
