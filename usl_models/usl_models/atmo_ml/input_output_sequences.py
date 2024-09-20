"""Data spatial window functions for AtmoML model."""

import tensorflow as tf
from usl_models.atmo_ml import constants


def create_input_output_sequences(
    inputs, labels, time_steps_per_day=constants.TIME_STEPS_PER_DAY, debug=False
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
    # Ensure inputs have enough time steps
    total_time_steps = inputs.shape[1]

    # Error checks for time_steps_per_day
    if not isinstance(time_steps_per_day, int) or time_steps_per_day <= 0:
        raise ValueError("time_steps_per_day must be a positive integer.")

    if total_time_steps % time_steps_per_day != 0:
        raise ValueError("Nb of time steps must be divisible by time_steps_per_day.")

    # Number of full days available
    num_days = total_time_steps // time_steps_per_day

    if debug:
        print(f"Number of full days: {num_days}")

    for day in range(num_days):
        daily_input_sequences = []
        daily_output_sequences = []

        if debug:
            print(f"Processing day {day + 1}/{num_days}")

        # Input sequences
        for t in range(time_steps_per_day):
            print(f"Time step: {t}")  # Debug: Print the current time step
            if day == 0:  # First day
                if t == 0:
                    input_seq = tf.stack(
                        [
                            inputs[:, day * time_steps_per_day + t],
                            inputs[:, day * time_steps_per_day + t],
                            inputs[:, day * time_steps_per_day + t + 1],
                        ],
                        axis=-1,
                    )
                else:
                    input_seq = tf.stack(
                        [
                            inputs[:, day * time_steps_per_day + t - 1],
                            inputs[:, day * time_steps_per_day + t],
                            inputs[:, day * time_steps_per_day + t + 1],
                        ],
                        axis=-1,
                    )
            elif day == num_days - 1:  # Last day
                if t == time_steps_per_day - 1:
                    input_seq = tf.stack(
                        [
                            inputs[:, day * time_steps_per_day + t - 1],
                            inputs[:, day * time_steps_per_day + t],
                            inputs[:, day * time_steps_per_day + t],
                        ],
                        axis=-1,
                    )
                else:
                    input_seq = tf.stack(
                        [
                            inputs[:, day * time_steps_per_day + t - 1],
                            inputs[:, day * time_steps_per_day + t],
                            inputs[:, day * time_steps_per_day + t + 1],
                        ],
                        axis=-1,
                    )
            else:  # Middle days
                if t == 0:
                    input_seq = tf.stack(
                        [
                            inputs[
                                :,
                                (day - 1) * time_steps_per_day + time_steps_per_day - 1,
                            ],
                            inputs[:, day * time_steps_per_day + t],
                            inputs[:, day * time_steps_per_day + t + 1],
                        ],
                        axis=-1,
                    )
                else:
                    input_seq = tf.stack(
                        [
                            inputs[:, day * time_steps_per_day + t - 1],
                            inputs[:, day * time_steps_per_day + t],
                            inputs[:, day * time_steps_per_day + t + 1],
                        ],
                        axis=-1,
                    )
            daily_input_sequences.append(input_seq)

        # Output sequences
        for t in range(0, time_steps_per_day):
            output_index = day * time_steps_per_day * 2 + t * 2
            if t < time_steps_per_day - 1:
                output_seq = tf.stack(
                    [labels[:, output_index], labels[:, output_index + 1]], axis=-1
                )
            else:
                output_seq = tf.stack(
                    [labels[:, output_index], labels[:, output_index + 1]], axis=-1
                )
            daily_output_sequences.append(output_seq)

        # Instead of stacking, we process sequences directly (e.g., pass to model)
        for input_seq, output_seq in zip(daily_input_sequences, daily_output_sequences):
            if debug:
                tf.print("Input sequence:", input_seq.numpy().flatten().tolist())
                tf.print("Output sequence:", output_seq.numpy().flatten().tolist())
            yield input_seq, output_seq
