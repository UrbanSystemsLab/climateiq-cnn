import tensorflow as tf


def create_input_output_sequences(inputs, labels, time_steps_per_day=4):
    num_days = inputs.shape[1] // time_steps_per_day
    input_sequences = []
    output_sequences = []

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

        # Collect input sequences for the day
        input_sequences.append(tf.stack(daily_input_sequences, axis=1))

        # Construct output sequences for the current day
        for t in range(time_steps_per_day - 1):
            output_seq = tf.stack(
                [
                    labels[:, day * time_steps_per_day + t],  # Y^t
                    labels[:, day * time_steps_per_day + (t + 1)],  # Y^{t+1}
                ],
                axis=-1,
            )
            daily_output_sequences.append(output_seq)

        # Collect output sequences for the day
        output_sequences.append(tf.stack(daily_output_sequences, axis=1))

    # Stack all days together
    input_sequences = tf.stack(input_sequences, axis=0)
    output_sequences = tf.stack(output_sequences, axis=0)

    # Squeeze to remove any extra dimensions
    # input_sequences = tf.squeeze(input_sequences, axis=[2, 3, 4])
    # output_sequences = tf.squeeze(output_sequences, axis=[2, 3, 4])

    return input_sequences, output_sequences
