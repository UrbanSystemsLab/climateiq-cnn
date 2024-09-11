# input_output_sequences.py

import tensorflow as tf


def create_input_output_sequences(inputs, labels, time_steps_per_day=4):
    """
    Creates input and output sequences for the AtmoML model.

    Args:
        inputs: A tensor of shape [B, T, H, W, C] where B is the batch size,
                T is the total time steps, H is height, W is width, and C is channels.
        labels: A tensor of shape [B, T, H, W, C] corresponding to the inputs.
        time_steps_per_day: Number of time steps representing a day. Default is 4 (e.g., 0:00, 6:00, 12:00, 18:00).

    Returns:
        input_sequences: Input sequences for the model.
        output_sequences: Corresponding output sequences for the model.
    """
    total_time_steps = tf.shape(inputs)[1]
    num_days = total_time_steps // time_steps_per_day

    print(
        f"Total time steps: {total_time_steps.numpy()}, Number of days: {num_days.numpy()}"
    )

    input_sequences = []
    output_sequences = []

    for day in range(num_days):
        # Create input sequence for each day
        current_input_sequence = []

        # Handle the historical input (X_{d-1}^{18})
        if day > 0:
            historical_input = inputs[
                :,
                (day - 1) * time_steps_per_day + 3 : (day - 1) * time_steps_per_day + 4,
                :,
                :,
                :,
            ]
            print(f"Day {day}: Historical input shape: {historical_input.shape}")
            current_input_sequence.append(historical_input)
        else:
            zero_pad = tf.zeros_like(inputs[:, :1, :, :, :])
            print(f"Day {day}: Zero padding shape: {zero_pad.shape}")
            current_input_sequence.append(
                zero_pad
            )  # Zero padding for missing historical data

        # Add current day's 0:00 and 6:00 data
        day_inputs = inputs[
            :, day * time_steps_per_day : day * time_steps_per_day + 2, :, :, :
        ]
        print(f"Day {day}: Current day inputs (0:00, 6:00) shape: {day_inputs.shape}")
        current_input_sequence.append(day_inputs)

        # Concatenate initial sequence to maintain consistent length of 3
        input_sequence = tf.concat(
            current_input_sequence, axis=1
        )  # Should result in length 3
        print(f"Day {day}: Input sequence after concat shape: {input_sequence.shape}")

        # Generate sequences using only the required indices:
        if day * time_steps_per_day + 3 < total_time_steps:
            next_day_input = inputs[
                :,
                day * time_steps_per_day + 3 : (day + 1) * time_steps_per_day,
                :,
                :,
                :,
            ]
            full_sequence = tf.concat(
                [input_sequence[:, 1:, :, :, :], next_day_input], axis=1
            )
            print(
                f"Day {day}: Final input sequence shape for day: {full_sequence.shape}"
            )
            input_sequences.append(full_sequence)

        # Create output sequence: (Y^0, Y^3), (Y^6, Y^9), etc.
        output_sequence = labels[
            :, day * time_steps_per_day : (day + 1) * time_steps_per_day, :, :, :
        ]
        print(f"Day {day}: Output sequence shape: {output_sequence.shape}")
        output_sequences.append(output_sequence)

    # Stack sequences for batch processing
    input_sequences = tf.stack(input_sequences, axis=1)  # [B, num_days, 4, 3, H, W, C]
    output_sequences = tf.stack(output_sequences, axis=1)  # [B, num_days, 4, H, W, C]

    return input_sequences, output_sequences
