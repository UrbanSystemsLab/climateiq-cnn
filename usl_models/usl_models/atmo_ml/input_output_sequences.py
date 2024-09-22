"""Data spatial window functions for AtmoML model."""

from usl_models.atmo_ml import constants


def create_input_output_sequences(
    inputs, labels, time_steps_per_day=constants.TIME_STEPS_PER_DAY, debug=False
):
    """Creates input/output sequences for model training."""
    total_time_steps_inputs = inputs.shape[1]
    total_time_steps_labels = labels.shape[1]

    if total_time_steps_inputs % time_steps_per_day != 0:
        raise ValueError(
            "Nb of time steps for inputs must be divisible by time_steps_per_day."
        )

    num_days_inputs = total_time_steps_inputs // time_steps_per_day
    num_days_labels = total_time_steps_labels // (time_steps_per_day * 2)

    if total_time_steps_labels % (time_steps_per_day * 2) != 0:
        raise ValueError(
            "Nb of timesteps for labels must be divisible by time_steps_per_day * 2."
        )

    # Input sequences processing
    spatiotemporal_sequences = []

    for day in range(num_days_inputs):
        day_start_idx = day * time_steps_per_day

        for t in range(time_steps_per_day):
            if day == 0 and t == 0:
                spatiotemporal_sequences.append(
                    [
                        inputs[0, day_start_idx, 0, 0, 0].numpy(),
                        inputs[0, day_start_idx, 0, 0, 0].numpy(),
                    ]
                )
            else:
                previous_input = (
                    inputs[0, day_start_idx + t - 1, 0, 0, 0].numpy()
                    if t > 0
                    else inputs[
                        0,
                        (day - 1) * time_steps_per_day + time_steps_per_day - 1,
                        0,
                        0,
                        0,
                    ].numpy()
                )
                current_input = inputs[0, day_start_idx + t, 0, 0, 0].numpy()
                spatiotemporal_sequences.append([previous_input, current_input])

        if day < num_days_inputs - 1:
            spatiotemporal_sequences.append(
                [
                    inputs[0, (day + 1) * time_steps_per_day - 1, 0, 0, 0].numpy(),
                    inputs[0, (day + 1) * time_steps_per_day, 0, 0, 0].numpy(),
                ]
            )
        else:
            spatiotemporal_sequences.append(
                [
                    inputs[0, total_time_steps_inputs - 1, 0, 0, 0].numpy(),
                    inputs[0, total_time_steps_inputs - 1, 0, 0, 0].numpy(),
                ]
            )

    # Output sequences processing
    labels_sequences = []

    for day in range(num_days_labels):
        if day == 0:
            labels_sequences.append(
                [labels[0, 0, 0, 0, 0].numpy(), labels[0, 0, 0, 0, 0].numpy()]
            )
        else:
            labels_sequences.append(
                [
                    labels[
                        0,
                        (day - 1) * time_steps_per_day * 2
                        + (time_steps_per_day * 2)
                        - 2,
                        0,
                        0,
                        0,
                    ].numpy(),
                    labels[
                        0,
                        (day - 1) * time_steps_per_day * 2
                        + (time_steps_per_day * 2)
                        - 1,
                        0,
                        0,
                        0,
                    ].numpy(),
                ]
            )

        for t in range(time_steps_per_day):
            output_index = day * time_steps_per_day * 2 + t * 2
            next_output_index = (
                output_index + 1
                if output_index + 1 < total_time_steps_labels
                else output_index
            )
            labels_sequences.append(
                [
                    labels[0, output_index, 0, 0, 0].numpy(),
                    labels[0, next_output_index, 0, 0, 0].numpy(),
                ]
            )
    if debug:
        print("Generated Inputs:", spatiotemporal_sequences)
        print("Generated Outputs:", labels_sequences)
    # Return the sequences in the expected format
    return spatiotemporal_sequences, labels_sequences
