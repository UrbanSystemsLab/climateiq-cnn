import tensorflow as tf


def divide_into_days(inputs, labels=None, input_steps_per_day=4, label_steps_per_day=8):
    """Splits inputs (and labels if provided) into daily sequences.

    Args:
        inputs (tf.Tensor): Input time series data.
        labels (tf.Tensor, optional): Label time series data.
        input_steps_per_day (int): Number of input steps per day. Defaults to 4.
        label_steps_per_day (int): Number of label steps per day. Defaults to 8.

    Returns:
        tuple: daily input tensors and, if labels are given, daily label tensors.
    """
    total_days = inputs.shape[0] // input_steps_per_day
    day_inputs_list = []
    day_labels_list = []
    for day_idx in range(total_days):
        day_inputs, day_labels = process_day(
            inputs,
            labels,
            day_idx,
            total_days,
            input_steps_per_day,
            label_steps_per_day,
        )
        day_inputs_list.append(day_inputs)
        if labels is not None:
            day_labels_list.append(day_labels)
        tf.print(f"Day {day_idx + 1} inputs shape:", tf.shape(day_inputs))
        if labels is not None:
            tf.print(f"Day {day_idx + 1} labels shape:", tf.shape(day_labels))
    return day_inputs_list, day_labels_list if labels is not None else None


def process_day(
    inputs, labels, day_idx, total_days, input_steps_per_day, label_steps_per_day
):
    """Retrieves input and label sequences for a specific day.

    Args:
        inputs (tf.Tensor): Full input time series data.
        labels (tf.Tensor, optional): Full label time series data.
        day_idx (int): Index of the day.
        total_days (int): Total number of days in data.
        input_steps_per_day (int): Input steps per day.
        label_steps_per_day (int): Label steps per day.

    Returns:
        tuple: Day-specific input tensor and, if labels are provided, label tensor.
    """
    # Get inputs for the day, with padding at start if needed
    if day_idx == 0:
        day_inputs = tf.concat(
            [inputs[0:1], inputs[0 : input_steps_per_day + 1]], axis=0
        )
    else:
        day_inputs = tf.concat(
            [
                inputs[
                    day_idx * input_steps_per_day - 1 : day_idx * input_steps_per_day
                ],
                inputs[
                    day_idx * input_steps_per_day : day_idx * input_steps_per_day
                    + input_steps_per_day
                    + 1
                ],
            ],
            axis=0,
        )
    # Add padding at the end for the last day if needed
    if day_idx == total_days - 1:
        day_inputs = tf.concat([day_inputs, inputs[-1:]], axis=0)

    # Get label sequence if provided
    day_labels = (
        labels[
            day_idx * label_steps_per_day : day_idx * label_steps_per_day
            + label_steps_per_day
        ]
        if labels is not None
        else None
    )
    return day_inputs, day_labels
