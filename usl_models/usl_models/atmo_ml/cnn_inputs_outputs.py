import tensorflow as tf


# Modify main function to allow labels to be None
def divide_into_days(inputs, labels=None, input_steps_per_day=4, label_steps_per_day=8):
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
        # Print shapes for debugging
        tf.print(f"Day {day_idx + 1} generated inputs shape:", tf.shape(day_inputs))
        if labels is not None:
            tf.print(f"Day {day_idx + 1} generated labels shape:", tf.shape(day_labels))
    return day_inputs_list, day_labels_list if labels is not None else None


# Modify process_day to handle no labels case
def process_day(
    inputs, labels, day_idx, total_days, input_steps_per_day, label_steps_per_day
):
    # Get inputs for current day
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
    if day_idx == total_days - 1:
        day_inputs = tf.concat([day_inputs, inputs[-1:]], axis=0)
    # Get labels only if provided
    day_labels = (
        labels[
            day_idx * label_steps_per_day : day_idx * label_steps_per_day
            + label_steps_per_day
        ]
        if labels is not None
        else None
    )
    return day_inputs, day_labels
