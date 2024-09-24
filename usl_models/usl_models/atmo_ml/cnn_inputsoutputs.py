import tensorflow as tf


# Main function to divide inputs and labels by days
def divide_into_days(inputs, labels, input_steps_per_day=4, label_steps_per_day=8):
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
        day_labels_list.append(day_labels)

    return day_inputs_list, day_labels_list


# Function to process inputs and labels for each day
def process_day(
    inputs, labels, day_idx, total_days, input_steps_per_day, label_steps_per_day
):
    # Get inputs for current day
    if day_idx == 0:
        # First day - no previous time step, duplicate the first value
        day_inputs = tf.concat(
            [inputs[0:1], inputs[0 : input_steps_per_day + 1]], axis=0
        )
    else:
        # Intermediate and last days
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
        # Last day - no next time step, duplicate the last value
        day_inputs = tf.concat([day_inputs, inputs[-1:]], axis=0)

    # Get labels for current day
    if day_idx == 0:
        # First day - duplicate the first two values for labels
        day_labels = tf.concat(
            [labels[0:1], labels[0:1], labels[0:label_steps_per_day]], axis=0
        )
    else:
        # Intermediate and last days take previous 2 time steps and next 8
        day_labels = tf.concat(
            [
                labels[
                    day_idx * label_steps_per_day - 2 : day_idx * label_steps_per_day
                ],
                labels[
                    day_idx * label_steps_per_day : day_idx * label_steps_per_day
                    + label_steps_per_day
                ],
            ],
            axis=0,
        )

    return day_inputs, day_labels
