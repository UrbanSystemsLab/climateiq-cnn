

def num_list_to_strings(num_list):
    return list(map(lambda v: str(v), num_list))


def write_polygons_to_text_file(
    polygons_with_masks, output_file, support_mask_values=False
):
    output_file.write(str(len(polygons_with_masks)) + "\n")
    for polygon_mask in polygons_with_masks:
        xx, yy = polygon_mask[0].exterior.coords.xy
        x_list = xx.tolist()
        y_list = yy.tolist()
        mask_prefix = ""
        if support_mask_values:
            mask_prefix = str(polygon_mask[1]) + " "
        output_file.write(
            mask_prefix + str(len(x_list)) + " " +
            " ".join(num_list_to_strings(x_list)) + " " +
            " ".join(num_list_to_strings(y_list)) + "\n"
            )
