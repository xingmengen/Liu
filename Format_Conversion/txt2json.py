import json
import pandas as pd
import os

# Class list and predefined category mapping
classList = ["car", "truck", "bus"]
# By default, COCO dataset category indices start from 1
PRE_DEFINE_CATEGORIES = {key: idx + 1 for idx, key in enumerate(classList)}

# Paths for input and output
txtdir = r'D:\GitHub3_1\UAVDT\GT'
out_json_file = "out.json"

# Initialize the main COCO JSON structure
json_dict = {
    "images": [],
    "type": "instances",
    "annotations": [],
    "categories": []
}

# Fill in the categories section
for category_name, category_id in PRE_DEFINE_CATEGORIES.items():
    category = {
        'supercategory': category_name,
        'id': category_id,
        'name': category_name
    }
    json_dict['categories'].append(category)


def get_annotation_data(txt_file):
    """
    Read annotation data into a pandas DataFrame.
    Expects columns: <frame_index>, <target_id>, <bbox_left>, <bbox_top>,
    <bbox_width>, <bbox_height>, <out-of-view>, <occlusion>, <object_category>
    """
    column_names = [
        '<frame_index>', '<target_id>', '<bbox_left>', '<bbox_top>',
        '<bbox_width>', '<bbox_height>', '<out-of-view>',
        '<occlusion>', '<object_category>'
    ]
    annot_data = pd.read_csv(txt_file, delimiter=',', names=column_names)
    return annot_data


# Map image filename prefixes to image IDs
image_name_to_id = {}

# Build the images section
current_image_id = 0
images_dir = r'D:\UAVDT\uavdtallimages_rename'
for filename in os.listdir(images_dir):
    width = 540
    height = 1024
    image_id = current_image_id
    current_image_id += 1

    # Store mapping from filename prefix (without extension) to ID
    name_without_ext = os.path.splitext(filename)[0]
    image_name_to_id[name_without_ext] = image_id

    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': image_id
    }
    json_dict['images'].append(image_info)

# Build the annotations section
annotation_id = 1
for txt_file in os.listdir(txtdir):
    if 'gt_whole' in txt_file:
        txt_path = os.path.join(txtdir, txt_file)
        annot_data = get_annotation_data(txt_path)

        # Iterate over each row in the DataFrame
        for _, row in annot_data.iterrows():
            # Convert bounding box values to integers to avoid dtype issues
            bbox_width = int(row['<bbox_width>'])
            bbox_height = int(row['<bbox_height>'])
            xmin = int(row['<bbox_left>'])
            ymin = int(row['<bbox_top>'])

            category_id = int(row['<object_category>'])
            # Frame index to zero-padded string, e.g. 000001
            frame_index = int(row['<frame_index>'])
            txt_prefix = txt_file[:5]
            padded_index = str(frame_index).zfill(6)
            image_key = f"{txt_prefix}_{padded_index}"
            image_id = image_name_to_id[image_key]

            annotation = {
                'area': bbox_width * bbox_height,
                'iscrowd': 0,
                'image_id': image_id,
                'bbox': [xmin, ymin, bbox_width, bbox_height],
                'category_id': category_id,
                'id': annotation_id,
                'ignore': 0,
                'segmentation': []
            }
            annotation_id += 1
            json_dict['annotations'].append(annotation)

# Write the final JSON to file
with open(out_json_file, 'w') as json_fp:
    json.dump(json_dict, json_fp)

print("COCO-format JSON file has been generated:", out_json_file)
