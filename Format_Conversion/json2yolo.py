import json
import os

# Load the generated JSON file
json_file_path = 'out.json'  # Modify the path as needed

# Output folder for YOLO-format annotation files
output_dir = r'D:\GitHub3_1\UAVDT\labels'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(json_file_path, 'r') as f:
    json_data = json.load(f)


def convert_to_yolo_format(image_width, image_height, bbox):
    """
    Convert a bounding box from top-left coordinates to YOLO format.
    bbox: [xmin, ymin, width, height]
    """
    x_min, y_min, width_box, height_box = bbox
    x_center = (x_min + width_box / 2) / image_width
    y_center = (y_min + height_box / 2) / image_height
    width_norm = width_box / image_width
    height_norm = height_box / image_height
    return [x_center, y_center, width_norm, height_norm]


# Iterate through all images in the JSON and generate YOLO-format annotations
for image_info in json_data['images']:
    image_id = image_info['id']
    image_width = image_info['width']
    image_height = image_info['height']

    # Retrieve annotations based on image_id
    annotations = [ann for ann in json_data['annotations'] if ann['image_id'] == image_id]

    # Generate the YOLO-format annotation file content
    yolo_annotations = []
    for ann in annotations:
        category_idx = ann['category_id'] - 1  # YOLO class IDs start from 0
        bbox = ann['bbox']
        yolo_format = convert_to_yolo_format(image_width, image_height, bbox)
        yolo_annotations.append(f"{category_idx} {' '.join(map(str, yolo_format))}")

    # Write the YOLO-format annotations to file
    image_name = image_info['file_name']
    yolo_file_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.txt")

    with open(yolo_file_path, 'w') as f:
        for annotation in yolo_annotations:
            f.write(annotation + '\n')

print("Conversion complete. YOLO-format annotation files have been generated.")
