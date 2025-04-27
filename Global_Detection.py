import os
import json
import cv2
from ultralytics import YOLO
from tqdm import tqdm

confidence_threshold = 0.1

model = YOLO("best11_C.pt")

input_folder = r"D:\GitHub3_1\VisDrone2019-DET\VisDrone2019-DET-val\Global\images"
input_json = r"D:\GitHub3_1\VisDrone2019-DET\VisDrone2019-DET-val\Global\test.json"
output_result_json = "Global_swin_tiny_test_results.bbox.json"

with open(input_json, 'r', encoding='utf-8') as f:
    existing_annotations = json.load(f)

image_id_mapping = {image["file_name"]: image["id"] for image in existing_annotations["images"]}
category_id_mapping = {category["id"]: category["name"] for category in existing_annotations["categories"]}

coco_results = []

for img_name in tqdm(os.listdir(input_folder)):
    if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join(input_folder, img_name)
    if not os.path.exists(img_path):
        continue

    if img_name not in image_id_mapping:
        print(f"Image {img_name} does not have a corresponding image_id in input JSON, skipping.")
        continue

    image_id = image_id_mapping[img_name]

    img = cv2.imread(img_path)
    if img is None:
        print(f"Unable to read image: {img_path}")
        continue

    results = model(img)

    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, score, class_id = box.cpu().numpy()
            class_id = int(class_id)

            if score < confidence_threshold:
                continue

            if class_id not in category_id_mapping:
                print(f"Undefined category ID {class_id}, skipping this detection box.")
                continue

            width_box = x2 - x1
            height_box = y2 - y1

            result_dict = {
                "image_id": image_id,
                "category_id": class_id + 1,
                "category_name": category_id_mapping[class_id],
                "bbox": [float(x1), float(y1), float(width_box), float(height_box)],
                "score": float(score)
            }

            coco_results.append(result_dict)

with open(output_result_json, "w", encoding='utf-8') as f:
    json.dump(coco_results, f, indent=4, ensure_ascii=False)

print(f"COCO result file saved to {output_result_json}")
