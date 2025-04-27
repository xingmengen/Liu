import json
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import os
import math
import numpy as np


def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes."""
    x1_min, y1_min, w1, h1 = box1
    x1_max, y1_max = x1_min + w1, y1_min + h1

    x2_min, y2_min, w2, h2 = box2
    x2_max, y2_max = x2_min + w2, y2_min + h2

    inter_min_x = max(x1_min, x2_min)
    inter_min_y = max(y1_min, y2_min)
    inter_max_x = min(x1_max, x2_max)
    inter_max_y = min(y1_max, y2_max)

    inter_width = max(0, inter_max_x - inter_min_x)
    inter_height = max(0, inter_max_y - inter_min_y)
    inter_area = inter_width * inter_height

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0


def filter_overlapping_annotations(annotations, iou_threshold=0.6):
    """Filter out highly overlapping annotations."""
    annotations = sorted(annotations, key=lambda ann: ann['bbox'][2] * ann['bbox'][3], reverse=True)
    filtered = []
    for ann in annotations:
        keep = True
        for kept in filtered:
            if compute_iou(ann['bbox'], kept['bbox']) > iou_threshold:
                keep = False
                break
        if keep:
            filtered.append(ann)
    return filtered


def visualize_coco_detections(original_annotation_path, result_annotation_path, image_path, selected_categories=None):
    # File existence check
    for path in [original_annotation_path, result_annotation_path, image_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    # Load original COCO data
    original_coco = COCO(original_annotation_path)

    # Get image ID
    image_filename = os.path.basename(image_path)
    img_id = next((img['id'] for img in original_coco.loadImgs(original_coco.getImgIds())
                   if img['file_name'] == image_filename), None)
    if img_id is None:
        raise ValueError(f"Image {image_filename} not found")

    # Load result data
    with open(result_annotation_path, 'r') as f:
        result_data = json.load(f)
    image_annotations = [ann for ann in (result_data['annotations'] if isinstance(result_data, dict) else result_data)
                         if ann['image_id'] == img_id]

    # Filter annotations
    filtered_anns = filter_overlapping_annotations(image_annotations)
    if not filtered_anns:
        print("No valid annotations")
        return

    # Handle categories
    categories = original_coco.loadCats(original_coco.getCatIds())
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
    if selected_categories:
        selected_ids = [cat['id'] for cat in categories if cat['name'] in selected_categories]
        result_anns = [ann for ann in filtered_anns if ann['category_id'] in selected_ids]
    else:
        result_anns = filtered_anns

    # Load image
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    # Preprocess annotations
    for ann in result_anns:
        bbox = ann['bbox']
        ann['center'] = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
        ann['area'] = bbox[2] * bbox[3]

    # Compute distance matrix
    total = len(result_anns)
    dist_matrix = np.zeros((total, total))
    for i in range(total):
        for j in range(total):
            if i != j:
                dx = result_anns[i]['center'][0] - result_anns[j]['center'][0]
                dy = result_anns[i]['center'][1] - result_anns[j]['center'][1]
                dist_matrix[i][j] = np.sqrt(dx ** 2 + dy ** 2)

    # Core algorithm logic
    start_idx = np.argmin(dist_matrix.sum(axis=1))
    linked = [start_idx]
    remaining = set(range(total)) - {start_idx}
    linking_order = [start_idx]

    while remaining:
        min_dist = np.inf
        next_idx = None
        for idx in remaining:
            current_min = min([dist_matrix[idx][i] for i in linked])
            if current_min < min_dist:
                min_dist = current_min
                next_idx = idx
        if next_idx is not None:
            linked.append(next_idx)
            linking_order.append(next_idx)
            remaining.remove(next_idx)

    # Compute metrics
    threshold = max(1, math.ceil(total * 0.3))
    max_combined = 0
    best_traditional = best_pixel = 0
    best_rect = None
    current_group = []

    for idx in linking_order:
        current_group.append(result_anns[idx])
        if len(current_group) < threshold:
            continue

        # Compute minimal enclosing rectangle
        x_min = min(ann['bbox'][0] for ann in current_group)
        y_min = min(ann['bbox'][1] for ann in current_group)
        x_max = max(ann['bbox'][0] + ann['bbox'][2] for ann in current_group)
        y_max = max(ann['bbox'][1] + ann['bbox'][3] for ann in current_group)
        rect_w = x_max - x_min
        rect_h = y_max - y_min
        rect_area = rect_w * rect_h

        # Traditional area ratio
        sum_area = sum(ann['area'] for ann in current_group)
        traditional = sum_area / rect_area if rect_area else 0

        # Pixel ratio
        mask = Image.new('1', (img_w, img_h), 0)
        draw = ImageDraw.Draw(mask)
        for ann in current_group:
            b = ann['bbox']
            l = max(0, int(b[0]))
            t = max(0, int(b[1]))
            r = min(img_w, int(b[0] + b[2]))
            btm = min(img_h, int(b[1] + b[3]))
            draw.rectangle([l, t, r, btm], fill=1)
        pixel_area = np.array(mask).sum()
        pixel = pixel_area / rect_area if rect_area else 0

        # Combined score
        combined = traditional + 0.8 * pixel

        if combined > max_combined:
            max_combined = combined
            best_traditional = traditional
            best_pixel = pixel
            best_rect = (x_min, y_min, rect_w, rect_h)

    # Visualization
    fig, ax = plt.subplots(figsize=(img_w / 300, img_h / 300), dpi=300)
    ax.imshow(image)

    # Draw best bounding rectangle
    if best_rect:
        color_levels = ['#33FF00', '#66FF00', '#99FF00', '#CCFF00', '#FFFF00',
                        '#FFCC00', '#FF9900', '#FF6600', '#FF3300', '#FF0000']
        norm_ratio = max(0, min(max_combined, 1.8)) / 1.8
        color = color_levels[min(int(norm_ratio * 10), 9)]

        rect = patches.Rectangle(
            (best_rect[0], best_rect[1]), best_rect[2], best_rect[3],
            linewidth=2, edgecolor=color, facecolor=color, alpha=0.3
        )
        ax.add_patch(rect)

        # Add score text
        text = f"Normalized Score: {max_combined / 1.8:.2f}"
        ax.text(
            best_rect[0], best_rect[1] - 40, text,
            fontsize=7, color='red',
            bbox=dict(facecolor='white', alpha=0.8)
        )

    plt.axis('off')
    plt.tight_layout()

    # Save results
    save_dir = 'FenJi3'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, image_filename)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()

    print(f"Traditional area ratio: {best_traditional:.2f}")
    print(f"Pixel ratio: {best_pixel:.2f}")
    print(f"Combined score: {max_combined:.2f}")


# Example usage
if __name__ == "__main__":
    original_anno = r'D:\VisDrone2019-DET\VisDrone2019-DET-val\Global\val.json'
    result_anno = r'D:\VisDrone2019-DET\VisDrone2019-DET-val\final_fusion_result.bbox.json'
    image_file = r'D:\VisDrone2019-DET\VisDrone2019-DET-val\Global\images\0000295_01800_d_0000030.jpg'

    selected_classes = ["pedestrian", "people", "bicycle", "car", "van",
                        "truck", "tricycle", "awning-tricycle", "bus", "motor"]

    visualize_coco_detections(original_anno, result_anno, image_file, selected_classes)
