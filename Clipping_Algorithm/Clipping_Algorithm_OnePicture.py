import json
import numpy as np
import cv2
import os
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from shapely.geometry import box

def find_optimal_densest_region_using_dbscan(coordinates, image_shape, original_bboxes, min_samples=None):

    height, width = image_shape[:2]
    coords = np.array(coordinates)
    if coords.size == 0:
        return [], [], []

    # Automatically set min_samples to the larger of 5 or 5% of total targets
    auto_min = max(5, int(0.05 * len(coords)))
    if min_samples is None:
        min_samples = auto_min
    else:
        min_samples = max(min_samples, auto_min)

    eps = calculate_optimal_eps(coords, min_samples)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(coords)
    unique_labels = set(labels) - {-1}
    if not unique_labels:
        return [], [], []

    best_windows, best_counts, best_density_ratios = [], [], []

    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        cluster_bboxes = [original_bboxes[i] for i in cluster_indices]

        xs = [b[0] for b in cluster_bboxes]
        ys = [b[1] for b in cluster_bboxes]
        ws = [b[2] for b in cluster_bboxes]
        hs = [b[3] for b in cluster_bboxes]

        min_x, min_y = min(xs), min(ys)
        max_x = max(x + w for x, w in zip(xs, ws))
        max_y = max(y + h for y, h in zip(ys, hs))
        window = (min_x, min_y, max_x - min_x, max_y - min_y)

        window_area = (max_x - min_x) * (max_y - min_y)
        target_polygons = [box(x, y, x + w, y + h) for x, y, w, h in cluster_bboxes]
        merged_area = target_polygons[0] if target_polygons else None
        for poly in target_polygons[1:]:
            merged_area = merged_area.union(poly)
        total_area = merged_area.area if merged_area else 0
        density_ratio = total_area / window_area if window_area > 0 else 0

        best_windows.append(window)
        best_counts.append(len(cluster_bboxes))
        best_density_ratios.append(density_ratio)

    return best_windows, best_counts, best_density_ratios

def calculate_optimal_eps(coords, min_samples):
    """
    Determine an appropriate eps value based on KNN distances.

    Args:
        coords: array of (x, y) coordinates
        min_samples: number of neighbors for distance calculation

    Returns:
        eps: selected neighborhood radius
    """
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(coords)
    distances, _ = neighbors.kneighbors(coords)
    sorted_kd = np.sort(distances[:, -1])
    mean_d = np.mean(sorted_kd)
    std_d = np.std(sorted_kd)
    return mean_d + 0.5 * std_d

def draw_bounding_box(image, windows=None, counts=None, density_ratios=None,
                      original_bboxes=None, color=(255, 0, 0), thickness=1):
    """
    Draw bounding boxes and annotate counts and density ratios.
    """
    img = image.copy()
    overlay = img.copy()
    if original_bboxes:
        for x, y, w, h in original_bboxes:
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
    if windows:
        for x, y, w, h in windows:
            cv2.rectangle(overlay, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    for (x, y, w, h), count, _ in zip(windows, counts, density_ratios):
        x, y, w, h = map(int, (x, y, w, h))
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        label = f'Count: {count}'
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x, y - lh - baseline - 5), (x + lw, y), color, -1)
        cv2.putText(img, label, (x, y - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img

def get_image_id_by_filename(original_coco_annotation, filename):
    """Find the image_id for a given filename in a COCO annotation."""
    for img in original_coco_annotation.get('images', []):
        if img.get('file_name') == filename:
            return img.get('id')
    return None

def main():
    original_coco_annotation_file = './val.json'
    result_annotation_file = './final_fusion_result.bbox.json'
    image_dir = './images'
    image_filename = '0000295_01800_d_0000030.jpg'

    if not os.path.exists(original_coco_annotation_file):
        print(f"Error: Original COCO annotation file not found at {original_coco_annotation_file}.")
        return
    if not os.path.exists(result_annotation_file):
        print(f"Error: COCO result annotation file not found at {result_annotation_file}.")
        return
    image_path = os.path.join(image_dir, image_filename)
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_filename} not found in {image_dir}.")
        return

    with open(original_coco_annotation_file, 'r', encoding='utf-8') as f:
        original_coco = json.load(f)
    image_id = get_image_id_by_filename(original_coco, image_filename)
    if image_id is None:
        print(f"Error: Filename {image_filename} not found in original annotations.")
        return

    img_info = next((img for img in original_coco['images'] if img['id'] == image_id), None)
    if img_info is None:
        print(f"Error: No image info for id {image_id}.")
        return

    image = cv2.imread(os.path.join(image_dir, img_info['file_name']))
    if image is None:
        print(f"Error: Could not load image {image_path}.")
        return

    with open(result_annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    preds = [ann for ann in annotations if ann['image_id'] == image_id and ann.get('score', 0) >= 0.1]
    coords, original_bboxes = [], []
    for ann in preds:
        x, y, w, h = ann['bbox']
        coords.append((x + w/2, y + h/2))
        original_bboxes.append([x, y, w, h])
    if not coords:
        print("No targets detected.")
        return

    best_windows, best_counts, best_density_ratios = find_optimal_densest_region_using_dbscan(
        coords, image.shape, original_bboxes
    )
    if not best_windows:
        print("No dense regions found.")
        return

    print(f"Dense regions: {best_windows}")
    for i, (count, ratio) in enumerate(zip(best_counts, best_density_ratios), start=1):
        print(f"Region {i}: Count={count}, DensityRatio={ratio:.4f}")

    img_out = draw_bounding_box(image, best_windows, best_counts, best_density_ratios, original_bboxes)
    output_dir = 'output_images'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{os.path.splitext(image_filename)[0]}_result.jpg')
    cv2.imwrite(output_path, img_out)
    print(f"Saved result to {output_path}")
    cv2.imshow('Result', img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
