import numpy as np
import cv2
import os
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from shapely.geometry import box

def find_optimal_densest_region_using_dbscan(coordinates, image_shape, original_bboxes, min_samples):
    height, width = image_shape[:2]
    coords = np.array(coordinates)

    if coords.size == 0 or len(coords) < min_samples:
        return [], [], []

    eps = calculate_optimal_eps(coords, min_samples)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(coords)

    unique_labels = set(labels)
    unique_labels.discard(-1)

    if not unique_labels:
        return [], [], []

    best_windows = []
    best_counts = []
    best_density_ratios = []

    for label in unique_labels:
        cluster_bboxes = [original_bboxes[i] for i, lbl in enumerate(labels) if lbl == label]

        min_x = min(b[0] for b in cluster_bboxes)
        min_y = min(b[1] for b in cluster_bboxes)
        max_x = max(b[0] + b[2] for b in cluster_bboxes)
        max_y = max(b[1] + b[3] for b in cluster_bboxes)

        window = (min_x, min_y, max_x - min_x, max_y - min_y)
        window_area = (max_x - min_x) * (max_y - min_y)

        merged = None
        for bbox in cluster_bboxes:
            geom = box(bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
            merged = geom if merged is None else merged.union(geom)

        total_target_area = merged.area if merged is not None else 0
        density_ratio = total_target_area / window_area if window_area > 0 else 0

        best_windows.append(window)
        best_counts.append(len(cluster_bboxes))
        best_density_ratios.append(density_ratio)

    return best_windows, best_counts, best_density_ratios

def calculate_optimal_eps(coords, min_samples):
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(coords)
    distances, _ = neighbors.kneighbors(coords)
    distances = np.sort(distances[:, -1])
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    return mean_distance + std_distance * 0.5

def draw_bounding_box(image, windows=None, counts=None, density_ratios=None,
                      original_bboxes=None, color=(0, 255, 0), thickness=2):
    img = image.copy()

    if original_bboxes:
        for x, y, w, h in original_bboxes:
            if x >= 0 and y >= 0 and w > 0 and h > 0:
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), thickness)

    if windows:
        for window, count, density in zip(windows, counts, density_ratios):
            x, y, w, h = map(int, window)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
            label = f'Count: {count}, Density: {density:.6f}'
            cv2.putText(img, label, (x, max(y - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img

def read_annotations_from_txt(txt_file, image_shape):
    coords, bboxes, confs, cids = [], [], [], []
    h, w = image_shape[:2]
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cid = int(parts[0])
            xc = float(parts[1]) * w
            yc = float(parts[2]) * h
            bw = float(parts[3]) * w
            bh = float(parts[4]) * h
            left = xc - bw / 2
            top = yc - bh / 2
            coords.append((xc, yc))
            bboxes.append([left, top, bw, bh])
            confs.append(1.0)
            cids.append(cid)
    return coords, bboxes, confs, cids

def gather_split_result(img_path, result, out_img_dir, out_anno_dir, orig_txt_dir):
    windows, counts, densities = result
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: could not load {img_path}")
        return
    base = os.path.basename(img_path)
    txt = os.path.splitext(base)[0] + '.txt'
    txt_path = os.path.join(orig_txt_dir, txt)
    if not os.path.exists(txt_path):
        print(f"Error: annotation {txt_path} not found")
        return
    h, w = img.shape[:2]
    ann = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cid = int(parts[0])
            xc = float(parts[1]) * w
            yc = float(parts[2]) * h
            bw = float(parts[3]) * w
            bh = float(parts[4]) * h
            left = xc - bw / 2
            top = yc - bh / 2
            ann.append((left, top, bw, bh, cid))

    for i, (win, cnt, dens) in enumerate(zip(windows, counts, densities), 1):
        x, y, cw, ch = win
        left, top, right, bottom = map(int, [x, y, x + cw, y + ch])
        crop = img[top:bottom, left:right]
        if crop.size == 0 or crop.shape[0]*crop.shape[1] < 70*70:
            continue
        crop_name = f"{top}_{left}_{bottom}_{right}_{base}"
        cv2.imwrite(os.path.join(out_img_dir, crop_name), crop)
        anno_name = f"{top}_{left}_{bottom}_{right}_{os.path.splitext(base)[0]}.txt"
        cw_w, cw_h = right-left, bottom-top
        with open(os.path.join(out_anno_dir, anno_name), 'w') as out:
            for ol, ot, ow, oh, cid in ann:
                if ol<left or ot<top or ol+ow>right or ot+oh>bottom:
                    continue
                nl = ol - left
                nt = ot - top
                xcn = (nl + ow/2) / cw_w
                ycn = (nt + oh/2) / cw_h
                out.write(f"{cid} {xcn:.6f} {ycn:.6f} {ow/cw_w:.6f} {oh/cw_h:.6f}\n")

def process_all_images_in_folder(image_dir, txt_dir, out_img_dir, out_anno_dir):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_anno_dir, exist_ok=True)

    for fname in os.listdir(image_dir):
        if not fname.lower().endswith('.jpg'):
            continue
        img_path = os.path.join(image_dir, fname)
        txt_path = os.path.join(txt_dir, os.path.splitext(fname)[0] + '.txt')
        if not os.path.exists(txt_path):
            print(f"Warning: missing {txt_path}")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: could not load {img_path}")
            continue
        coords, bboxes, confs, cids = read_annotations_from_txt(txt_path, img.shape)
        n = len(coords)
        dynamic_min = int(max(5, n * 0.05))
        if n < dynamic_min:
            print(f"Skipping {fname}: targets fewer than {dynamic_min}")
            continue
        result = find_optimal_densest_region_using_dbscan(coords, img.shape, bboxes, dynamic_min)
        gather_split_result(img_path, result, out_img_dir, out_anno_dir, txt_dir)
        print(f"Processed {fname}")

def main():
    image_dir = r'D:\GitHub3_1\UAVDT\dataset_uavdt\val\images'
    txt_dir = r'D:\GitHub3_1\UAVDT\dataset_uavdt\val\labels'
    output_img_dir = r'D:\GitHub3_1\UAVDT\dataset_uavdt\CaiJian\val\images'
    output_anno_dir = r'D:\GitHub3_1\UAVDT\dataset_uavdt\CaiJian\val\labels'

    process_all_images_in_folder(image_dir, txt_dir, output_img_dir, output_anno_dir)

if __name__ == "__main__":
    main()