import numpy as np
import cv2
import os
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from shapely.geometry import box

def find_optimal_densest_region_using_dbscan(coordinates, image_shape, original_bboxes, min_samples):
    coords = np.array(coordinates)
    if coords.size == 0:
        return [], [], []

    eps = calculate_optimal_eps(coords, min_samples)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(coords)

    unique_labels = set(labels)
    unique_labels.discard(-1)
    if not unique_labels:
        return [], [], []

    best_windows, best_counts, best_density_ratios = [], [], []
    for label in unique_labels:
        cluster_bboxes = [original_bboxes[i] for i, l in enumerate(labels) if l == label]
        xs = [b[0] for b in cluster_bboxes]
        ys = [b[1] for b in cluster_bboxes]
        ws = [b[2] for b in cluster_bboxes]
        hs = [b[3] for b in cluster_bboxes]

        min_x, min_y = min(xs), min(ys)
        max_x = max(x + w for x, w in zip(xs, ws))
        max_y = max(y + h for y, h in zip(ys, hs))
        window = (min_x, min_y, max_x - min_x, max_y - min_y)
        window_area = (max_x - min_x) * (max_y - min_y)

        # merge target areas
        merged = None
        for b in cluster_bboxes:
            rect = box(b[0], b[1], b[0] + b[2], b[1] + b[3])
            merged = rect if merged is None else merged.union(rect)
        total_area = merged.area if merged is not None else 0
        density = total_area / window_area if window_area > 0 else 0

        best_windows.append(window)
        best_counts.append(len(cluster_bboxes))
        best_density_ratios.append(density)

    return best_windows, best_counts, best_density_ratios

def calculate_optimal_eps(coords, min_samples):
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(coords)
    distances, _ = neighbors.kneighbors(coords)
    d = np.sort(distances[:, -1])
    return float(np.mean(d) + 0.5 * np.std(d))

def draw_bounding_box(image, windows=None, counts=None, density_ratios=None,
                      original_bboxes=None, color=(0, 255, 0), thickness=2):
    img = image.copy()
    if original_bboxes:
        for x, y, w, h in original_bboxes:
            if w > 0 and h > 0:
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), thickness)
    if windows:
        for (x, y, w, h), cnt, dens in zip(windows, counts, density_ratios):
            x, y, w, h = map(int, (x, y, w, h))
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
            cv2.putText(img, f"Count: {cnt}, Density: {dens:.6f}", (x, max(y-10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img

def read_annotations_from_txt(txt_file, img_w, img_h):
    coords, bboxes, classes = [], [], []
    with open(txt_file) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx = float(parts[1]) * img_w
            cy = float(parts[2]) * img_h
            w = float(parts[3]) * img_w
            h = float(parts[4]) * img_h
            coords.append((cx, cy))
            bboxes.append([cx - w/2, cy - h/2, w, h])
            classes.append(cls)
    return coords, bboxes, classes

def gather_split_result(img_path, result, out_img_dir, out_anno_dir, anno_dir, img_w, img_h):
    windows, counts, densities = result
    img = cv2.imread(img_path)
    if img is None:
        return
    base = os.path.splitext(os.path.basename(img_path))[0]
    for (x, y, w, h), _, _ in zip(windows, counts, densities):
        top, left, bot, right = int(y), int(x), int(y+h), int(x+w)
        crop = img[top:bot, left:right]
        if crop.size == 0 or crop.shape[0]*crop.shape[1] < 4900:
            continue
        fname = f"{top}_{left}_{bot}_{right}_{base}.jpg"
        cv2.imwrite(os.path.join(out_img_dir, fname), crop)
        anno_out = os.path.join(out_anno_dir, f"{top}_{left}_{bot}_{right}_{base}.txt")
        with open(anno_out, 'w') as fo:
            ch, cw = crop.shape[:2]
            for (bx, by, bw, bh), cls in zip(*read_annotations_from_txt(os.path.join(anno_dir, base+'.txt'), img_w, img_h)[1:]):
                if bx < left or by < top or bx+bw > right or by+bh > bot:
                    continue
                ncx = (bx + bw/2 - left) / cw
                ncy = (by + bh/2 - top) / ch
                nw, nh = bw/cw, bh/ch
                fo.write(f"{cls} {min(max(ncx,0),1):.6f} {min(max(ncy,0),1):.6f} "
                         f"{min(max(nw,0),1):.6f} {min(max(nh,0),1):.6f}\n")

def process_all_images_in_folder(image_dir, txt_dir, out_img_dir, out_anno_dir):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_anno_dir, exist_ok=True)
    for fname in os.listdir(image_dir):
        if not fname.lower().endswith(('.jpg','.jpeg','.png')):
            continue
        img_path = os.path.join(image_dir, fname)
        base = os.path.splitext(fname)[0]
        txt_path = os.path.join(txt_dir, base+'.txt')
        if not os.path.exists(txt_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        coords, bboxes, cls_ids = read_annotations_from_txt(txt_path, w, h)
        total = len(coords)
        k = max(5, int(0.05 * total))
        if total < k:
            continue
        result = find_optimal_densest_region_using_dbscan(coords, img.shape, bboxes, k)
        gather_split_result(img_path, result, out_img_dir, out_anno_dir, txt_dir, w, h)
        print(f"Processed image: {fname} (k={k})")

def main():
    image_dir = r'D:\VisDrone2019-DET\train\images'
    txt_dir = r'D:\VisDrone2019-DET\train\annotations'
    out_img_dir = r'D:\VisDrone2019-DET\C\train\images'
    out_anno_dir = r'D:\VisDrone2019-DET\C\train\annotations'
    process_all_images_in_folder(image_dir, txt_dir, out_img_dir, out_anno_dir)

if __name__ == '__main__':
    main()
