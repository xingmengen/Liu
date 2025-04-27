from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np

def evaluate_coco_annotations(gt_file, dt_file):
    """
    Evaluate the accuracy of generated COCO annotations against ground truth.

    Parameters:
    gt_file: str - path to the ground truth COCO annotation file
    dt_file: str - path to the generated COCO annotation file

    Returns:
    None - prints evaluation results for AP and AR
    """
    coco_gt = COCO(gt_file)
    coco_dt = coco_gt.loadRes(dt_file)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.imgIds = coco_gt.getImgIds()
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    category_ids = coco_gt.getCatIds()
    category_map = {cat['id']: cat['name'] for cat in coco_gt.loadCats(category_ids)}
    precisions = coco_eval.eval['precision']
    cat_ids = coco_gt.getCatIds()

    ap_per_category = []
    for idx, cat_id in enumerate(cat_ids):
        nm = coco_gt.loadCats(cat_id)[0]
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float('nan')
        ap_per_category.append((nm['name'], ap))

    print("\nAverage Precision (AP) per category:")
    for category_name, ap in ap_per_category:
        print(f"Category: {category_name}, AP: {ap:.4f}")


if __name__ == "__main__":
    gt_file = r'D:\VisDrone2019-DET\VisDrone2019-DET-val\Global\test.json'
    dt_file = r'D:\VisDrone2019-DET\VisDrone2019-DET-val\8\Global_swin_tiny_test_results.bbox.json'
    evaluate_coco_annotations(gt_file, dt_file)
