"""Evaluate predicted instance masks against ground truth.

Example:
    python evaluate.py --pred-dir outputs/examples --gt-dir examples/test/gt

Predictions are ``masks_<stem>.npy`` files written by ``segment_floorplans.py``.
Ground truth is ``<stem>.png`` (instance label map, 0 = background) or
``<stem>.npy`` ((N, H, W) masks, the format written by ``extract_cubicasa_data.py``).
Instances match at IoU >= 0.5 with greedy one-to-one matching.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from fewshot_sam import evaluate_image, load_instances, summarize

COLUMNS = ['image', 'n_gt', 'n_pred', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'matched_iou', 'pixel_miou']


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pred-dir', required=True, help='Directory with masks_<stem>.npy files.')
    parser.add_argument('--gt-dir', required=True, help='Directory with <stem>.png or <stem>.npy ground truth.')
    parser.add_argument('--csv', default=None, help='Optional path for a per-image CSV.')
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    rows, skipped = [], []
    for pred_path in sorted(Path(args.pred_dir).glob('masks_*.npy')):
        stem = pred_path.stem[len('masks_'):]
        gt_path = next((p for p in (gt_dir / f'{stem}.png', gt_dir / f'{stem}.npy') if p.exists()), None)
        if gt_path is None:
            skipped.append((stem, 'no ground truth'))
            continue
        pred, gt = np.load(pred_path), load_instances(gt_path)
        if len(pred) and len(gt) and pred.shape[1:] != gt.shape[1:]:
            skipped.append((stem, f'size mismatch {pred.shape[1:]} vs {gt.shape[1:]}'))
            continue
        rows.append(dict(image=stem, **evaluate_image(pred, gt)))

    if not rows:
        raise SystemExit('nothing to evaluate')
    rows.sort(key=lambda r: (0, int(r['image'])) if r['image'].isdigit() else (1, r['image']))
    print(f"{'image':>8} {'gt':>4} {'pred':>5} {'tp':>4} {'fp':>4} {'fn':>4} {'P':>6} {'R':>6} {'F1':>6} {'mIoU':>6}")
    for r in rows:
        print(f"{r['image']:>8} {r['n_gt']:>4} {r['n_pred']:>5} {r['tp']:>4} {r['fp']:>4} {r['fn']:>4} "
              f"{r['precision']:6.3f} {r['recall']:6.3f} {r['f1']:6.3f} {r['matched_iou']:6.3f}")
    s = summarize(rows)
    print(f"\n{s['images']} images, tp={s['tp']} fp={s['fp']} fn={s['fn']}")
    print(f"micro:  precision {s['precision']:.4f}  recall {s['recall']:.4f}  F1 {s['f1']:.4f}")
    print(f"macro:  precision {s['macro_precision']:.4f}  recall {s['macro_recall']:.4f}  F1 {s['macro_f1']:.4f}")
    print(f"matched-instance IoU {s['matched_iou']:.4f}  pixel mIoU {s['pixel_miou']:.4f}  "
          f"pixel accuracy {s['pixel_accuracy']:.4f}")
    if skipped:
        print('skipped:', ', '.join(f'{stem} ({why})' for stem, why in skipped))
    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
