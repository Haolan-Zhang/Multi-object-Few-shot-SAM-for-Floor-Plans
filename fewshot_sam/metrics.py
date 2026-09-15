"""Instance-level evaluation.

A predicted mask and a ground-truth instance match when their IoU is at least
0.5.  Pairs are matched greedily in order of decreasing IoU, each prediction
and each ground-truth instance at most once.  Unmatched predictions are false
positives, unmatched ground-truth instances are false negatives.

Pixel metrics ignore instance identity and compare the union of predicted
masks with the union of ground-truth masks.  ``pixel_miou`` is the mean of the
foreground IoU and the background IoU.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

IOU_THRESHOLD = 0.5


def iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """(P, H, W) and (G, H, W) bool arrays -> (P, G) IoU matrix."""
    pred = np.asarray(pred).astype(bool)
    gt = np.asarray(gt).astype(bool)
    out = np.zeros((len(pred), len(gt)))
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            union = np.count_nonzero(p | g)
            out[i, j] = np.count_nonzero(p & g) / union if union else 1.0
    return out


def match_instances(pred: np.ndarray, gt: np.ndarray, threshold: float = IOU_THRESHOLD) -> List[Tuple[int, int, float]]:
    """Greedy one-to-one matching.  Returns [(pred index, gt index, IoU), ...]."""
    if len(pred) == 0 or len(gt) == 0:
        return []
    ious = iou_matrix(pred, gt)
    pairs = sorted(((ious[i, j], i, j) for i in range(ious.shape[0]) for j in range(ious.shape[1])
                    if ious[i, j] >= threshold), reverse=True)
    used_p, used_g, matches = set(), set(), []
    for v, i, j in pairs:
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            matches.append((i, j, float(v)))
    return matches


def pixel_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    H, W = np.asarray(gt).shape[1:]
    p = np.asarray(pred).astype(bool).any(0) if len(pred) else np.zeros((H, W), dtype=bool)
    g = np.asarray(gt).astype(bool).any(0) if len(gt) else np.zeros((H, W), dtype=bool)
    tp = np.count_nonzero(p & g)
    fp = np.count_nonzero(p & ~g)
    fn = np.count_nonzero(~p & g)
    tn = np.count_nonzero(~p & ~g)
    iou_fg = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    iou_bg = tn / (tn + fn + fp) if tn + fn + fp else 1.0
    return dict(pixel_accuracy=(tp + tn) / p.size, pixel_miou=(iou_fg + iou_bg) / 2)


def evaluate_image(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Metrics for one image.  ``pred`` and ``gt`` are (N, H, W) mask stacks of the same H, W."""
    n_pred, n_gt = len(pred), len(gt)
    if n_pred and n_gt and np.asarray(pred).shape[1:] != np.asarray(gt).shape[1:]:
        raise ValueError(f'prediction {np.asarray(pred).shape[1:]} and ground truth {np.asarray(gt).shape[1:]} differ in size')
    matches = match_instances(pred, gt)
    tp = len(matches)
    fp, fn = n_pred - tp, n_gt - tp
    if n_pred == 0 or n_gt == 0:
        precision = 0.0 if n_pred > 0 else 1.0
        recall = 0.0 if n_gt > 0 else 1.0
        f1 = 0.0
    else:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    matched_iou = float(np.mean([v for _, _, v in matches])) if matches else 0.0
    return dict(n_pred=n_pred, n_gt=n_gt, tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1,
                matched_iou=matched_iou, **pixel_metrics(pred, gt))


def summarize(results: Iterable[Dict[str, float]]) -> Dict[str, float]:
    """Micro-averaged (pooled counts) and macro-averaged (mean over images) metrics."""
    results = list(results)
    if not results:
        return {}
    tp = sum(r['tp'] for r in results)
    fp = sum(r['fp'] for r in results)
    fn = sum(r['fn'] for r in results)
    p = tp / (tp + fp) if tp + fp else 0.0
    r_ = tp / (tp + fn) if tp + fn else 0.0
    mean = lambda k: float(np.mean([r[k] for r in results]))
    return dict(images=len(results), tp=tp, fp=fp, fn=fn,
                precision=p, recall=r_, f1=2 * p * r_ / (p + r_) if p + r_ else 0.0,
                macro_precision=mean('precision'), macro_recall=mean('recall'), macro_f1=mean('f1'),
                matched_iou=mean('matched_iou'), pixel_accuracy=mean('pixel_accuracy'), pixel_miou=mean('pixel_miou'))
