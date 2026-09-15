"""Overlays for masks and evaluation results."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .metrics import match_instances

TP_COLOR = (46, 170, 80)
FP_COLOR = (220, 60, 50)
FN_COLOR = (60, 110, 230)


def _resize(image: np.ndarray, max_side: Optional[int]) -> np.ndarray:
    if not max_side or max(image.shape[:2]) <= max_side:
        return image
    s = max_side / max(image.shape[:2])
    return cv2.resize(image, (round(image.shape[1] * s), round(image.shape[0] * s)), interpolation=cv2.INTER_AREA)


def _paint(vis: np.ndarray, mask: np.ndarray, color, alpha: float) -> None:
    if mask.shape != vis.shape[:2]:
        mask = cv2.resize(mask.astype(np.uint8), (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    vis[mask] = ((1 - alpha) * vis[mask] + alpha * np.asarray(color)).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, tuple(int(c * 0.6) for c in color), 1)


def overlay_masks(image: np.ndarray, masks: np.ndarray, alpha: float = 0.5, max_side: Optional[int] = None,
                  seed: int = 0) -> np.ndarray:
    """Draw every mask in a distinct random color."""
    vis = _resize(image, max_side).copy()
    rng = np.random.RandomState(seed)
    for m in masks:
        _paint(vis, np.asarray(m, dtype=bool), rng.randint(40, 230, 3), alpha)
    return vis


def overlay_evaluation(image: np.ndarray, pred: np.ndarray, gt: np.ndarray, alpha: float = 0.5,
                       max_side: Optional[int] = None) -> np.ndarray:
    """Green = matched prediction, red = false positive, blue = missed ground-truth instance."""
    vis = _resize(image, max_side).copy()
    matches = match_instances(pred, gt)
    matched_p = {i for i, _, _ in matches}
    matched_g = {j for _, j, _ in matches}
    for j, g in enumerate(gt):
        if j not in matched_g:
            _paint(vis, np.asarray(g, dtype=bool), FN_COLOR, alpha)
    for i, p in enumerate(pred):
        _paint(vis, np.asarray(p, dtype=bool), TP_COLOR if i in matched_p else FP_COLOR, alpha)
    return vis
