"""Loading and saving images, reference masks and instance masks."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

PathLike = Union[str, Path]
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def _natural_key(path: Path):
    stem = path.stem
    return (0, int(stem), stem) if stem.isdigit() else (1, 0, stem)


def list_images(directory: PathLike) -> List[Path]:
    """Image files in a directory, numeric stems in numeric order."""
    files = [p for p in Path(directory).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=_natural_key)


def load_image(path: PathLike) -> np.ndarray:
    """Read an image as RGB uint8."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask(path: PathLike) -> np.ndarray:
    """Read a reference mask as uint8 grayscale; any non-zero pixel is foreground."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask


def load_references(image_dir: PathLike, mask_dir: PathLike,
                    names: Optional[Sequence[str]] = None) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Pair reference images with masks of the same stem: [(name, RGB image, mask), ...].

    Masks are looked up as ``<mask_dir>/<stem>.png``.  Pass ``names`` to use a subset.
    """
    mask_dir = Path(mask_dir)
    refs = []
    for image_path in list_images(image_dir):
        if names is not None and image_path.stem not in names:
            continue
        mask_path = mask_dir / f'{image_path.stem}.png'
        if not mask_path.exists():
            continue
        image, mask = load_image(image_path), load_mask(mask_path)
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f'{image_path.name}: image {image.shape[:2]} and mask {mask.shape[:2]} differ in size')
        refs.append((image_path.stem, image, mask))
    if not refs:
        raise FileNotFoundError(f'no image/mask pairs found in {image_dir} and {mask_dir}')
    return refs


def load_instances(path: PathLike) -> np.ndarray:
    """Load instance masks as a (N, H, W) bool array.

    ``.png``: instance label map, 0 = background and 1..N = instances.
    ``.npy``: array of shape (N, H, W); any non-zero value is foreground.
    """
    path = Path(path)
    if path.suffix == '.npy':
        return np.load(path) > 0
    labels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if labels is None:
        raise FileNotFoundError(path)
    if labels.ndim == 3:
        labels = labels[..., 0]
    ids = [int(v) for v in np.unique(labels) if v != 0]
    if not ids:
        return np.zeros((0, *labels.shape), dtype=bool)
    return np.stack([labels == v for v in ids])


def save_instance_labels(masks: np.ndarray, path: PathLike) -> None:
    """Save non-overlapping (N, H, W) masks as a label PNG (0 = background, i + 1 = mask i)."""
    masks = np.asarray(masks).astype(bool)
    if masks.shape[0] > 65535:
        raise ValueError('too many instances for a PNG label map')
    if masks.shape[0] and (masks.sum(0) > 1).any():
        raise ValueError('instances overlap; save them as .npy instead')
    dtype = np.uint8 if masks.shape[0] < 256 else np.uint16
    labels = np.zeros(masks.shape[1:], dtype=dtype)
    for i, m in enumerate(masks):
        labels[m] = i + 1
    cv2.imwrite(str(path), labels)
