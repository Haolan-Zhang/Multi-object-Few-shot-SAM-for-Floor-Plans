from __future__ import annotations

import argparse
import io
import os
import sys
import types
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from cairosvg import svg2png
from PIL import Image
from tqdm import tqdm


def _allow_txt_loader_without_lmdb() -> None:
    """FloorplanSVG imports lmdb at module import time, even for txt files."""
    try:
        import lmdb  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["lmdb"] = types.SimpleNamespace(
            open=lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("lmdb is required only when using format='lmdb'")
            )
        )


_allow_txt_loader_without_lmdb()
from floortrans.loaders.svg_loader import FloorplanSVG  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CubiCasa SVG JPGs, room mask NPys, and selected mask PNGs."
    )
    parser.add_argument("--data-path", default="data/cubicasa5k/", help="CubiCasa5k root.")
    parser.add_argument("--txt", default="test.txt", help="Split file under --data-path.")
    parser.add_argument(
        "--svg-dir",
        default="svg_images_out",
        help="Output directory for rendered SVG JPGs.",
    )
    parser.add_argument(
        "--npy-dir",
        default="mask_npys_out",
        help="Output directory for all extracted room-mask arrays.",
    )
    parser.add_argument(
        "--selected-mask-dir",
        default="train_masks_out",
        help="Output directory for one selected room mask per sample.",
    )
    parser.add_argument(
        "--original-size",
        action="store_true",
        help="Use original-size labels instead of scaled labels.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Process at most this many samples.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First dataset index to process.",
    )
    parser.add_argument(
        "--selected-rank",
        type=int,
        default=2,
        help="Save the Nth largest room as PNG. The notebook used 2.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output directories.",
    )
    return parser.parse_args()


def ensure_output_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def output_stem(index: int) -> str:
    return f"{index:02}"


def render_svg(svg_path: Path) -> np.ndarray:
    png = svg2png(bytestring=svg_path.read_bytes(), background_color="white")
    image = Image.open(io.BytesIO(png)).convert("RGB")
    return np.array(image)


def align_image_to_mask(image: np.ndarray, mask_shape: tuple[int, int]) -> np.ndarray:
    """Pad with white or crop so the rendered SVG matches the mask H/W."""
    target_h, target_w = mask_shape
    h, w = image.shape[:2]

    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    if pad_h or pad_w:
        image = np.pad(
            image,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode="constant",
            constant_values=255,
        )

    return image[:target_h, :target_w]


def get_room_masks(label: np.ndarray) -> np.ndarray:
    """Extract room components from the wall channel, matching test.ipynb."""
    room_label = label[0]
    wall_mask = (room_label == 2).astype(np.uint8) * 255

    gray_inv = cv2.bitwise_not(wall_mask)
    _, binary = cv2.threshold(gray_inv, 1, 255, cv2.THRESH_BINARY)
    num_labels, labels_im = cv2.connectedComponents(binary)

    room_masks = []
    for component_id in range(1, num_labels):
        mask = np.zeros_like(wall_mask, dtype=np.uint8)
        mask[labels_im == component_id] = 255
        room_masks.append(mask)

    # The first connected non-wall component is the outside/background region.
    room_masks = room_masks[1:]
    if not room_masks:
        return np.empty((0, *wall_mask.shape), dtype=np.uint8)
    return np.asarray(room_masks, dtype=np.uint8)


def select_room_mask(room_masks: np.ndarray, selected_rank: int) -> np.ndarray:
    if len(room_masks) == 0:
        raise ValueError("No room masks were extracted.")

    rank = max(1, selected_rank)
    rank = min(rank, len(room_masks))
    sorted_masks = sorted(room_masks, key=np.sum)
    return sorted_masks[-rank]


def save_if_allowed(image: Image.Image, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    image.save(path)


def save_npy_if_allowed(array: np.ndarray, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    np.save(path, array)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    loader_data_path = str(data_path)
    if not loader_data_path.endswith(os.sep):
        loader_data_path += os.sep
    svg_dir = Path(args.svg_dir)
    npy_dir = Path(args.npy_dir)
    selected_mask_dir = Path(args.selected_mask_dir)
    ensure_output_dirs([svg_dir, npy_dir, selected_mask_dir])

    dataset = FloorplanSVG(
        loader_data_path,
        args.txt,
        format="txt",
        original_size=args.original_size,
    )

    stop = len(dataset)
    if args.max_samples is not None:
        stop = min(stop, args.start_index + args.max_samples)

    failures: list[tuple[int, str]] = []
    for index in tqdm(range(args.start_index, stop), desc=f"Extracting {args.txt}"):
        stem = output_stem(index)
        try:
            sample = dataset[index]
            label = sample["label"].data.numpy()
            room_masks = get_room_masks(label)
            if len(room_masks) == 0:
                raise ValueError("no room masks found")

            folder = sample["folder"].strip("/")
            svg_path = data_path / folder / "model.svg"
            svg_image = align_image_to_mask(render_svg(svg_path), room_masks[0].shape)

            save_if_allowed(
                Image.fromarray(svg_image),
                svg_dir / f"{stem}.jpg",
                args.overwrite,
            )
            save_npy_if_allowed(room_masks, npy_dir / f"{stem}.npy", args.overwrite)
            save_if_allowed(
                Image.fromarray(select_room_mask(room_masks, args.selected_rank)),
                selected_mask_dir / f"{stem}.png",
                args.overwrite,
            )
        except Exception as exc:  # Keep long batch jobs moving.
            failures.append((index, str(exc)))

    if failures:
        print(f"Completed with {len(failures)} failures:", file=sys.stderr)
        for index, message in failures[:20]:
            print(f"  {index:02}: {message}", file=sys.stderr)
        if len(failures) > 20:
            print(f"  ... {len(failures) - 20} more", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
