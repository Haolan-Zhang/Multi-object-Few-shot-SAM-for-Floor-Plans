"""Segment every room in a folder of floor plans from a few annotated references.

Example:
    python segment_floorplans.py \
        --ref-images examples/references/images \
        --ref-masks examples/references/masks \
        --images examples/test/images \
        --output-dir outputs/examples \
        --save-vis

Writes ``masks_<stem>.npy`` (a (K, H, W) bool array per image), optional
``overlay_<stem>.jpg`` previews and ``config.json`` with every setting used.
Every field of ``SegmenterConfig`` is also available as a flag, for example
``--min-similarity 0.5`` or ``--no-merge-gap``.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import asdict, fields
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from fewshot_sam import FewShotRoomSegmenter, SegmenterConfig, list_images, load_image, load_references, overlay_masks


def parse_args() -> tuple[argparse.Namespace, SegmenterConfig]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    io = parser.add_argument_group('input / output')
    io.add_argument('--ref-images', required=True, help='Directory with reference floor plans.')
    io.add_argument('--ref-masks', required=True,
                    help='Directory with one binary mask per reference, same file stem, .png.')
    io.add_argument('--images', required=True, help='Directory with floor plans to segment.')
    io.add_argument('--output-dir', default='outputs/segmentation', help='Where predictions are written.')
    io.add_argument('--save-vis', action='store_true', help='Also write overlay_<stem>.jpg previews.')
    io.add_argument('--overwrite', action='store_true', help='Recompute images that already have predictions.')

    cfg = parser.add_argument_group('segmenter settings (see fewshot_sam/segmenter.py)')
    defaults = SegmenterConfig()
    for f in fields(SegmenterConfig):
        flag = '--' + f.name.replace('_', '-')
        default = getattr(defaults, f.name)
        help_text = f"{f.metadata.get('help', '')} (default: {default})"
        if isinstance(default, bool):
            cfg.add_argument(flag, dest=f.name, action=argparse.BooleanOptionalAction, default=default, help=help_text)
        else:
            cfg.add_argument(flag, dest=f.name, type=type(default), default=default, help=help_text)

    args = parser.parse_args()
    config = SegmenterConfig(**{f.name: getattr(args, f.name) for f in fields(SegmenterConfig)})
    return args, config


def main() -> None:
    warnings.filterwarnings('ignore', category=FutureWarning)
    args, config = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'config.json').write_text(json.dumps(dict(
        asdict(config), ref_images=args.ref_images, ref_masks=args.ref_masks, images=args.images), indent=2))

    references = load_references(args.ref_images, args.ref_masks)
    print(f'references: {", ".join(name for name, _, _ in references)}')
    segmenter = FewShotRoomSegmenter(config)
    segmenter.set_references([(image, mask) for _, image, mask in references])

    images = list_images(args.images)
    n_masks, seconds = 0, 0.0
    for path in tqdm(images, desc='segmenting'):
        target = out_dir / f'masks_{path.stem}.npy'
        if target.exists() and not args.overwrite:
            continue
        image = load_image(path)
        t0 = time.time()
        masks = segmenter.segment(image)
        seconds += time.time() - t0
        n_masks += len(masks)
        np.save(target, masks)
        if args.save_vis:
            vis = overlay_masks(image, masks)
            cv2.imwrite(str(out_dir / f'overlay_{path.stem}.jpg'), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f'{len(images)} images, {n_masks} instances, {seconds / max(len(images), 1):.2f} s/image -> {out_dir}')


if __name__ == '__main__':
    main()
