"""Training-free few-shot multi-object segmentation of floor plans with SAM and DINOv2."""
from .data import list_images, load_image, load_instances, load_mask, load_references, save_instance_labels
from .metrics import evaluate_image, match_instances, summarize
from .segmenter import FewShotRoomSegmenter, SegmenterConfig
from .viz import overlay_evaluation, overlay_masks

__all__ = [
    'FewShotRoomSegmenter', 'SegmenterConfig',
    'list_images', 'load_image', 'load_mask', 'load_references', 'load_instances', 'save_instance_labels',
    'evaluate_image', 'match_instances', 'summarize',
    'overlay_masks', 'overlay_evaluation',
]
