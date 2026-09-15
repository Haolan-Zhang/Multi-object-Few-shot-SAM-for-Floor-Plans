"""Training-free few-shot multi-instance segmentation of floor plans.

Given a handful of reference floor plans in which ONE instance (for example one
room) is marked by a binary mask, :class:`FewShotRoomSegmenter` finds every
instance of that concept in new floor plans.

Pipeline for one image
----------------------
1. **Encode once.** SAM (ViT-H) computes the image embedding; DINOv2 computes
   dense semantic features.  Both are shared by all prompts.
2. **Prototype similarity.** The foreground DINOv2 features of all references
   are clustered into a small prototype bank (k-means).  The similarity map is
   the max cosine similarity to the bank, normalised per image to [0, 1].
3. **Dense proposals.** A uniform point grid is decoded by SAM in batches on the
   single embedding.  Every point gives three multimask candidates; only SAM's
   256x256 low-res logits are kept, so thousands of candidates fit in memory.
4. **Validity gates** (all in the 256x256 frame):
   * similarity: mean prototype similarity inside the mask,
   * ink: mean darkness inside the mask (instances have blank interiors),
   * edge alignment: share of the mask boundary lying on dark line work
     (applied to small masks only),
   * consensus: number of near-duplicate candidates proposed by other points,
   * area limits.
5. **Containment-aware NMS.** Candidates are visited in order of
   ``iou_pred * stability * similarity * area^p_area * votes^p_votes``.
   Overlaps and masks inside a kept mask are dropped, thin parts hugging a kept
   mask are absorbed into it, a mask that swallows only sub-parts replaces them,
   a mask that swallows complete instances is dropped, and a final split pass
   replaces a large early mask by the instances it contains.
6. **Open-gap merge.** Kept masks joined by a long, wall-less contact are merged
   (open-plan areas count as one room; rooms behind a door do not).
7. **Refinement.** Masks are upsampled, hole-filled, reduced to their largest
   component and re-prompted with their bounding box; the box mask is unioned in
   when it agrees with the original.

No parameters are trained.  The only learned components are the frozen SAM and
DINOv2 weights.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, fields, replace
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

LOW_RES = 256  # SAM's low-res mask frame


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
@dataclass
class SegmenterConfig:
    """All tunable settings and their defaults."""

    # models
    sam_checkpoint: str = field(default='checkpoints/sam_vit_h_4b8939.pth',
                                metadata={'help': 'SAM v1 checkpoint (.pth)'})
    sam_model_type: str = field(default='vit_h', metadata={'help': 'SAM v1 model type: vit_h, vit_l or vit_b'})
    dino_model: str = field(default='facebook/dinov2-base', metadata={'help': 'Hugging Face DINOv2 model id'})
    dino_long_side: int = field(default=896, metadata={'help': 'long side (px) of the DINOv2 input'})
    device: str = field(default='cuda', metadata={'help': 'torch device'})

    # few-shot prototype
    n_prototypes: int = field(default=3, metadata={'help': 'k-means prototypes over reference foreground features'})

    # dense proposals
    points_per_side: int = field(default=32, metadata={'help': 'point grid size (points_per_side^2 prompts)'})
    points_per_batch: int = field(default=128, metadata={'help': 'prompts decoded per batch'})
    pre_iou_thresh: float = field(default=0.5, metadata={'help': 'drop candidates with SAM predicted IoU below this'})
    pre_stability_thresh: float = field(default=0.7, metadata={'help': 'drop candidates with stability below this'})

    # validity gates
    min_similarity: float = field(default=0.6, metadata={'help': 'min mean prototype similarity inside a mask'})
    max_ink: float = field(default=0.1, metadata={'help': 'max mean darkness (1 - gray) inside a mask'})
    edge_dark: float = field(default=0.3, metadata={'help': 'darkness above which a pixel counts as line work'})
    min_edge_frac: float = field(default=0.4, metadata={'help': 'min share of the boundary band on line work'})
    edge_small_area: float = field(default=0.01, metadata={'help': 'edge gate applies only below this area fraction'})
    vote_iou: float = field(default=0.7, metadata={'help': 'IoU for two candidates to count as duplicates'})
    min_votes: int = field(default=3, metadata={'help': 'min number of near-duplicate candidates'})
    min_area_frac: float = field(default=0.0005, metadata={'help': 'min mask area as a fraction of the image'})
    max_area_frac: float = field(default=0.45, metadata={'help': 'max mask area as a fraction of the image'})
    min_pixels: int = field(default=100, metadata={'help': 'drop final masks smaller than this'})

    # ordering and selection
    p_area: float = field(default=0.25, metadata={'help': 'area exponent in the visiting order'})
    p_votes: float = field(default=0.25, metadata={'help': 'votes exponent in the visiting order'})
    nms_iou: float = field(default=0.5, metadata={'help': 'drop candidates overlapping a kept mask above this IoU'})
    contain_thresh: float = field(default=0.8, metadata={'help': 'share of a mask inside another to count as contained'})
    cover: float = field(default=0.6, metadata={'help': 'swallowed kept masks covering less than this are sub-parts'})
    absorb: int = field(default=3, metadata={'help': 'absorb candidates inside a kept mask dilated by N low-res px'})
    split_cover: float = field(default=0.6, metadata={'help': 'split a kept mask into >=2 parts covering this share'})
    split_max_frac: float = field(default=0.5, metadata={'help': 'a split part must be smaller than this share'})
    max_objects: int = field(default=40, metadata={'help': 'max instances per image'})

    # open-plan merge
    merge_gap: bool = field(default=True, metadata={'help': 'merge kept masks joined by a wall-less contact'})
    gap_r: int = field(default=3, metadata={'help': 'contact search radius in low-res px'})
    gap_max_dark: float = field(default=0.2, metadata={'help': 'max share of line work in the contact region'})
    gap_frac: float = field(default=0.6, metadata={'help': 'min contact length relative to the smaller boundary'})
    gap_min: int = field(default=4, metadata={'help': 'min contact size in low-res px'})

    # refinement
    refine_box: bool = field(default=True, metadata={'help': 're-prompt SAM with each final mask box'})
    refine_min_iou: float = field(default=0.6, metadata={'help': 'accept the box mask above this IoU'})
    refine_max_grow: float = field(default=1.5, metadata={'help': 'accept the box mask below this area growth'})

    def updated(self, **overrides) -> 'SegmenterConfig':
        unknown = set(overrides) - {f.name for f in fields(self)}
        if unknown:
            raise TypeError(f'unknown config fields: {sorted(unknown)}')
        return replace(self, **overrides)


# ----------------------------------------------------------------------------
# Encoders
# ----------------------------------------------------------------------------
class DinoEncoder:
    """Dense DINOv2 patch features for an image of arbitrary aspect ratio."""

    def __init__(self, model_name: str, device: str, long_side: int):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.device = torch.device(device)
        self.long_side = long_side
        self.patch = getattr(self.model.config, 'patch_size', 14)
        self.num_register = getattr(self.model.config, 'num_register_tokens', 0)
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        import inspect
        try:
            self._interp_ok = 'interpolate_pos_encoding' in inspect.signature(self.model.forward).parameters
        except (TypeError, ValueError):
            self._interp_ok = False

    @torch.no_grad()
    def encode(self, image: np.ndarray) -> torch.Tensor:
        """RGB uint8 image -> feature grid [C, h, w]."""
        H0, W0 = image.shape[:2]
        s = self.long_side / max(H0, W0)
        h1 = max(self.patch, (int(round(H0 * s)) // self.patch) * self.patch)
        w1 = max(self.patch, (int(round(W0 * s)) // self.patch) * self.patch)
        img = cv2.resize(image, (w1, h1), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img).to(self.device).float().permute(2, 0, 1)[None] / 255.0
        x = (x - self._mean) / self._std
        if self._interp_ok:
            seq = self.model(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state
        else:
            seq = self._manual_forward(x)
        tok = seq[:, 1 + self.num_register:, :]
        hp, wp = h1 // self.patch, w1 // self.patch
        return tok.reshape(1, hp, wp, -1).permute(0, 3, 1, 2)[0].contiguous()

    @torch.no_grad()
    def _manual_forward(self, x: torch.Tensor) -> torch.Tensor:
        # Older transformers releases do not expose `interpolate_pos_encoding`.
        emb = self.model.embeddings
        B, _, h_in, w_in = x.shape
        tokens = torch.cat([emb.cls_token.expand(B, -1, -1), emb.patch_embeddings(x)], dim=1)
        tokens = tokens + emb.interpolate_pos_encoding(tokens, h_in, w_in)
        if hasattr(emb, 'dropout'):
            tokens = emb.dropout(tokens)
        if self.num_register > 0 and hasattr(emb, 'register_tokens'):
            reg = emb.register_tokens.expand(B, -1, -1)
            tokens = torch.cat([tokens[:, :1], reg, tokens[:, 1:]], dim=1)
        seq = self.model.encoder(tokens)[0]
        if hasattr(self.model, 'layernorm'):
            seq = self.model.layernorm(seq)
        return seq


class SamBackend:
    """SAM v1 (`segment_anything`) exposing batched point decoding in the full-image frame.

    SAM resizes the longest image side to 1024 px and pads the rest, so its
    embedding and low-res logits contain padding.  Logits are cropped to the
    valid region and resized to a 256x256 frame that covers the whole image.
    """

    def __init__(self, checkpoint: str, model_type: str, device: str):
        from segment_anything import SamPredictor, sam_model_registry
        self.model = sam_model_registry[model_type](checkpoint=checkpoint).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.predictor = SamPredictor(self.model)
        self.device = torch.device(device)

    def set_image(self, image: np.ndarray) -> None:
        self.predictor.set_image(image)
        self.hw = image.shape[:2]
        h_in, w_in = self.predictor.input_size
        self.valid_low = (int(np.ceil(h_in / 4)), int(np.ceil(w_in / 4)))

    @torch.no_grad()
    def decode_points(self, pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """pts [B, 2] pixel (x, y) -> logits [B, 3, 256, 256] over the whole image, iou_pred [B, 3]."""
        coords = self.predictor.transform.apply_coords_torch(pts[:, None, :], self.hw)
        labels = torch.ones(coords.shape[0], 1, dtype=torch.int, device=self.device)
        sparse, dense = self.model.prompt_encoder(points=(coords, labels), boxes=None, masks=None)
        low_res, iou_pred = self.model.mask_decoder(
            image_embeddings=self.predictor.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=True)
        lh, lw = self.valid_low
        low_res = F.interpolate(low_res[:, :, :lh, :lw], size=(LOW_RES, LOW_RES), mode='bilinear', align_corners=False)
        return low_res.clamp(-32, 32), iou_pred

    @torch.no_grad()
    def predict_box(self, box: np.ndarray) -> np.ndarray:
        masks = self.predictor.predict(box=box[None], multimask_output=False)[0]
        return np.asarray(masks[0]).astype(bool)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def build_point_grid(n_per_side: int) -> np.ndarray:
    """Uniform grid in [0, 1]^2 with (x, y) rows, same layout as SAM's mask generator."""
    offset = 1 / (2 * n_per_side)
    pts = np.linspace(offset, 1 - offset, n_per_side)
    gx, gy = np.meshgrid(pts, pts)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def robust_normalise(s: torch.Tensor, lo_q: float = 0.05, hi_q: float = 0.95) -> torch.Tensor:
    flat = s.flatten()
    lo, hi = torch.quantile(flat, lo_q), torch.quantile(flat, hi_q)
    return ((s - lo) / (hi - lo + 1e-6)).clamp(0, 1)


def stability_score(logits: torch.Tensor, offset: float = 1.0) -> torch.Tensor:
    hi = (logits > offset).flatten(1).sum(1).float()
    lo = (logits > -offset).flatten(1).sum(1).float()
    return hi / lo.clamp_min(1)


def fill_and_clean(mask: np.ndarray) -> np.ndarray:
    """Fill holes and keep the largest connected component."""
    m = ndimage.binary_fill_holes(mask)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, index=np.arange(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    return m


# ----------------------------------------------------------------------------
# Segmenter
# ----------------------------------------------------------------------------
class FewShotRoomSegmenter:
    """Few-shot multi-instance segmenter.

    Example::

        seg = FewShotRoomSegmenter(SegmenterConfig(sam_checkpoint='checkpoints/sam_vit_h_4b8939.pth'))
        seg.set_references([(ref_image, ref_mask), ...])   # RGB uint8 images, binary masks
        masks = seg.segment(image)                          # (K, H, W) bool
    """

    def __init__(self, config: Optional[SegmenterConfig] = None, **overrides):
        self.config = (config or SegmenterConfig()).updated(**overrides)
        c = self.config
        self.device = torch.device(c.device)
        self.sam = SamBackend(c.sam_checkpoint, c.sam_model_type, c.device)
        self.dino = DinoEncoder(c.dino_model, c.device, c.dino_long_side)
        self.grid = build_point_grid(c.points_per_side)
        self.prototypes: Optional[torch.Tensor] = None

    MODEL_FIELDS = ('sam_checkpoint', 'sam_model_type', 'dino_model', 'dino_long_side', 'device')

    def update_config(self, **overrides) -> SegmenterConfig:
        """Change settings without reloading the models.

        Model settings (checkpoints, device, DINOv2 input size) need a new
        segmenter.  Changing ``n_prototypes`` clears the prototype bank, so call
        :meth:`set_references` again afterwards.
        """
        locked = sorted(set(overrides) & set(self.MODEL_FIELDS))
        if locked:
            raise ValueError(f'create a new FewShotRoomSegmenter to change {locked}')
        self.config = self.config.updated(**overrides)
        self.grid = build_point_grid(self.config.points_per_side)
        if 'n_prototypes' in overrides:
            self.prototypes = None
        return self.config

    # ---- references ---------------------------------------------------------
    @torch.no_grad()
    def set_references(self, references: Iterable[Tuple[np.ndarray, np.ndarray]]) -> torch.Tensor:
        """Build the prototype bank from (RGB image, mask) pairs.  Returns it as [n_prototypes, C]."""
        from sklearn.cluster import KMeans
        fg = []
        for image, mask in references:
            f = F.normalize(self.dino.encode(image), dim=0)
            C, h, w = f.shape
            binary = (mask > 0).astype(np.uint8)
            m = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            if m.sum() == 0:  # very small reference: fall back to area resampling
                m = cv2.resize(binary, (w, h), interpolation=cv2.INTER_AREA) > 0.2
            if m.sum() == 0:
                continue
            fg.append(f.permute(1, 2, 0)[torch.from_numpy(m).to(f.device)])
        if not fg:
            raise ValueError('no usable reference masks')
        X = torch.cat(fg, 0)
        n = min(self.config.n_prototypes, len(X))
        if n <= 1:
            P = X.mean(0, keepdim=True)
        else:
            km = KMeans(n_clusters=n, n_init=5, random_state=0).fit(X.float().cpu().numpy())
            P = torch.from_numpy(km.cluster_centers_).to(X.device, X.dtype)
        self.prototypes = F.normalize(P, dim=1)
        return self.prototypes

    # ---- per-image steps ----------------------------------------------------
    @torch.no_grad()
    def similarity_map(self, image: np.ndarray) -> torch.Tensor:
        """Prototype similarity in the 256x256 frame, normalised to [0, 1]."""
        if self.prototypes is None:
            raise RuntimeError('call set_references() first')
        f = F.normalize(self.dino.encode(image), dim=0)
        C, h, w = f.shape
        s = (self.prototypes @ f.reshape(C, -1)).max(0).values.reshape(1, 1, h, w)
        s = F.interpolate(s, size=(LOW_RES, LOW_RES), mode='bilinear', align_corners=False)[0, 0]
        return robust_normalise(s)

    @torch.no_grad()
    def _propose(self, H: int, W: int) -> dict:
        c = self.config
        pts = torch.as_tensor(self.grid * np.array([W, H])[None], dtype=torch.float32, device=self.device)
        logits_l, iou_l, stab_l = [], [], []
        for i in range(0, len(pts), c.points_per_batch):
            low, iou = self.sam.decode_points(pts[i:i + c.points_per_batch])
            low = low.flatten(0, 1)
            iou = iou.flatten()
            stab = stability_score(low, 1.0)
            keep = (iou > c.pre_iou_thresh) & (stab > c.pre_stability_thresh)
            logits_l.append(low[keep].half())
            iou_l.append(iou[keep])
            stab_l.append(stab[keep])
        return dict(logits=torch.cat(logits_l), iou=torch.cat(iou_l), stab=torch.cat(stab_l))

    @torch.no_grad()
    def _candidate_stats(self, masks: torch.Tensor, sim: torch.Tensor, gray: torch.Tensor) -> dict:
        c = self.config
        mf = masks.float()
        area = mf.flatten(1).sum(1)
        sim_in = (mf * sim[None]).flatten(1).sum(1) / area.clamp_min(1)
        ink = 1.0 - (mf * gray[None]).flatten(1).sum(1) / area.clamp_min(1)
        # boundary band = mask dilated by 2 px minus mask eroded by 1 px
        dil2 = F.max_pool2d(mf[:, None], 5, 1, 2)[:, 0] > 0
        ero1 = (-F.max_pool2d(-mf[:, None], 3, 1, 1)[:, 0]) > 0.5
        band = dil2 & ~ero1
        dark = (1.0 - gray) > c.edge_dark
        edge_frac = (band & dark[None]).flatten(1).sum(1).float() / band.flatten(1).sum(1).clamp_min(1).float()
        votes = self._count_votes(masks, area, c.vote_iou)
        return dict(area_frac=area / (LOW_RES * LOW_RES), similarity=sim_in, ink=ink, edge_frac=edge_frac, votes=votes)

    @torch.no_grad()
    def _count_votes(self, masks: torch.Tensor, area: torch.Tensor, iou_thr: float, block: int = 1024) -> torch.Tensor:
        X = masks.flatten(1).to(torch.float16 if masks.is_cuda else torch.float32)
        votes = torch.zeros(X.shape[0], device=masks.device)
        for i in range(0, X.shape[0], block):
            inter = (X[i:i + block] @ X.T).float()
            union = area[i:i + block, None] + area[None] - inter
            votes[i:i + block] = ((inter / union.clamp_min(1)) >= iou_thr).sum(1).float()
        return votes

    @torch.no_grad()
    def _select(self, masks: torch.Tensor, order_scores: torch.Tensor, valid: torch.Tensor):
        """Containment-aware greedy NMS.  Returns [(kept index, [absorbed indices]), ...]."""
        c = self.config
        order = torch.argsort(order_scores, descending=True)
        areas = masks.flatten(1).sum(1).float()
        kept, absorbed = [], {}
        for idx in order.tolist():
            if not valid[idx]:
                continue
            if len(kept) >= c.max_objects:
                break
            if kept:
                K = masks[kept]
                inter = (K & masks[idx][None]).flatten(1).sum(1).float()
                iou = inter / (areas[kept] + areas[idx] - inter).clamp_min(1)
                in_kept = inter / areas[idx].clamp_min(1)   # candidate inside a kept mask
                swallow = inter / areas[kept].clamp_min(1)  # kept mask inside the candidate
                if (iou > c.nms_iou).any() or (in_kept > c.contain_thresh).any():
                    continue
                if c.absorb > 0:
                    # thin parts along a kept mask (counters, cabinets) -> union into it
                    Kd = F.max_pool2d(K.float()[:, None], 2 * c.absorb + 1, 1, c.absorb)[:, 0] > 0
                    in_dil = (Kd & masks[idx][None]).flatten(1).sum(1).float() / areas[idx].clamp_min(1)
                    if (in_dil > c.contain_thresh).any():
                        absorbed.setdefault(kept[int(torch.argmax(in_dil))], []).append(idx)
                        continue
                sw = swallow > c.contain_thresh
                if sw.any():
                    sw_idx = [kept[j] for j in torch.nonzero(sw).flatten().tolist()]
                    cov = (masks[sw_idx].any(0) & masks[idx]).sum().float() / areas[idx].clamp_min(1)
                    if cov >= c.cover:      # swallows complete instances: it is a merge
                        continue
                    kept = [k for k in kept if k not in sw_idx]   # swallows sub-parts: replace them
                    for k in sw_idx:
                        absorbed.pop(k, None)
            kept.append(idx)

        # split pass: a kept mask containing >= 2 valid, disjoint parts that
        # cover most of it is a merge of several instances
        if c.split_cover > 0:
            new_kept = []
            for k in kept:
                inside = (masks & masks[k][None]).flatten(1).sum(1).float() / areas.clamp_min(1)
                cand = torch.nonzero(valid & (inside > c.contain_thresh) & (areas < c.split_max_frac * areas[k])).flatten().tolist()
                cand = sorted((j for j in cand if j != k), key=lambda j: -float(order_scores[j]))
                parts = []
                for j in cand:
                    if parts:
                        P = masks[parts]
                        inter = (P & masks[j][None]).flatten(1).sum(1).float()
                        if (inter / (areas[parts] + areas[j] - inter).clamp_min(1) > c.nms_iou).any() \
                                or (inter / areas[j] > c.contain_thresh).any() or (inter / areas[parts] > c.contain_thresh).any():
                            continue
                    parts.append(j)
                if len(parts) >= 2 and (masks[parts].any(0) & masks[k]).sum().float() / areas[k] >= c.split_cover:
                    new_kept += parts
                    absorbed.pop(k, None)
                else:
                    new_kept.append(k)
            kept = new_kept
        return [(k, absorbed.get(k, [])) for k in kept]

    @torch.no_grad()
    def _merge_open_gaps(self, groups, masks: torch.Tensor, gray: torch.Tensor):
        """Merge groups whose masks touch through a long contact without line work."""
        c = self.config
        dark = (1.0 - gray) > c.edge_dark
        lows = [masks[g].any(0) for g in groups]
        changed = True
        while changed and len(lows) > 1:
            changed = False
            n = len(lows)
            L = torch.stack(lows)
            dil = F.max_pool2d(L.float()[:, None], 2 * c.gap_r + 1, 1, c.gap_r)[:, 0] > 0
            ero = (-F.max_pool2d(-L.float()[:, None], 3, 1, 1))[:, 0] > 0.5
            bnd_len = (L & ~ero).flatten(1).sum(1)
            best = None
            for i in range(n):
                for j in range(i + 1, n):
                    contact = (dil[i] & dil[j] & ~L[i] & ~L[j]) | (L[i] & dil[j]) | (L[j] & dil[i])
                    nc = int(contact.sum())
                    if nc < c.gap_min:
                        continue
                    small = i if bnd_len[i] < bnd_len[j] else j
                    if nc / max(int(bnd_len[small]), 1) < c.gap_frac:
                        continue
                    dark_share = float((contact & dark).sum()) / nc
                    if dark_share < c.gap_max_dark and (best is None or dark_share < best[0]):
                        best = (dark_share, i, j)
            if best is not None:
                _, i, j = best
                groups = [g for k, g in enumerate(groups) if k not in (i, j)] + [groups[i] + groups[j]]
                lows = [m for k, m in enumerate(lows) if k not in (i, j)] + [lows[i] | lows[j]]
                changed = True
        return groups

    @torch.no_grad()
    def _refine_with_boxes(self, masks):
        c = self.config
        out = []
        for m in masks:
            ys, xs = np.nonzero(m)
            box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
            r = fill_and_clean(self.sam.predict_box(box))
            inter = np.logical_and(r, m).sum()
            union = np.logical_or(r, m).sum()
            if union > 0 and inter / union >= c.refine_min_iou and r.sum() <= c.refine_max_grow * m.sum():
                out.append(r | m)
            else:
                out.append(m)
        return out

    # ---- public entry point -------------------------------------------------
    @torch.no_grad()
    def segment(self, image: np.ndarray, return_details: bool = False):
        """Segment all instances in an RGB uint8 image.

        Returns a (K, H, W) bool array.  With ``return_details=True`` it returns
        ``(masks, details)`` where ``details`` holds the similarity map, the
        candidate count after every gate, the low-res masks kept by the NMS and
        timings.
        """
        c = self.config
        H, W = image.shape[:2]
        t0 = time.time()
        self.sam.set_image(image)
        sim = self.similarity_map(image)
        gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), (LOW_RES, LOW_RES), interpolation=cv2.INTER_AREA)
        gray = torch.from_numpy(gray.astype(np.float32) / 255.0).to(self.device)

        prop = self._propose(H, W)
        n_cand = int(prop['logits'].shape[0])
        if n_cand == 0:
            empty = np.zeros((0, H, W), dtype=bool)
            return (empty, {'n_candidates': 0}) if return_details else empty
        masks = prop['logits'] > 0
        stats = self._candidate_stats(masks, sim, gray)
        quality = prop['iou'].float() * prop['stab'] * stats['similarity'].clamp(0, 1)
        order_scores = quality * stats['area_frac'].clamp_min(1e-4) ** c.p_area
        order_scores = order_scores * stats['votes'].clamp_min(1) ** c.p_votes
        gates = [
            ('similarity', stats['similarity'] >= c.min_similarity),
            ('ink', stats['ink'] <= c.max_ink),
            ('edge', (stats['edge_frac'] >= c.min_edge_frac) | (stats['area_frac'] >= c.edge_small_area)),
            ('votes', stats['votes'] >= c.min_votes),
            ('area', (stats['area_frac'] >= c.min_area_frac) & (stats['area_frac'] <= c.max_area_frac)),
        ]
        valid = torch.ones_like(gates[0][1])
        funnel = {'candidates': n_cand}
        for name, g in gates:
            valid = valid & g
            funnel[name] = int(valid.sum())
        kept = self._select(masks, order_scores, valid)
        t_select = time.time()

        groups = [[idx] + parts for idx, parts in kept]
        nms_lowres = [masks[g].any(0) for g in groups]
        if c.merge_gap and len(groups) > 1:
            groups = self._merge_open_gaps(groups, masks, gray)

        def upsample(idx):
            lg = prop['logits'][idx][None, None].float()
            return (F.interpolate(lg, size=(H, W), mode='bilinear', align_corners=False)[0, 0] > 0).cpu().numpy()

        final = []
        for g in groups:
            m = upsample(g[0])
            for j in g[1:]:
                m |= upsample(j)
            m = fill_and_clean(m)
            if m.sum() >= c.min_pixels:
                final.append(m)
        if c.refine_box and final:
            final = self._refine_with_boxes(final)
        out = np.stack(final) if final else np.zeros((0, H, W), dtype=bool)
        if self.device.type == 'cuda':
            del prop, masks
            torch.cuda.empty_cache()
        if not return_details:
            return out
        funnel.update(after_nms=len(nms_lowres), after_merge=len(groups), final=len(out))
        details = dict(
            similarity=sim.cpu().numpy(),
            funnel=funnel,
            nms_masks_lowres=torch.stack(nms_lowres).cpu().numpy() if nms_lowres else np.zeros((0, LOW_RES, LOW_RES), bool),
            seconds_to_selection=t_select - t0,
            seconds_total=time.time() - t0,
        )
        return out, details
