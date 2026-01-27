#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic ROI dataset generator for seal-ID metric learning.

Guarantees:
- For each seal/background pair, generate n_pairs_per_bg clean and defect ROIs from the same compose (same bg crop, placement, pad).
- Backgrounds are iterated once per seal (no sampling).
- Defect coverage is clamped to [min_cover, max_cover] and hard-limited to [0.10, 0.40].
- Outputs: out_dir/train/<seal_id>/*.jpg and out_dir/val/<seal_id>/*.jpg
    Naming: <seal_id>__<bg_type>__<bg_idx:04d>__s<sample:06d>__pad<pad:02d>__v(clean|def).jpg
- Manifests: out_dir/manifest_train.csv and out_dir/manifest_val.csv with rel_path, split, seal_id, variant, bg_type, bg_idx, pad, sample_idx, pair_id.
- Optional debug: out_dir/_debug/<split>/<seal_id>/*__stampmask.png, *__defmask.png, *__stamp.png
"""

import os
import glob
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm
import pandas as pd

IMG_EXTS = (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")


# ----------------------------- utils -----------------------------
def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _pil_to_np_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"), dtype=np.uint8)


def _np_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _bbox_from_alpha(alpha: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(alpha > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return x0, y0, x1, y1


def _rand_factor(j: float) -> float:
    return 1.0 + random.uniform(-j, j)


def _jpeg_roundtrip_rgb(arr_rgb: np.ndarray, quality: int) -> np.ndarray:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR), encode_param)
    if not ok:
        return arr_rgb
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if dec is None:
        return arr_rgb
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def list_images(dir_path: str) -> List[str]:
    paths: List[str] = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(dir_path, f"*{ext}")))
    return sorted(paths)


# ----------------------------- blend -----------------------------
def multiply_blend(
    base_rgb: np.ndarray,
    stamp_rgba: np.ndarray,
    x: int,
    y: int,
    blend_alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multiply blend stamp onto base. Returns composited RGB and stamp mask (0/255) in base coords."""
    H, W = base_rgb.shape[:2]
    sh, sw = stamp_rgba.shape[:2]

    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + sw), min(H, y + sh)
    if x0 >= x1 or y0 >= y1:
        return base_rgb, np.zeros((H, W), dtype=np.uint8)

    bg_roi = base_rgb[y0:y1, x0:x1].astype(np.float32) / 255.0
    st_roi = stamp_rgba[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32) / 255.0
    st_rgb = st_roi[..., :3]
    st_a = st_roi[..., 3:4] * float(blend_alpha)

    multiplied = bg_roi * st_rgb
    out_roi = (1.0 - st_a) * bg_roi + st_a * multiplied

    out = base_rgb.copy()
    out[y0:y1, x0:x1] = np.clip(out_roi * 255.0, 0, 255).astype(np.uint8)

    m = (st_a[..., 0] > 0.001).astype(np.uint8) * 255
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y0:y1, x0:x1] = m
    return out, mask


def alpha_composite(
    base_rgb: np.ndarray,
    stamp_rgba: np.ndarray,
    x: int,
    y: int,
    blend_alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Standard alpha composite. Returns composited RGB and stamp mask (0/255) in base coords."""
    H, W = base_rgb.shape[:2]
    sh, sw = stamp_rgba.shape[:2]

    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + sw), min(H, y + sh)
    if x0 >= x1 or y0 >= y1:
        return base_rgb, np.zeros((H, W), dtype=np.uint8)

    bg_roi = base_rgb[y0:y1, x0:x1].astype(np.float32) / 255.0
    st_roi = stamp_rgba[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32) / 255.0
    st_rgb = st_roi[..., :3]
    st_a = st_roi[..., 3:4] * float(blend_alpha)

    out_roi = (1.0 - st_a) * bg_roi + st_a * st_rgb

    out = base_rgb.copy()
    out[y0:y1, x0:x1] = np.clip(out_roi * 255.0, 0, 255).astype(np.uint8)

    m = (st_a[..., 0] > 0.001).astype(np.uint8) * 255
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[y0:y1, x0:x1] = m
    return out, mask


# ----------------------------- under-text & bg perturb -----------------------------
def _make_under_text_layer(H: int, W: int) -> np.ndarray:
    layer = np.zeros((H, W), dtype=np.uint8)
    n_lines = random.randint(10, 28)
    for _ in range(n_lines):
        x0 = random.randint(0, W - 1)
        y0 = random.randint(0, H - 1)
        x1 = min(W - 1, x0 + random.randint(30, 200))
        y1 = min(H - 1, y0 + random.randint(-10, 10))
        thickness = random.choice([1, 1, 2])
        cv2.line(layer, (x0, y0), (x1, y1), 255, thickness, lineType=cv2.LINE_AA)
    layer = cv2.GaussianBlur(layer, (3, 3), 0)
    return layer


def simulate_under_text_bleed(full_rgb: np.ndarray, stamp_mask: np.ndarray, alpha: float, layer: np.ndarray) -> np.ndarray:
    out = full_rgb.copy()
    inside = (stamp_mask > 0)
    dark = (layer.astype(np.float32) / 255.0) * float(alpha)
    f = out.astype(np.float32) / 255.0
    for c in range(3):
        ch = f[..., c]
        ch[inside] = np.clip(ch[inside] * (1.0 - dark[inside]), 0.0, 1.0)
        f[..., c] = ch
    return np.clip(f * 255.0, 0, 255).astype(np.uint8)


def perturb_background_only(
    full_rgb: np.ndarray,
    stamp_mask: np.ndarray,
    b_jitter: float,
    c_jitter: float,
    noise_std_range: Tuple[float, float],
    blur_prob: float,
    blur_k_choices: List[int],
    jpeg_q_range: Tuple[int, int],
) -> np.ndarray:
    out = full_rgb.copy()
    H, W = out.shape[:2]
    bg = (stamp_mask == 0)

    b_factor = _rand_factor(b_jitter)
    c_factor = _rand_factor(c_jitter)

    f = out.astype(np.float32)
    f_bg = f[bg]
    f_bg = (f_bg - 128.0) * c_factor + 128.0
    f_bg = f_bg * b_factor
    f[bg] = f_bg
    out = np.clip(f, 0, 255).astype(np.uint8)

    std = random.uniform(*noise_std_range)
    if std > 0.01:
        noise = np.random.normal(0.0, std, size=(H, W, 3)).astype(np.float32)
        f = out.astype(np.float32)
        f[bg] += noise[bg]
        out = np.clip(f, 0, 255).astype(np.uint8)

    if random.random() < blur_prob:
        k = random.choice(blur_k_choices)
        blurred = cv2.GaussianBlur(out, (k, k), 0)
        out[bg] = blurred[bg]

    q = random.randint(*jpeg_q_range)
    out = _jpeg_roundtrip_rgb(out, q)
    return out


# ----------------------------- config -----------------------------
@dataclass
class GenCfg:
    # stamp transforms
    scale_range: Tuple[float, float] = (0.30, 0.50)
    rotation_range: Tuple[float, float] = (0.0, 360.0)
    color_enh_range: Tuple[float, float] = (0.55, 0.95)
    contrast_range: Tuple[float, float] = (0.80, 1.05)
    alpha_range: Tuple[float, float] = (0.55, 0.85)

    # Simulate scan artifacts (red stamp -> gray/black) with 20% probability
    stamp_gray_prob: float = 0.20
    stamp_gray_mode_probs: Tuple[float, float] = (0.60, 0.40)  # (dark, light)
    stamp_gray_dark_factor_range: Tuple[float, float] = (0.18, 0.50)
    stamp_gray_light_factor_range: Tuple[float, float] = (0.50, 0.95)
    stamp_gray_bias_range: Tuple[int, int] = (0, 35)

    # ink texture
    ink_tex_prob: float = 0.85
    ink_tex_grid: Tuple[int, int] = (28, 28)
    ink_tex_strength_range: Tuple[float, float] = (0.15, 0.45)

    # feather edge
    feather_prob: float = 0.70
    feather_k_choices: Tuple[int, ...] = (3, 5)

    # compose
    use_multiply: bool = True
    blend_alpha_range: Tuple[float, float] = (0.85, 1.00)

    # under-text
    under_text_prob: float = 0.50
    under_text_alpha_range: Tuple[float, float] = (0.05, 0.12)

    # bg perturb
    bg_perturb_prob: float = 0.95
    bg_brightness_jitter: float = 0.06
    bg_contrast_jitter: float = 0.06
    bg_noise_std_range: Tuple[float, float] = (0.0, 2.0)
    bg_blur_prob: float = 0.25
    bg_blur_k_choices: Tuple[int, ...] = (3,)
    bg_jpeg_quality_range: Tuple[int, int] = (80, 95)

    # pad policy
    pad_policy: Tuple[Tuple[float, Tuple[int, int]], ...] = (
        (0.85, (3, 10)),
        (0.12, (10, 25)),
        (0.03, (25, 60)),
    )

    # defect generation
    defect_max_tries: int = 160
    defect_target_bias: float = 0.80
    blob_grid_range: Tuple[int, int] = (10, 22)
    blob_sigma_range: Tuple[float, float] = (1.2, 3.0)
    blob_thresh_range: Tuple[float, float] = (0.42, 0.58)
    blob_morph_k_choices: Tuple[int, ...] = (7, 9, 11)
    blob_morph_iters_range: Tuple[int, int] = (1, 3)
    ring_bias_prob: float = 0.35
    ring_bias_strength: float = 0.40
    defect_soft_edge_px_range: Tuple[int, int] = (3, 9)


def sample_pad(cfg: GenCfg) -> int:
    r = random.random()
    cum = 0.0
    for prob, (a, b) in cfg.pad_policy:
        cum += float(prob)
        if r <= cum:
            return random.randint(int(a), int(b))
    a, b = cfg.pad_policy[-1][1]
    return random.randint(int(a), int(b))


def make_lowfreq_noise_base(grid_hw: Tuple[int, int]) -> np.ndarray:
    gh, gw = grid_hw
    return np.random.rand(gh, gw).astype(np.float32)


def ink_noise_for_size(noise_base: np.ndarray, h: int, w: int) -> np.ndarray:
    up = cv2.resize(noise_base, (w, h), interpolation=cv2.INTER_CUBIC)
    up = cv2.GaussianBlur(up, (0, 0), sigmaX=1.0)
    mn, mx = float(up.min()), float(up.max())
    if mx - mn < 1e-6:
        return np.zeros((h, w), dtype=np.float32)
    return (up - mn) / (mx - mn)


# ----------------------------- stamp color shift -----------------------------
def maybe_turn_stamp_gray(stamp_rgba: np.ndarray, cfg: GenCfg) -> np.ndarray:
    """
    With prob cfg.stamp_gray_prob, convert stamp RGB into gray/black-ish to simulate scan artifacts.
    Alpha channel is preserved. Output is ALWAYS RGBA (H,W,4).
    """
    if stamp_rgba is None or stamp_rgba.size == 0:
        return stamp_rgba

    # Safety: if somehow not RGBA, don't touch it
    if stamp_rgba.ndim != 3 or stamp_rgba.shape[2] != 4:
        return stamp_rgba

    if random.random() >= float(cfg.stamp_gray_prob):
        return stamp_rgba

    rgb = stamp_rgba[..., :3].astype(np.float32)          # (H,W,3)
    a = stamp_rgba[..., 3:4].astype(np.float32)           # (H,W,1)

    # luminance gray in [0,255], shape (H,W,1)
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])[..., None]

    # choose dark/light mode
    p_dark, p_light = cfg.stamp_gray_mode_probs
    denom = max(1e-6, float(p_dark + p_light))
    if random.random() < float(p_dark) / denom:
        factor = random.uniform(*cfg.stamp_gray_dark_factor_range)
    else:
        factor = random.uniform(*cfg.stamp_gray_light_factor_range)

    bias = random.randint(*cfg.stamp_gray_bias_range)

    # out_gray: (H,W,1)
    out_gray = np.clip(gray * float(factor) + float(bias), 0, 255)

    # replicate to RGB: (H,W,3)
    out_rgb = np.repeat(out_gray, 3, axis=2)

    out = np.concatenate([out_rgb, a], axis=2).astype(np.uint8)  # (H,W,4)
    return out



# ----------------------------- stamp process -----------------------------
def process_stamp_rgba(seal_rgba: Image.Image, cfg: GenCfg) -> np.ndarray:
    sf = random.uniform(*cfg.scale_range)
    new_size = (
        max(8, int(seal_rgba.width * sf)),
        max(8, int(seal_rgba.height * sf)),
    )
    seal_rgba = seal_rgba.resize(new_size, Image.Resampling.LANCZOS)

    ang = random.uniform(*cfg.rotation_range)
    seal_rgba = seal_rgba.rotate(ang, expand=True, resample=Image.BICUBIC)

    stamp = np.array(seal_rgba, dtype=np.uint8)

    bbox = _bbox_from_alpha(stamp[..., 3])
    if bbox is None:
        return np.zeros((8, 8, 4), dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    stamp = stamp[y0:y1, x0:x1]

    pil = Image.fromarray(stamp, mode="RGBA")
    pil = ImageEnhance.Color(pil).enhance(random.uniform(*cfg.color_enh_range))
    pil = ImageEnhance.Contrast(pil).enhance(random.uniform(*cfg.contrast_range))
    stamp = np.array(pil, dtype=np.uint8)

    # 20% chance to turn stamp gray/black-ish (simulate scan)
    stamp = maybe_turn_stamp_gray(stamp, cfg)
    if stamp.ndim != 3 or stamp.shape[2] != 4:
        raise RuntimeError(f"stamp must be RGBA (H,W,4), got {stamp.shape}")

    oa = random.uniform(*cfg.alpha_range)
    stamp[..., 3] = np.clip(stamp[..., 3].astype(np.float32) * oa, 0, 255).astype(np.uint8)

    if random.random() < cfg.ink_tex_prob:
        h, w = stamp.shape[:2]
        base = make_lowfreq_noise_base(cfg.ink_tex_grid)
        noise01 = ink_noise_for_size(base, h, w)
        s = random.uniform(*cfg.ink_tex_strength_range)
        mod = (1.0 - s) + (2.0 * s) * noise01
        a = stamp[..., 3].astype(np.float32) * mod
        stamp[..., 3] = np.clip(a, 0, 255).astype(np.uint8)

    if random.random() < cfg.feather_prob:
        k = random.choice(list(cfg.feather_k_choices))
        alpha = stamp[..., 3].astype(np.float32)
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
        stamp[..., 3] = np.clip(alpha, 0, 255).astype(np.uint8)

    return stamp


# ----------------------------- defect mask -----------------------------
def _keep_largest_cc(mask_u8: np.ndarray) -> np.ndarray:
    bin01 = (mask_u8 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin01, connectivity=8)
    if num <= 1:
        return mask_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    return (labels == best).astype(np.uint8) * 255


def _ring_weight_map(stamp_mask_u8: np.ndarray) -> np.ndarray:
    bin01 = (stamp_mask_u8 > 0).astype(np.uint8)
    dist_in = cv2.distanceTransform(bin01, cv2.DIST_L2, 3).astype(np.float32)
    if dist_in.max() < 1e-6:
        return np.ones_like(dist_in, dtype=np.float32)
    d = dist_in / (dist_in.max() + 1e-6)
    return np.clip(1.0 - d, 0.0, 1.0)


def sample_defect_mask(
    stamp_mask_u8: np.ndarray,
    min_cover_ratio: float,
    max_cover_ratio: float,
    cfg: GenCfg,
) -> np.ndarray:
    H, W = stamp_mask_u8.shape[:2]
    stamp_area = int((stamp_mask_u8 > 0).sum())
    if stamp_area < 120:
        return np.zeros((H, W), dtype=np.uint8)

    # hard clamp to your spec
    min_cover_ratio = float(min_cover_ratio)
    max_cover_ratio = float(max_cover_ratio)
    min_cover_ratio = max(0.10, min(min_cover_ratio, 0.40))
    max_cover_ratio = max(min_cover_ratio, min(max_cover_ratio, 0.40))

    target = random.uniform(cfg.defect_target_bias * max_cover_ratio, max_cover_ratio)

    stamp_bin = (stamp_mask_u8 > 0).astype(np.uint8)
    ring_w = _ring_weight_map(stamp_mask_u8)

    best_blob = np.zeros((H, W), dtype=np.uint8)
    best_cover = 0.0

    for _ in range(cfg.defect_max_tries):
        gh = random.randint(*cfg.blob_grid_range)
        gw = random.randint(*cfg.blob_grid_range)
        base = np.random.rand(gh, gw).astype(np.float32)
        field = cv2.resize(base, (W, H), interpolation=cv2.INTER_CUBIC)

        sigma = random.uniform(*cfg.blob_sigma_range)
        if sigma > 0.01:
            field = cv2.GaussianBlur(field, (0, 0), sigmaX=sigma)

        mn, mx = float(field.min()), float(field.max())
        if mx - mn > 1e-6:
            field = (field - mn) / (mx - mn)
        else:
            field = np.zeros_like(field, dtype=np.float32)

        if random.random() < cfg.ring_bias_prob:
            field = np.clip(field + cfg.ring_bias_strength * ring_w, 0.0, 1.0)

        thr = random.uniform(*cfg.blob_thresh_range)
        blob = (field >= thr).astype(np.uint8) * 255
        blob = ((blob > 0).astype(np.uint8) & stamp_bin).astype(np.uint8) * 255

        k = random.choice(list(cfg.blob_morph_k_choices))
        iters = random.randint(*cfg.blob_morph_iters_range)
        kernel = np.ones((k, k), np.uint8)

        if random.random() < 0.7:
            blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, kernel, iterations=iters)
        blob = cv2.dilate(blob, kernel, iterations=max(1, iters - 1))

        blob = (blob > 0).astype(np.uint8) * 255
        blob = _keep_largest_cc(blob)

        area = int((blob > 0).sum())
        cover = area / float(stamp_area)

        if cover > best_cover and cover <= max_cover_ratio:
            best_cover = cover
            best_blob = blob

        if cover < min_cover_ratio:
            continue

        if cover > max_cover_ratio:
            shrink_k = random.choice([3, 5, 7])
            shrink_it = 1 + (1 if cover > max_cover_ratio * 1.2 else 0)
            blob2 = cv2.erode(blob, np.ones((shrink_k, shrink_k), np.uint8), iterations=shrink_it)
            blob2 = _keep_largest_cc((blob2 > 0).astype(np.uint8) * 255)
            area2 = int((blob2 > 0).sum())
            cover2 = area2 / float(stamp_area)
            if min_cover_ratio <= cover2 <= max_cover_ratio:
                blob = blob2
                cover = cover2
            else:
                continue

        if cover >= 0.85 * target:
            return blob

    return best_blob


def apply_defect_erase_to_background(
    roi_rgb: np.ndarray,
    roi_bg_rgb: np.ndarray,
    stamp_mask_roi_u8: np.ndarray,
    min_cover: float,
    max_cover: float,
    cfg: GenCfg,
) -> Tuple[np.ndarray, np.ndarray]:
    def_u8 = sample_defect_mask(
        stamp_mask_u8=stamp_mask_roi_u8,
        min_cover_ratio=min_cover,
        max_cover_ratio=max_cover,
        cfg=cfg,
    )
    m = (def_u8.astype(np.float32) / 255.0)

    soft_px = random.randint(*cfg.defect_soft_edge_px_range)
    if soft_px > 0:
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=float(soft_px))

    stamp01 = (stamp_mask_roi_u8 > 0).astype(np.float32)
    m = m * stamp01

    out = roi_rgb.astype(np.float32)
    bg = roi_bg_rgb.astype(np.float32)
    alpha = np.clip(m, 0.0, 1.0)[..., None]
    out = (1.0 - alpha) * out + alpha * bg
    out = np.clip(out, 0, 255).astype(np.uint8)

    def_soft_u8 = np.clip(m * 255.0, 0, 255).astype(np.uint8)
    return out, def_soft_u8


# ----------------------------- compose & crop -----------------------------
def compose_one(
    seal_png_path: str,
    bg_rgb: np.ndarray,
    cfg: GenCfg,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]]:
    """
    Returns:
      roi_rgb:    composed+perturbed ROI (before defect)
      roi_bg_rgb: ROI with stamp pixels replaced by the original background crop
      stamp_mask_roi_u8
      pad
      debug_stamp_rgba (stamp after transforms, in stamp coords)
    """
    H, W = bg_rgb.shape[:2]

    seal_pil = Image.open(seal_png_path).convert("RGBA")
    stamp_rgba = process_stamp_rgba(seal_pil, cfg)
    sh, sw = stamp_rgba.shape[:2]
    if sw <= 0 or sh <= 0:
        return None

    if sw >= W or sh >= H:
        scale = min((0.6 * W) / max(sw, 1), (0.6 * H) / max(sh, 1))
        sw2 = max(8, int(sw * scale))
        sh2 = max(8, int(sh * scale))
        stamp_rgba = cv2.resize(stamp_rgba, (sw2, sh2), interpolation=cv2.INTER_AREA)
        sh, sw = stamp_rgba.shape[:2]

    if sw >= W or sh >= H:
        return None

    cx = random.uniform(0.20, 0.80) * W
    cy = random.uniform(0.20, 0.80) * H
    x = _clamp_int(int(cx) - sw // 2, 0, W - sw)
    y = _clamp_int(int(cy) - sh // 2, 0, H - sh)

    blend_alpha = random.uniform(*cfg.blend_alpha_range)

    if cfg.use_multiply:
        full_rgb, stamp_mask = multiply_blend(bg_rgb, stamp_rgba, x, y, blend_alpha=blend_alpha)
    else:
        full_rgb, stamp_mask = alpha_composite(bg_rgb, stamp_rgba, x, y, blend_alpha=blend_alpha)

    if random.random() < cfg.under_text_prob:
        layer = _make_under_text_layer(H, W)
        alpha = random.uniform(*cfg.under_text_alpha_range)
        full_rgb = simulate_under_text_bleed(full_rgb, stamp_mask, alpha=alpha, layer=layer)

    if random.random() < cfg.bg_perturb_prob:
        full_rgb = perturb_background_only(
            full_rgb,
            stamp_mask,
            b_jitter=cfg.bg_brightness_jitter,
            c_jitter=cfg.bg_contrast_jitter,
            noise_std_range=cfg.bg_noise_std_range,
            blur_prob=cfg.bg_blur_prob,
            blur_k_choices=list(cfg.bg_blur_k_choices),
            jpeg_q_range=cfg.bg_jpeg_quality_range,
        )

    pad = sample_pad(cfg)
    left = max(0, x - pad)
    top = max(0, y - pad)
    right = min(W, x + sw + pad)
    bottom = min(H, y + sh + pad)

    roi = full_rgb[top:bottom, left:right]
    if roi.size == 0:
        return None
    stamp_mask_roi = stamp_mask[top:bottom, left:right]

    roi_bg = roi.copy()
    inside = (stamp_mask_roi > 0)
    bg_crop = bg_rgb[top:bottom, left:right]
    roi_bg[inside] = bg_crop[inside]

    return roi, roi_bg, stamp_mask_roi, pad, stamp_rgba


# ----------------------------- manifest & save -----------------------------
def _save_one(out_path: str, roi_rgb: np.ndarray, jpeg_quality: int) -> None:
    _safe_mkdir(os.path.dirname(out_path))
    _np_to_pil_rgb(roi_rgb).save(out_path, quality=int(jpeg_quality))


def _write_manifest(out_root: str, split: str, rows: List[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df_path = os.path.join(out_root, f"manifest_{split}.csv")
    df.to_csv(df_path, index=False)
    print(f"[OK] wrote manifest: {df_path} ({len(rows)} rows)")


# ----------------------------- generation -----------------------------
def generate_split(
    split_name: str,
    seal_paths: List[str],
    bg_paths_by_type: Dict[str, List[str]],
    out_root: str,
    cfg: GenCfg,
    n_pairs_per_bg: int,
    min_cover: float,
    max_cover: float,
    save_jpeg_quality: int,
    save_debug: bool,
) -> None:
    """
    For each seal, for each background, generate exactly:
      n_pairs_per_bg clean + n_pairs_per_bg defect (defect is derived from the SAME compose).
    """
    split_dir = os.path.join(out_root, split_name)
    _safe_mkdir(split_dir)

    dbg_root = os.path.join(out_root, "_debug", split_name) if save_debug else None
    if dbg_root:
        _safe_mkdir(dbg_root)

    manifest_rows: List[dict] = []

    for seal_path in tqdm(seal_paths, desc=f"{split_name} seals", leave=True):
        seal_id = os.path.splitext(os.path.basename(seal_path))[0]
        sample_global = 0

        for bg_type, bg_list in bg_paths_by_type.items():
            if not bg_list:
                continue

            for bg_idx, bg_path in enumerate(bg_list):
                try:
                    bg_pil = Image.open(bg_path).convert("RGB")
                    bg_rgb = _pil_to_np_rgb(bg_pil)
                except Exception:
                    continue

                for _ in range(int(n_pairs_per_bg)):
                    comp = compose_one(seal_path, bg_rgb, cfg)
                    if comp is None:
                        continue

                    roi_rgb, roi_bg_rgb, stamp_mask_roi, pad, debug_stamp = comp
                    pair_id = f"{seal_id}__{bg_type}__{bg_idx:04d}__s{sample_global:06d}__pad{pad:02d}"
                    sample_idx = sample_global
                    sample_global += 1

                    # CLEAN
                    out_name_clean = f"{pair_id}__vclean.jpg"
                    out_rel_clean = os.path.join(split_name, seal_id, out_name_clean)
                    out_path_clean = os.path.join(out_root, out_rel_clean)
                    _save_one(out_path_clean, roi_rgb, save_jpeg_quality)

                    manifest_rows.append({
                        "rel_path": out_rel_clean,
                        "split": split_name,
                        "seal_id": seal_id,
                        "variant": "clean",
                        "bg_type": bg_type,
                        "bg_idx": int(bg_idx),
                        "pad": int(pad),
                        "sample_idx": int(sample_idx),
                        "pair_id": pair_id,
                    })

                    # DEFECT (always)
                    roi_def, defmask = apply_defect_erase_to_background(
                        roi_rgb=roi_rgb,
                        roi_bg_rgb=roi_bg_rgb,
                        stamp_mask_roi_u8=stamp_mask_roi,
                        min_cover=min_cover,
                        max_cover=max_cover,
                        cfg=cfg,
                    )
                    out_name_def = f"{pair_id}__vdef.jpg"
                    out_rel_def = os.path.join(split_name, seal_id, out_name_def)
                    out_path_def = os.path.join(out_root, out_rel_def)
                    _save_one(out_path_def, roi_def, save_jpeg_quality)

                    manifest_rows.append({
                        "rel_path": out_rel_def,
                        "split": split_name,
                        "seal_id": seal_id,
                        "variant": "defect",
                        "bg_type": bg_type,
                        "bg_idx": int(bg_idx),
                        "pad": int(pad),
                        "sample_idx": int(sample_idx),
                        "pair_id": pair_id,
                    })

                    if dbg_root:
                        dbg_dir = os.path.join(dbg_root, seal_id)
                        _safe_mkdir(dbg_dir)
                        Image.fromarray(stamp_mask_roi, mode="L").save(os.path.join(dbg_dir, f"{pair_id}__stampmask.png"))
                        Image.fromarray(defmask, mode="L").save(os.path.join(dbg_dir, f"{pair_id}__defmask.png"))
                        Image.fromarray(debug_stamp, mode="RGBA").save(os.path.join(dbg_dir, f"{pair_id}__stamp.png"))

    _write_manifest(out_root, split_name, manifest_rows)


def main():
    ap = __import__("argparse").ArgumentParser()

    ap.add_argument("--train_stamp_dir", required=True, help="Train seal PNG dir (RGBA).")
    ap.add_argument("--val_stamp_dir", required=True, help="Val seal PNG dir (RGBA).")

    ap.add_argument("--bg_scan_dir", required=True, help="Scan-like background dir.")
    ap.add_argument("--bg_photo_dir", required=True, help="Photo-like background dir.")
    ap.add_argument("--out_dir", required=True, help="Output root directory.")

    # This is the key knob:
    # per (seal, bg): n_pairs_per_bg clean + n_pairs_per_bg defect
    ap.add_argument("--train_pairs_per_bg", type=int, default=2)
    ap.add_argument("--val_pairs_per_bg", type=int, default=0)

    # defect cover range
    ap.add_argument("--min_cover", type=float, default=0.10)
    ap.add_argument("--max_cover", type=float, default=0.40)

    # misc
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--save_jpeg_quality", type=int, default=90)
    ap.add_argument("--save_debug_masks", type=str, default="False")

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    save_debug = str(args.save_debug_masks).lower() in ("1", "true", "yes", "y")

    # enforce cover constraints to [0.10, 0.40]
    min_cover = float(args.min_cover)
    max_cover = float(args.max_cover)
    min_cover = max(0.10, min(min_cover, 0.40))
    max_cover = max(min_cover, min(max_cover, 0.40))

    train_seals = sorted(glob.glob(os.path.join(args.train_stamp_dir, "*.png")))
    if not train_seals:
        raise RuntimeError(f"No train PNGs found in: {args.train_stamp_dir}")

    val_seals = sorted(glob.glob(os.path.join(args.val_stamp_dir, "*.png")))
    if not val_seals:
        raise RuntimeError(f"No val PNGs found in: {args.val_stamp_dir}")

    bg_paths_by_type = {
        "scan": list_images(args.bg_scan_dir),
        "photo": list_images(args.bg_photo_dir),
    }
    if sum(len(v) for v in bg_paths_by_type.values()) == 0:
        raise RuntimeError("No backgrounds found in bg_scan_dir/bg_photo_dir")

    _safe_mkdir(args.out_dir)
    cfg = GenCfg()

    print("==== Dataset generation config ====")
    print("out_dir:", args.out_dir)
    print("train stamps:", len(train_seals), "val stamps:", len(val_seals))
    print("backgrounds:", {k: len(v) for k, v in bg_paths_by_type.items()})
    print("train_pairs_per_bg:", args.train_pairs_per_bg, "val_pairs_per_bg:", args.val_pairs_per_bg)
    print("per (seal,bg): N clean + N defect, where N = pairs_per_bg")
    print("cover ratio:", (min_cover, max_cover), " (hard-clamped to [0.10, 0.40])")
    print("stamp gray/black prob:", cfg.stamp_gray_prob, "(simulate scan turning stamp gray/black)")
    print("Naming: <seal_id>__<bg_type>__<bg_idx:04d>__s<sample:06d>__pad<pad:02d>__v(clean|def).jpg")
    print("Manifests: manifest_train.csv / manifest_val.csv")
    print("===================================")

    generate_split(
        split_name="train",
        seal_paths=train_seals,
        bg_paths_by_type=bg_paths_by_type,
        out_root=args.out_dir,
        cfg=cfg,
        n_pairs_per_bg=int(args.train_pairs_per_bg),
        min_cover=min_cover,
        max_cover=max_cover,
        save_jpeg_quality=int(args.save_jpeg_quality),
        save_debug=save_debug,
    )

    # generate_split(
    #     split_name="val",
    #     seal_paths=val_seals,
    #     bg_paths_by_type=bg_paths_by_type,
    #     out_root=args.out_dir,
    #     cfg=cfg,
    #     n_pairs_per_bg=int(args.val_pairs_per_bg),
    #     min_cover=min_cover,
    #     max_cover=max_cover,
    #     save_jpeg_quality=int(args.save_jpeg_quality),
    #     save_debug=save_debug,
    # )

    print("\nDone.")
    print("Output structure:")
    print("  out_dir/train/<seal_id>/*.jpg")
    print("  out_dir/val/<seal_id>/*.jpg")
    print("  out_dir/manifest_train.csv")
    print("  out_dir/manifest_val.csv")
    if save_debug:
        print("  out_dir/_debug/train/<seal_id>/*")
        print("  out_dir/_debug/val/<seal_id>/*")


if __name__ == "__main__":
    main()
