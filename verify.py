#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
from typing import List

import torch
from PIL import Image
import pandas as pd

from torchvision import transforms
import torchvision.transforms.functional as TF

from model import SealEmbeddingNet


class SquarePad:
    def __init__(self, fill: int = 255, mode: str = "constant"):
        self.fill = int(fill)
        self.mode = mode

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return TF.pad(image, padding, fill=self.fill, padding_mode=self.mode)


class ExtraPad:
    def __init__(self, pad: int, fill: int = 255, mode: str = "constant"):
        self.pad = int(pad)
        self.fill = int(fill)
        self.mode = mode

    def __call__(self, image):
        if self.pad <= 0:
            return image
        padding = (self.pad, self.pad, self.pad, self.pad)
        return TF.pad(image, padding, fill=self.fill, padding_mode=self.mode)


def build_transform_from_cfg(cfg: dict, extra_pad: int = 0):
    pp = cfg["preprocess"]
    img_size = int(pp["img_size"])
    mean = pp["normalize"]["mean"]
    std = pp["normalize"]["std"]
    sp = pp["square_pad"]
    fill = int(sp.get("fill", 255))
    mode = sp.get("mode", "constant")

    interp_name = pp.get("resize", {}).get("interpolation", "bilinear").lower()
    if interp_name == "bilinear":
        interp = transforms.InterpolationMode.BILINEAR
    elif interp_name == "bicubic":
        interp = transforms.InterpolationMode.BICUBIC
    elif interp_name == "nearest":
        interp = transforms.InterpolationMode.NEAREST
    else:
        raise ValueError(f"Unknown interpolation: {interp_name}")

    return transforms.Compose([
        SquarePad(fill=fill, mode=mode),
        ExtraPad(extra_pad, fill=fill, mode=mode),
        transforms.Resize((img_size, img_size), interpolation=interp),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def list_images(d: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
    if not os.path.isdir(d):
        raise SystemExit(f"Not a directory: {d}")
    return sorted([n for n in os.listdir(d) if n.lower().endswith(exts)])


@torch.no_grad()
def extract_feats(
    model: SealEmbeddingNet,
    data_dir: str,
    names: List[str],
    transform,
    device: torch.device,
    batch_size: int = 32,
) -> torch.Tensor:
    feats = []
    for i in range(0, len(names), batch_size):
        batch_names = names[i:i + batch_size]
        xs = []
        for n in batch_names:
            img = Image.open(os.path.join(data_dir, n)).convert("RGB")
            xs.append(transform(img))
        x = torch.stack(xs, dim=0).to(device)
        f = model.extract_feat(x).cpu()  # [B, D] L2-normalized
        feats.append(f)
    return torch.cat(feats, dim=0)  # [N, D]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--query-dir", default=os.path.expanduser("./data/query"))
    ap.add_argument("--cand-dir", default=os.path.expanduser("./data/candidate"))

    ap.add_argument(
        "--pkg",
        default="exported/seal_pkg_v1",
        help="Exported package directory (default: exported/seal_pkg_v1)"
    )

    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out-csv", type=str, default="", help="Optional path to save the similarity matrix CSV")

    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_path = os.path.join(args.pkg, "config.json")
    w_path = os.path.join(args.pkg, "model.pt")
    if not os.path.exists(cfg_path):
        raise SystemExit(f"Missing config.json: {cfg_path}")
    if not os.path.exists(w_path):
        raise SystemExit(f"Missing model.pt: {w_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = SealEmbeddingNet(
        embedding_dim=int(cfg["embedding_dim"]),
        freeze_backbone=False,
        model_arch=str(cfg["model_arch"]),
        load_backbone_weights=False,
        verbose=True,
    ).to(device)

    sd = torch.load(w_path, map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval()

    print(f"Loaded package: {args.pkg}")
    if missing:
        print(f"[WARN] missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)}")

    q_names = list_images(args.query_dir)
    c_names = list_images(args.cand_dir)
    if not q_names:
        raise SystemExit(f"Empty query dir: {args.query_dir}")
    if not c_names:
        raise SystemExit(f"Empty candidate dir: {args.cand_dir}")

    transform = build_transform_from_cfg(cfg, extra_pad=0)

    q_feats = extract_feats(model, args.query_dir, q_names, transform, device, args.batch_size)  # [Nq, D]
    c_feats = extract_feats(model, args.cand_dir, c_names, transform, device, args.batch_size)  # [Nc, D]

    print(f"Query embedding shape: {list(q_feats.shape)}")
    print(f"Cand  embedding shape: {list(c_feats.shape)}")

    sim = q_feats @ c_feats.t()  # cosine similarity (L2-normalized embeddings)

    df = pd.DataFrame(sim.numpy(), index=q_names, columns=c_names)

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)

    print("\n==============================")
    print("Similarity matrix (Query × Candidate)")
    print("==============================")
    print(df.round(4).to_string())

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        df.to_csv(args.out_csv, float_format="%.6f")
        print(f"\nSaved similarity CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
