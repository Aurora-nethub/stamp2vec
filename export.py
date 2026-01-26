#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  package_dir/
    ├── model.pt
    ├── config.json
    └── manifest.json
"""

import os
import json
import time
import argparse
import platform
import hashlib
from typing import Any, Dict

import torch

from model import SealEmbeddingNet


# ---------------------------
# Utilities
# ---------------------------
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def json_dump(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_float_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


# ---------------------------
# Load checkpoint (lenient)
# ---------------------------
def load_ckpt_flexible(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected, sd


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()

    # required
    ap.add_argument("--ckpt", required=True, help="Trained checkpoint file, e.g., checkpoints_768/latest.pth")
    ap.add_argument("--out", required=True, help="Output package directory, e.g., exported/seal_pkg_v1")

    # model structure (must match training)
    ap.add_argument("--embedding-dim", type=int, default=768)
    ap.add_argument("--model-arch", type=str, default="vit_small_patch14_reg4_dinov2.lvd142m")

    # export-time backbone init (structure only; weights will be overwritten by ckpt)
    ap.add_argument("--local-base-path", type=str,
                    default="./local_models/timm/vit_small_patch14_reg4_dinov2.lvd142m")
    ap.add_argument("--backbone-weight-path", type=str, default="")
    ap.add_argument("--backbone-strict", action="store_true", default=False)

    # preprocess info (written to config.json for inference)
    ap.add_argument("--img-size", type=int, default=518)
    ap.add_argument("--mean", type=str, default="0.485,0.456,0.406")
    ap.add_argument("--std", type=str, default="0.229,0.224,0.225")
    ap.add_argument("--square-pad-fill", type=int, default=255)
    ap.add_argument("--square-pad-mode", type=str, default="constant")
    ap.add_argument("--resize-interp", type=str, default="bilinear")

    ap.add_argument("--cpu", action="store_true", default=False)

    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    os.makedirs(args.out, exist_ok=True)

    mean = parse_float_list(args.mean)
    std = parse_float_list(args.std)

    # 1) Build model (allow loading local backbone weights during export)
    model = SealEmbeddingNet(
        embedding_dim=args.embedding_dim,
        freeze_backbone=False,
        model_arch=args.model_arch,
        local_base_path=args.local_base_path,
        load_backbone_weights=True,
        backbone_weight_path=args.backbone_weight_path,
        backbone_strict=args.backbone_strict,
        verbose=True,
    ).to(device)

    # 2) Load training checkpoint (overwrites all weights)
    missing, unexpected, raw_sd = load_ckpt_flexible(model, args.ckpt, device)
    model.eval()

    # 3) Save full weights
    model_path = os.path.join(args.out, "model.pt")
    torch.save(model.state_dict(), model_path)

    # 4) Write config.json (for inference)
    config = {
        "arch": "SealEmbeddingNet",
        "model_arch": args.model_arch,
        "embedding_dim": int(args.embedding_dim),
        "preprocess": {
            "pipeline": "SquarePad->ExtraPad->Resize->ToTensor->Normalize",
            "img_size": int(args.img_size),
            "rgb": True,
            "square_pad": {
                "fill": int(args.square_pad_fill),
                "mode": str(args.square_pad_mode),
                "align": "center",
            },
            "extra_pad_default": 0,
            "resize": {
                "size": [int(args.img_size), int(args.img_size)],
                "interpolation": str(args.resize_interp),
            },
            "normalize": {
                "mean": mean,
                "std": std,
            },
        },
        "inference": {
            "load_backbone_weights": False
        },
        "source_ckpt": os.path.abspath(args.ckpt),
    }

    config_path = os.path.join(args.out, "config.json")
    json_dump(config_path, config)

    # 5) Write manifest.json (provenance / environment)
    manifest: Dict[str, Any] = {
        "created_at": now_iso(),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "runtime": {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available() else None,
            "gpu_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available() and device.type == "cuda" else None,
        },
        "timm": None,
        "files": {
            "model.pt": {
                "sha256": sha256_file(model_path),
                "bytes": os.path.getsize(model_path),
            },
            "config.json": {
                "sha256": sha256_file(config_path),
                "bytes": os.path.getsize(config_path),
            },
        },
        "checkpoint_source": {
            "path": os.path.abspath(args.ckpt),
            "sha256": sha256_file(args.ckpt)
            if os.path.exists(args.ckpt) else None,
        },
        "state_dict_report": {
            "missing_keys_on_load_ckpt": len(missing),
            "unexpected_keys_on_load_ckpt": len(unexpected),
            "num_state_dict_keys_in_ckpt": len(raw_sd)
            if isinstance(raw_sd, dict) else None,
        },
    }

    try:
        import timm  # noqa
        manifest["timm"] = timm.__version__
    except Exception:
        manifest["timm"] = "import_failed"

    manifest_path = os.path.join(args.out, "manifest.json")
    json_dump(manifest_path, manifest)

    # 6) Summary
    print("=" * 70)
    print(f"Export finished: {os.path.abspath(args.out)}")
    print(f"  - model.pt       : {model_path}")
    print(f"  - config.json    : {config_path}")
    print(f"  - manifest.json  : {manifest_path}")
    if missing:
        print(f"[WARN] missing keys when loading ckpt: {len(missing)}")
    if unexpected:
        print(f"[WARN] unexpected keys when loading ckpt: {len(unexpected)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
