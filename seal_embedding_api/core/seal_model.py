import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models import load_checkpoint


class SealEmbeddingNet(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        freeze_backbone: bool = True,
        model_arch: str = "vit_small_patch14_reg4_dinov2.lvd142m",
        local_base_path: str = "./local_models/timm/vit_small_patch14_reg4_dinov2.lvd142m",
        load_backbone_weights: bool = True,
        backbone_weight_path: str = "",
        backbone_strict: bool = True,
        verbose: bool = True,
    ):
        super().__init__()

        self.model_arch = model_arch
        self.embedding_dim = embedding_dim
        self.freeze_backbone = freeze_backbone

        self.backbone = timm.create_model(model_arch, pretrained=False, num_classes=0)

        if load_backbone_weights:
            local_weight_path = backbone_weight_path.strip() if backbone_weight_path else ""

            if not local_weight_path:
                base_path = local_base_path
                if not os.path.exists(base_path):
                    base_path = "./local_models/timm/vit_small_patch14_reg4_dinov2__lvd142m"

                cand1 = os.path.join(base_path, "model.safetensors")
                cand2 = os.path.join(base_path, "pytorch_model.bin")
                if os.path.exists(cand1):
                    local_weight_path = cand1
                elif os.path.exists(cand2):
                    local_weight_path = cand2
                else:
                    local_weight_path = ""

            if local_weight_path and os.path.exists(local_weight_path):
                load_checkpoint(self.backbone, local_weight_path, strict=backbone_strict)
                if verbose:
                    print(f"Loaded local backbone weights: {local_weight_path}")
            else:
                raise FileNotFoundError(
                    "Backbone weights not found. "
                    "Set load_backbone_weights=True but no weights were located under local_base_path/backbone_weight_path."
                )
        else:
            if verbose:
                print("Skipping backbone weight load (structure only) because load_backbone_weights=False")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        in_dim = self.backbone.num_features

        self.projector = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, embedding_dim),
        )

    def set_backbone_trainable(self, trainable: bool):
        for p in self.backbone.parameters():
            p.requires_grad = trainable
        if not trainable:
            self.backbone.eval()

    def forward(self, x):
        feat = self.backbone(x)
        emb = self.projector(feat)
        emb = F.normalize(emb, p=2, dim=1)
        return emb

    @torch.no_grad()
    def extract_feat(self, x):
        self.eval()
        return self.forward(x)

    def to_config(self, img_size: int = 518, mean=None, std=None) -> dict:
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        return {
            "arch": "SealEmbeddingNet",
            "model_arch": self.model_arch,
            "embedding_dim": int(self.embedding_dim),
            "freeze_backbone": bool(self.freeze_backbone),
            "preprocess": {
                "img_size": int(img_size),
                "mean": list(map(float, mean)),
                "std": list(map(float, std)),
                "square_pad_fill": 255,
            },
        }

    @staticmethod
    def from_package(pkg_dir: str, device: torch.device, verbose: bool = True):
        cfg_path = os.path.join(pkg_dir, "config.json")
        w_path = os.path.join(pkg_dir, "model.pt")

        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Missing config.json: {cfg_path}")
        if not os.path.exists(w_path):
            raise FileNotFoundError(f"Missing model.pt: {w_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        model = SealEmbeddingNet(
            embedding_dim=cfg["embedding_dim"],
            freeze_backbone=cfg.get("freeze_backbone", False),
            model_arch=cfg["model_arch"],
            load_backbone_weights=False,
            verbose=verbose,
        ).to(device)

        sd = torch.load(w_path, map_location=device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        model.eval()

        if verbose:
            print(f"Loaded package: {pkg_dir}")
            if missing:
                print(f"[WARN] missing keys: {len(missing)}")
            if unexpected:
                print(f"[WARN] unexpected keys: {len(unexpected)}")

        return model, cfg
