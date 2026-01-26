#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SupCon training loop with adaptive random/pk phases, collapse guards, and rollback-based recovery."""

import os
import csv
import copy
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataset import get_dataloaders
from model import SealEmbeddingNet


# ---------------------------
# Logging
# ---------------------------
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record))
            self.flush()
        except Exception:
            self.handleError(record)


def get_logger(ckpt_dir: str) -> logging.Logger:
    logger = logging.getLogger("SupConTrain")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        os.makedirs(ckpt_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(ckpt_dir, "train.log"), mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(fh)
        logger.addHandler(TqdmLoggingHandler())
    return logger


# ---------------------------
# Loss
# ---------------------------
class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.tau = float(temperature)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        b = emb.size(0)
        if b < 2:
            return emb.new_tensor(0.0)

        labels = labels.view(-1, 1)
        diag = torch.eye(b, dtype=torch.bool, device=emb.device)
        mask_pos = torch.eq(labels, labels.T) & (~diag)

        logits = (emb @ emb.T) / self.tau
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        exp_logits = torch.exp(logits) * (~diag)
        denom = exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
        log_prob = logits - torch.log(denom)

        pos_count = mask_pos.sum(dim=1)
        valid = pos_count > 0
        if valid.sum() == 0:
            return emb.new_tensor(0.0)

        mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / pos_count.clamp_min(1)
        return -mean_log_prob_pos[valid].mean()


# ---------------------------
# Utils
# ---------------------------
def has_nan_inf(x: torch.Tensor) -> bool:
    return bool(torch.isnan(x).any().item() or torch.isinf(x).any().item())


def first_bad_grad(model: nn.Module) -> Optional[str]:
    for n, p in model.named_parameters():
        if p.grad is not None and (torch.isnan(p.grad).any().item() or torch.isinf(p.grad).any().item()):
            return n
    return None


def freeze_patch_embed_proj(model: nn.Module, logger: logging.Logger) -> None:
    frozen = 0
    if hasattr(model, "backbone") and hasattr(model.backbone, "patch_embed"):
        pe = model.backbone.patch_embed
        if hasattr(pe, "proj"):
            for p in pe.proj.parameters():
                if p.requires_grad:
                    p.requires_grad = False
                    frozen += p.numel()
    logger.info(f">>> freeze_patch_embed_proj: frozen_params={frozen}")


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    if hasattr(model, "backbone"):
        for p in model.backbone.parameters():
            p.requires_grad = bool(trainable)


def is_no_decay_param(name: str) -> bool:
    n = name.lower()
    return name.endswith(".bias") or ("norm" in n) or ("bn" in n) or ("batchnorm" in n)


def build_adamw(
    model: nn.Module,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float,
    lr_patch_embed_mult: float = 1.0,
) -> torch.optim.Optimizer:
    """
    Parameter grouping:
      - backbone (except patch_embed) : lr_backbone
      - patch_embed                  : lr_backbone * lr_patch_embed_mult
      - projector/head               : lr_head
    """
    bb_decay, bb_nodecay = [], []
    hd_decay, hd_nodecay = [], []
    pe_decay, pe_nodecay = [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        is_head = name.startswith("projector.") or ("projector" in name and "backbone" not in name)
        is_patch_embed = name.startswith("backbone.patch_embed") or ("backbone.patch_embed" in name)
        no_decay = is_no_decay_param(name)

        if is_head:
            (hd_nodecay if no_decay else hd_decay).append(p)
        elif is_patch_embed:
            (pe_nodecay if no_decay else pe_decay).append(p)
        else:
            (bb_nodecay if no_decay else bb_decay).append(p)

    groups = []
    if bb_decay:
        groups.append({"params": bb_decay, "lr": lr_backbone, "weight_decay": weight_decay})
    if bb_nodecay:
        groups.append({"params": bb_nodecay, "lr": lr_backbone, "weight_decay": 0.0})

    if pe_decay:
        groups.append({"params": pe_decay, "lr": lr_backbone * lr_patch_embed_mult, "weight_decay": weight_decay})
    if pe_nodecay:
        groups.append({"params": pe_nodecay, "lr": lr_backbone * lr_patch_embed_mult, "weight_decay": 0.0})

    if hd_decay:
        groups.append({"params": hd_decay, "lr": lr_head, "weight_decay": weight_decay})
    if hd_nodecay:
        groups.append({"params": hd_nodecay, "lr": lr_head, "weight_decay": 0.0})

    if not groups:
        raise RuntimeError("No trainable params found.")
    return torch.optim.AdamW(groups)


def optimizer_set_lrs(optimizer: torch.optim.Optimizer, lr_backbone: float, lr_head: float, lr_patch_embed_mult: float):
    """
    Our optimizer groups were created in order: bb_decay, bb_nodecay, pe_decay, pe_nodecay, hd_decay, hd_nodecay
    But some groups might be missing depending on freeze. So we update by heuristic:
      - if group lr matches old backbone scale (or is not head) -> set to lr_backbone (or patch_embed)
      - else head -> lr_head
    Safer: we tag groups at creation time, but we keep minimal assumptions.
    """
    for g in optimizer.param_groups:
        tag = g.get("tag", None)
        if tag == "bb":
            g["lr"] = lr_backbone
        elif tag == "pe":
            g["lr"] = lr_backbone * lr_patch_embed_mult
        elif tag == "hd":
            g["lr"] = lr_head
        else:
            # fallback: infer by param name not accessible; use weight_decay as weak signal not good.
            # keep as-is if not tagged.
            pass


def tag_optimizer_groups(optimizer: torch.optim.Optimizer, model: nn.Module):
    """
    Add 'tag' to param_groups for robust LR update.
    Must be called immediately after build_adamw, before training.
    """
    # Build a reverse map param -> name
    p2n = {p: n for n, p in model.named_parameters()}
    for g in optimizer.param_groups:
        names = []
        for p in g["params"]:
            n = p2n.get(p, "")
            if n:
                names.append(n)
        tag = "bb"
        if any(("projector" in n and "backbone" not in n) or n.startswith("projector.") for n in names):
            tag = "hd"
        elif any("backbone.patch_embed" in n for n in names):
            tag = "pe"
        g["tag"] = tag


def supports_bf16() -> bool:
    fn = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            return False
    return False


# ---------------------------
# Validation (full retrieval)
# ---------------------------
@torch.no_grad()
def _embed_indices(
    model: nn.Module,
    ds,
    indices: List[int],
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    feats_list: List[torch.Tensor] = []
    labels_list: List[int] = []

    model.eval()
    for s in range(0, len(indices), batch_size):
        chunk = indices[s:s + batch_size]
        xs = []
        ys = []
        for idx in chunk:
            out = ds[idx]
            if isinstance(out, (tuple, list)) and len(out) == 3:
                x, y, _ = out
            else:
                x, y = out
            xs.append(x)
            ys.append(int(y.item()) if torch.is_tensor(y) else int(y))

        x = torch.stack(xs, dim=0).to(device, non_blocking=True)
        f = model.extract_feat(x)
        f = F.normalize(f.float(), dim=-1, eps=1e-6).cpu()
        feats_list.append(f)
        labels_list.extend(ys)

    feats = torch.cat(feats_list, dim=0) if feats_list else torch.empty(0, 768)
    labels = torch.tensor(labels_list, dtype=torch.long)
    return feats, labels


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    q = min(max(q, 0.0), 1.0)
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    w = pos - lo
    return float(xs[lo] * (1.0 - w) + xs[hi] * w)


def _safe_topk_mean(x: torch.Tensor, k: int) -> float:
    if x.numel() == 0:
        return 0.0
    kk = min(int(k), int(x.numel()))
    return float(torch.topk(x, k=kk).values.mean().item())


@torch.no_grad()
def validate_full_retrieval_with_details(
    model: nn.Module,
    val_loader,
    device: torch.device,
    embed_bs: int = 64,
    hardneg_topk: int = 5,
    export_csv_path: Optional[str] = None,
) -> Dict[str, float]:
    ds = getattr(val_loader, "dataset", None)
    if ds is None or not hasattr(ds, "labels"):
        return {
            "N": 0, "classes": 0,
            "r1_mean": 0.0, "r1_min": 0.0, "r1_p05_cls": 0.0,
            "gap_mean_topk": 0.0, "gap_p05_topk": 0.0,
            "gap_mean_max": 0.0, "gap_p05_max": 0.0,
            "delta_mean": 0.0, "delta_p05": 0.0,
        }

    indices = list(range(len(ds)))
    feats, labels = _embed_indices(model, ds, indices, device=device, batch_size=embed_bs)
    if feats.numel() == 0 or labels.numel() < 3:
        n = int(labels.numel())
        return {
            "N": n,
            "classes": int(labels.unique().numel()) if n else 0,
            "r1_mean": 0.0, "r1_min": 0.0, "r1_p05_cls": 0.0,
            "gap_mean_topk": 0.0, "gap_p05_topk": 0.0,
            "gap_mean_max": 0.0, "gap_p05_max": 0.0,
            "delta_mean": 0.0, "delta_p05": 0.0,
        }

    sim = feats @ feats.t()
    N = labels.numel()

    r1_hits = 0
    cls_total: Dict[int, int] = {}
    cls_hit: Dict[int, int] = {}

    gaps_topk: List[float] = []
    gaps_max: List[float] = []
    deltas: List[float] = []

    rows: List[Dict[str, object]] = []
    img_paths = getattr(ds, "image_paths", None)

    for i in range(N):
        y = int(labels[i].item())

        same = (labels == y)
        same[i] = False
        diff = ~same
        diff[i] = False

        if not same.any() or not diff.any():
            continue

        pos = sim[i, same]
        neg = sim[i, diff]

        intra = float(pos.mean().item())
        hardneg_max = float(neg.max().item())
        hardneg_topk_mean = _safe_topk_mean(neg, hardneg_topk)

        gap_max = intra - hardneg_max
        gap_topk = intra - hardneg_topk_mean

        gaps_max.append(gap_max)
        gaps_topk.append(gap_topk)

        s = sim[i].clone()
        s[i] = -1e9
        top2 = torch.topk(s, k=2).values
        idx2 = torch.topk(s, k=2).indices
        j1 = int(idx2[0].item())
        j2 = int(idx2[1].item())
        sim1 = float(top2[0].item())
        sim2 = float(top2[1].item())
        delta = sim1 - sim2
        deltas.append(delta)

        pred = int(labels[j1].item())
        hit = 1 if pred == y else 0
        r1_hits += hit

        cls_total[y] = cls_total.get(y, 0) + 1
        cls_hit[y] = cls_hit.get(y, 0) + hit

        pos_max = float(pos.max().item()) if pos.numel() else 0.0

        row = {
            "i": i,
            "query": (os.path.basename(img_paths[i]) if isinstance(img_paths, list) else str(i)),
            "y": y,
            "top1_j": j1,
            "top1_y": pred,
            "sim_top1": sim1,
            "top2_j": j2,
            "top2_y": int(labels[j2].item()),
            "sim_top2": sim2,
            "delta_1_2": delta,
            "hit_r1": hit,
            "intra_pos_mean": intra,
            "pos_max": pos_max,
            "hardneg_max": hardneg_max,
            f"hardneg_top{hardneg_topk}_mean": hardneg_topk_mean,
            "gap_max": gap_max,
            f"gap_top{hardneg_topk}": gap_topk,
        }
        rows.append(row)

    classes = sorted(cls_total.keys())
    cls_r1 = [(cls_hit[c] / max(1, cls_total[c])) for c in classes] if classes else [0.0]

    r1_mean = float(r1_hits / max(1, len(rows))) if rows else 0.0
    r1_min = float(min(cls_r1)) if cls_r1 else 0.0
    r1_p05_cls = _percentile(cls_r1, 0.05) if cls_r1 else 0.0

    gap_mean_topk = float(sum(gaps_topk) / len(gaps_topk)) if gaps_topk else 0.0
    gap_p05_topk = _percentile(gaps_topk, 0.05) if gaps_topk else 0.0
    gap_mean_max = float(sum(gaps_max) / len(gaps_max)) if gaps_max else 0.0
    gap_p05_max = _percentile(gaps_max, 0.05) if gaps_max else 0.0

    delta_mean = float(sum(deltas) / len(deltas)) if deltas else 0.0
    delta_p05 = _percentile(deltas, 0.05) if deltas else 0.0

    if export_csv_path is not None:
        os.makedirs(os.path.dirname(export_csv_path), exist_ok=True)
        with open(export_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["empty"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    return {
        "N": int(N),
        "classes": int(len(classes)),
        "r1_mean": r1_mean,
        "r1_min": r1_min,
        "r1_p05_cls": float(r1_p05_cls),
        "gap_mean_topk": gap_mean_topk,
        "gap_p05_topk": float(gap_p05_topk),
        "gap_mean_max": gap_mean_max,
        "gap_p05_max": float(gap_p05_max),
        "delta_mean": delta_mean,
        "delta_p05": float(delta_p05),
    }


# ---------------------------
# Dataloader wrapper (compat)
# ---------------------------
def _build_dataloaders_safe(**kwargs):
    try:
        return get_dataloaders(**kwargs)
    except TypeError:
        kwargs.pop("return_val_index", None)
        kwargs.pop("random_epoch_len", None)
        return get_dataloaders(**kwargs)


# ---------------------------
# Snapshot comparison (delivery)
# ---------------------------
def _is_better(a: Dict[str, float], b: Dict[str, float]) -> bool:
    """
    Lexicographic comparison:
      1) r1_min        higher better
      2) r1_mean       higher better
      3) delta_p05     higher better
      4) gap_p05_topk  higher better
    """
    keys = ["r1_min", "r1_mean", "delta_p05", "gap_p05_topk"]
    for k in keys:
        av = float(a.get(k, 0.0))
        bv = float(b.get(k, 0.0))
        if av > bv + 1e-12:
            return True
        if av < bv - 1e-12:
            return False
    return False


def _is_collapse(val: Dict[str, float]) -> bool:
    """
    Collapse guard (tuned to your logs):
    - r1_mean drops below 0.60 OR r1_min below 0.30
    - delta_p05 ~ 0 indicates top1/top2 indistinguishable
    """
    r1_mean = float(val.get("r1_mean", 0.0))
    r1_min = float(val.get("r1_min", 0.0))
    delta_p05 = float(val.get("delta_p05", 0.0))
    if r1_mean < 0.60:
        return True
    if r1_min < 0.30:
        return True
    if delta_p05 < 5e-5:
        return True
    return False


def _is_hard_degrade(cur: Dict[str, float], best: Dict[str, float]) -> bool:
    """
    Hard degrade trigger in RANDOM:
    - r1_min drop >= 0.10 or r1_mean drop >= 0.05 vs best
    This prevents switching just because "no improve" on noisy small VAL.
    """
    r1m = float(cur.get("r1_mean", 0.0))
    r1mn = float(cur.get("r1_min", 0.0))
    br1m = float(best.get("r1_mean", 0.0))
    br1mn = float(best.get("r1_min", 0.0))

    if (br1mn - r1mn) >= 0.10:
        return True
    if (br1m - r1m) >= 0.05:
        return True
    return False


# ---------------------------
# Main train
# ---------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = "./checkpoints_768"
    ckpt_path = os.path.join(ckpt_dir, "latest.pth")
    best_path = os.path.join(ckpt_dir, "best_model.pth")
    details_dir = os.path.join(ckpt_dir, "val_details")

    train_path = os.path.expanduser("/home/zhangjunjie/data_generation/output_dataset/train")
    val_path = os.path.expanduser("/home/zhangjunjie/stamp_cmp/stamps")

    batch_size = 32
    img_size = 518
    num_workers = 4
    key_level = "bg_s_pad"

    # Epoch length caps (in batches)
    pk_P, pk_K = 8, 4
    pk_epoch_len = 400
    random_epoch_len = 1000

    max_epochs = 200
    patience_global = 20

    # Base LR (RANDOM)
    lr_head_base = 5e-5
    lr_backbone_base = 1e-6
    weight_decay = 1e-2

    # LR multipliers used for recovery / pk soft landing
    lr_drop_factor_recover = 0.25   # after hard degrade/collapse: LR *= 0.25
    lr_drop_factor_pk = 0.20        # before entering PK: LR *= 0.20 (soft landing)
    lr_drop_factor_pk_more = 0.50   # if PK still unstable: LR *= 0.50 again

    # SupCon
    supcon_tau = 0.10

    # Stability
    clip_norm_random = 0.5
    clip_norm_pk = 0.3
    max_bad_batches_per_epoch = 10

    use_amp = True
    prefer_bf16 = True
    disable_amp_first_epoch = False
    scaler_init_scale = 2 ** 8
    scaler_growth_interval = 2000
    scaler_backoff_factor = 0.5

    freeze_patch_embed = True
    lr_patch_embed_mult = 0.1

    # Validation settings
    val_embed_bs = 64
    hardneg_topk = 5
    export_val_csv_each_epoch = True

    # Phase logic
    train_mode = "random"
    no_improve = 0
    global_no_improve = 0

    # How sensitive to switch:
    max_no_improve_before_action = 20       # after 20 no-improve, do recovery (rollback+lr drop) BEFORE switching
    recovery_epochs_to_try = 1             # number of recovery epochs to run before switching
    pk_warmup_freeze_backbone_epochs = 1   # 1 epoch only train projector in PK (stability)

    logger = get_logger(ckpt_dir)

    # Build VAL once
    _, val_loader, num_classes = _build_dataloaders_safe(
        train_path=train_path,
        val_path=val_path,
        batch_size=batch_size,
        img_size=img_size,
        mode="random",  # ignored
        key_level=key_level,
        num_workers=num_workers,
        seed=123,
        return_val_index=True,
    )
    logger.info(f">>> num_classes={num_classes}")

    # Model
    model = SealEmbeddingNet(embedding_dim=768, freeze_backbone=False).to(device)
    if hasattr(model, "set_backbone_trainable"):
        model.set_backbone_trainable(True)
    else:
        for p in model.parameters():
            p.requires_grad = True

    if freeze_patch_embed:
        freeze_patch_embed_proj(model, logger)

    # Optimizer
    lr_head = lr_head_base
    lr_backbone = lr_backbone_base
    optimizer = build_adamw(
        model,
        lr_backbone=lr_backbone,
        lr_head=lr_head,
        weight_decay=weight_decay,
        lr_patch_embed_mult=(0.0 if freeze_patch_embed else lr_patch_embed_mult),
    )
    tag_optimizer_groups(optimizer, model)

    criterion = SupConLoss(temperature=supcon_tau)

    # AMP
    amp_mode = "fp32"
    scaler = None
    autocast_dtype = torch.float16
    if device.type == "cuda":
        bf16_ok = supports_bf16()
        if use_amp and prefer_bf16 and bf16_ok:
            amp_mode = "bf16"
            autocast_dtype = torch.bfloat16
            scaler = None
        else:
            amp_mode = "fp16" if use_amp else "fp32"
            autocast_dtype = torch.float16
            scaler = torch.amp.GradScaler(
                "cuda",
                enabled=(use_amp and amp_mode == "fp16"),
                init_scale=scaler_init_scale,
                growth_interval=scaler_growth_interval,
                backoff_factor=scaler_backoff_factor,
            )
        logger.info(f">>> AMP: requested={use_amp}, mode={amp_mode}, bf16_supported={bf16_ok}")

    # Dataloader builder
    def rebuild_train_loader(mode: str):
        nonlocal train_mode
        train_mode = mode
        if mode == "random":
            tl, _, _ = _build_dataloaders_safe(
                train_path=train_path,
                val_path=val_path,
                batch_size=batch_size,
                img_size=img_size,
                mode="random",
                key_level=key_level,
                num_workers=num_workers,
                seed=123,
                random_epoch_len=random_epoch_len,
                return_val_index=True,
            )
            logger.info(f">>> TRAIN mode=random (batch_size={batch_size}, epoch_len={random_epoch_len})")
            return tl
        elif mode == "pk":
            tl, _, _ = _build_dataloaders_safe(
                train_path=train_path,
                val_path=val_path,
                batch_size=batch_size,
                img_size=img_size,
                mode="pk",
                key_level=key_level,
                num_workers=num_workers,
                pk_P=pk_P,
                pk_K=pk_K,
                seed=123,
                pk_epoch_len=pk_epoch_len,
                return_val_index=True,
            )
            logger.info(f">>> TRAIN mode=pk (P={pk_P}, K={pk_K}, epoch_len={pk_epoch_len}, eff_bs={pk_P*pk_K})")
            return tl
        else:
            raise ValueError(f"Unknown mode: {mode}")

    train_loader = rebuild_train_loader("random")

    # Best snapshot + best weights (in-memory rollback)
    best_snapshot: Dict[str, float] = {
        "r1_min": -1.0, "r1_mean": -1.0, "delta_p05": -1e9, "gap_p05_topk": -1e9
    }
    best_state_dict = copy.deepcopy(model.state_dict())
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    best_scaler_state = copy.deepcopy(scaler.state_dict()) if scaler is not None else None

    # Recovery state
    pending_recovery = 0        # remaining recovery epochs to run
    pending_pk_warmup = 0       # remaining pk warmup epochs (freeze backbone)
    pk_unstable_strikes = 0     # counts pk collapses

    start_epoch = 0
    resume = False
    if resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            logger.info(f">>> optimizer not loaded: {e}")
        if scaler is not None and ckpt.get("scaler") is not None:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                logger.info(f">>> scaler not loaded: {e}")

        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_snapshot = ckpt.get("best_snapshot", best_snapshot)
        train_mode = ckpt.get("train_mode", train_mode)
        no_improve = int(ckpt.get("no_improve", no_improve))
        global_no_improve = int(ckpt.get("global_no_improve", global_no_improve))

        lr_head = float(ckpt.get("lr_head", lr_head))
        lr_backbone = float(ckpt.get("lr_backbone", lr_backbone))
        pk_unstable_strikes = int(ckpt.get("pk_unstable_strikes", pk_unstable_strikes))
        pending_recovery = int(ckpt.get("pending_recovery", pending_recovery))
        pending_pk_warmup = int(ckpt.get("pending_pk_warmup", pending_pk_warmup))

        # rebuild loader
        train_loader = rebuild_train_loader(train_mode)
        logger.info(
            f">>> resumed: start_epoch={start_epoch}, mode={train_mode}, "
            f"no_improve={no_improve}, global_no_improve={global_no_improve}, "
            f"lr_bb={lr_backbone:.2e}, lr_hd={lr_head:.2e}, best_snapshot={best_snapshot}"
        )

    for epoch in range(start_epoch, max_epochs):
        # Warmup behavior: in PK warmup epochs, freeze backbone
        if train_mode == "pk" and pending_pk_warmup > 0:
            set_backbone_trainable(model, False)
        else:
            set_backbone_trainable(model, True)

        model.train()
        epoch_use_amp = (device.type == "cuda") and use_amp and not (disable_amp_first_epoch and epoch == 0)

        # Clip norm per mode
        clip_norm = clip_norm_pk if train_mode == "pk" else clip_norm_random

        t_loss, steps_done = 0.0, 0
        skipped_no_pos, bad_batches = 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train:{train_mode}]", leave=False)
        for step, (imgs, labels) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if device.type == "cuda":
                with torch.amp.autocast("cuda", enabled=epoch_use_amp, dtype=autocast_dtype):
                    emb = model(imgs)
            else:
                emb = model(imgs)

            if has_nan_inf(emb):
                bad_batches += 1
                logger.info(f"[BAD] emb NaN/Inf epoch={epoch} step={step} bad={bad_batches}")
                if bad_batches >= max_bad_batches_per_epoch:
                    break
                continue

            if device.type == "cuda":
                with torch.amp.autocast("cuda", enabled=False):
                    emb32 = F.normalize(emb.float(), dim=-1, eps=1e-6)
                    loss = criterion(emb32, labels)
            else:
                emb32 = F.normalize(emb.float(), dim=-1, eps=1e-6)
                loss = criterion(emb32, labels)

            if float(loss.detach().item()) == 0.0:
                skipped_no_pos += 1
                continue

            if torch.isnan(loss).item() or torch.isinf(loss).item():
                bad_batches += 1
                logger.info(f"[BAD] loss NaN/Inf epoch={epoch} step={step} bad={bad_batches}")
                if bad_batches >= max_bad_batches_per_epoch:
                    break
                continue

            if device.type == "cuda" and scaler is not None and epoch_use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

            badg = first_bad_grad(model)
            if badg is not None:
                bad_batches += 1
                logger.info(
                    f"[BAD] grad NaN/Inf: {badg} epoch={epoch} step={step} "
                    f"gnorm={float(total_norm):.3e} scale={(scaler.get_scale() if scaler is not None else 'NA')} "
                    f"bad={bad_batches}"
                )
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda" and scaler is not None and epoch_use_amp:
                    try:
                        scaler.update()
                    except Exception:
                        pass
                if bad_batches >= max_bad_batches_per_epoch:
                    break
                continue

            if device.type == "cuda" and scaler is not None and epoch_use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            t_loss += float(loss.item())
            steps_done += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", gnorm=f"{float(total_norm):.2e}", bad=f"{bad_batches}")

        if steps_done == 0:
            logger.info(f"Epoch {epoch}: 0 steps. skipped_no_pos={skipped_no_pos}, bad_batches={bad_batches}")
            break

        avg_loss = t_loss / steps_done

        export_csv = None
        if export_val_csv_each_epoch:
            export_csv = os.path.join(details_dir, f"val_details_epoch{epoch:04d}.csv")

        val = validate_full_retrieval_with_details(
            model,
            val_loader,
            device=device,
            embed_bs=val_embed_bs,
            hardneg_topk=hardneg_topk,
            export_csv_path=export_csv,
        )

        snapshot = {
            "r1_min": float(val["r1_min"]),
            "r1_mean": float(val["r1_mean"]),
            "delta_p05": float(val["delta_p05"]),
            "gap_p05_topk": float(val["gap_p05_topk"]),
        }

        logger.info(
            f"Epoch {epoch}: TrainLoss={avg_loss:.4f} | TRAIN={train_mode} | "
            f"VAL(full N={val['N']}, classes={val['classes']}) "
            f"R1(mean={val['r1_mean']:.4f}, min_cls={val['r1_min']:.4f}, p05_cls={val['r1_p05_cls']:.4f}) "
            f"| Δ(mean={val['delta_mean']:.6f}, p05={val['delta_p05']:.6f}) "
            f"| GAP_top{hardneg_topk}(mean={val['gap_mean_topk']:.4f}, p05={val['gap_p05_topk']:.4f}) "
            f"| GAP_max(mean={val['gap_mean_max']:.4f}, p05={val['gap_p05_max']:.4f}) "
            f"| skipped_no_pos={skipped_no_pos} bad_batches={bad_batches} "
            f"| lr(bb={lr_backbone:.2e}, hd={lr_head:.2e}) "
            f"{('| export=' + export_csv) if export_csv else ''}"
        )

        # Save latest checkpoint each epoch
        latest = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "amp_mode": amp_mode,
            "freeze_patch_embed": freeze_patch_embed,
            "clip_norm": clip_norm,
            "key_level": key_level,
            "val_embed_bs": val_embed_bs,
            "val_snapshot": val,
            "best_snapshot": best_snapshot,
            "train_mode": train_mode,
            "no_improve": no_improve,
            "global_no_improve": global_no_improve,
            "paths": {"train_path": train_path, "val_path": val_path},
            "pk": {"P": pk_P, "K": pk_K, "epoch_len": pk_epoch_len},
            "random_epoch_len": random_epoch_len,
            "lrs": {"lr_backbone": lr_backbone, "lr_head": lr_head, "weight_decay": weight_decay},
            "supcon_tau": supcon_tau,
            "hardneg_topk": hardneg_topk,
            "pk_unstable_strikes": pk_unstable_strikes,
            "pending_recovery": pending_recovery,
            "pending_pk_warmup": pending_pk_warmup,
        }
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(latest, ckpt_path)

        # Collapse handling (mode-aware)
        collapsed = _is_collapse(val)
        if collapsed:
            logger.info(
                f">>> COLLAPSE detected! mode={train_mode} | "
                f"r1_mean={val['r1_mean']:.4f}, r1_min={val['r1_min']:.4f}, delta_p05={val['delta_p05']:.6f}"
            )

            # Rollback to best weights immediately
            model.load_state_dict(best_state_dict, strict=True)
            try:
                optimizer.load_state_dict(best_optimizer_state)
            except Exception as e:
                logger.info(f">>> optimizer rollback failed (will rebuild): {e}")

            if scaler is not None and best_scaler_state is not None:
                try:
                    scaler.load_state_dict(best_scaler_state)
                except Exception as e:
                    logger.info(f">>> scaler rollback failed: {e}")

            # LR drop
            lr_backbone = max(lr_backbone * lr_drop_factor_recover, 1e-8)
            lr_head = max(lr_head * lr_drop_factor_recover, 1e-7)
            optimizer_set_lrs(optimizer, lr_backbone, lr_head, (0.0 if freeze_patch_embed else lr_patch_embed_mult))
            logger.info(f">>> after rollback: LR dropped to bb={lr_backbone:.2e}, hd={lr_head:.2e}")

            # If collapse happened in PK, count strike and optionally retreat to RANDOM
            if train_mode == "pk":
                pk_unstable_strikes += 1
                if pk_unstable_strikes >= 2:
                    logger.info(">>> PK unstable strikes>=2, retreat to RANDOM (recovery)")
                    train_loader = rebuild_train_loader("random")
                    pending_recovery = recovery_epochs_to_try
                    pending_pk_warmup = 0
                    pk_unstable_strikes = 0
                else:
                    # stay in PK but enforce warmup freeze again
                    pending_pk_warmup = max(pending_pk_warmup, pk_warmup_freeze_backbone_epochs)
            else:
                # collapse in RANDOM: do recovery epoch(s)
                pending_recovery = max(pending_recovery, recovery_epochs_to_try)

            no_improve += 1
            global_no_improve += 1
            if global_no_improve >= patience_global:
                logger.info(f">>> early stop at epoch={epoch} (global_no_improve={global_no_improve})")
                break

            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        # Normal improvement logic
        improved = _is_better(snapshot, best_snapshot)
        if improved:
            best_snapshot = dict(snapshot)
            best_state_dict = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_scaler_state = copy.deepcopy(scaler.state_dict()) if scaler is not None else None

            no_improve = 0
            global_no_improve = 0
            pk_unstable_strikes = 0

            best_ckpt = dict(latest)
            best_ckpt["best_snapshot"] = best_snapshot
            torch.save(best_ckpt, best_path)

            logger.info(f"✨ New best saved: best_snapshot={best_snapshot}")
        else:
            no_improve += 1
            global_no_improve += 1
            logger.info(
                f">>> no improve={no_improve} (mode={train_mode}) | global={global_no_improve}/{patience_global} "
                f"(best_snapshot={best_snapshot})"
            )

        # Decide actions after validation (event-driven)
        if train_mode == "random":
            hard_degrade = _is_hard_degrade(snapshot, best_snapshot)

            if hard_degrade or no_improve >= max_no_improve_before_action:
                # Step 1: rollback to best + LR drop
                logger.info(
                    f">>> RANDOM action triggered (hard_degrade={hard_degrade}, no_improve={no_improve}). "
                    f"Do rollback+LR drop then {recovery_epochs_to_try} recovery epoch(s)."
                )
                model.load_state_dict(best_state_dict, strict=True)
                try:
                    optimizer.load_state_dict(best_optimizer_state)
                except Exception:
                    # rebuild optimizer if state is incompatible
                    optimizer = build_adamw(
                        model,
                        lr_backbone=lr_backbone,
                        lr_head=lr_head,
                        weight_decay=weight_decay,
                        lr_patch_embed_mult=(0.0 if freeze_patch_embed else lr_patch_embed_mult),
                    )
                    tag_optimizer_groups(optimizer, model)

                if scaler is not None and best_scaler_state is not None:
                    try:
                        scaler.load_state_dict(best_scaler_state)
                    except Exception:
                        pass

                lr_backbone = max(lr_backbone * lr_drop_factor_recover, 1e-8)
                lr_head = max(lr_head * lr_drop_factor_recover, 1e-7)
                optimizer_set_lrs(optimizer, lr_backbone, lr_head, (0.0 if freeze_patch_embed else lr_patch_embed_mult))
                logger.info(f">>> LR dropped to bb={lr_backbone:.2e}, hd={lr_head:.2e} (RANDOM recovery)")

                pending_recovery = recovery_epochs_to_try
                no_improve = 0  # reset local counter after taking action

            # If we just finished recovery epochs and still not improving, consider switching to PK
            if pending_recovery > 0:
                pending_recovery -= 1
                if pending_recovery == 0:
                    # After recovery epoch(s), if still no clear improvement pressure, soft-land into PK
                    # Condition: global_no_improve is accumulating OR snapshot is not better than best (still)
                    if not improved:
                        logger.info(
                            ">>> recovery finished and still not improving -> SWITCH to PK with soft-landing LR."
                        )
                        # Enter PK from best weights
                        model.load_state_dict(best_state_dict, strict=True)
                        try:
                            optimizer.load_state_dict(best_optimizer_state)
                        except Exception:
                            pass
                        if scaler is not None and best_scaler_state is not None:
                            try:
                                scaler.load_state_dict(best_scaler_state)
                            except Exception:
                                pass

                        # Soft landing LR for PK
                        lr_backbone = max(lr_backbone * lr_drop_factor_pk, 1e-8)
                        lr_head = max(lr_head * lr_drop_factor_pk, 1e-7)
                        optimizer_set_lrs(
                            optimizer, lr_backbone, lr_head, (0.0 if freeze_patch_embed else lr_patch_embed_mult)
                        )
                        logger.info(f">>> enter PK: LR soft-landing bb={lr_backbone:.2e}, hd={lr_head:.2e}")

                        train_loader = rebuild_train_loader("pk")
                        pending_pk_warmup = pk_warmup_freeze_backbone_epochs
                        pk_unstable_strikes = 0

        elif train_mode == "pk":
            # PK warmup countdown
            if pending_pk_warmup > 0:
                pending_pk_warmup -= 1
                if pending_pk_warmup == 0:
                    logger.info(">>> PK warmup finished: backbone unfrozen for subsequent epochs.")

            # If PK no-improve keeps going, reduce LR slightly (without switching back immediately)
            if no_improve >= 3:
                lr_backbone = max(lr_backbone * lr_drop_factor_pk_more, 1e-8)
                lr_head = max(lr_head * lr_drop_factor_pk_more, 1e-7)
                optimizer_set_lrs(optimizer, lr_backbone, lr_head, (0.0 if freeze_patch_embed else lr_patch_embed_mult))
                logger.info(f">>> PK no-improve>=3: LR reduced to bb={lr_backbone:.2e}, hd={lr_head:.2e}")
                no_improve = 0

        # Global early stop
        if global_no_improve >= patience_global:
            logger.info(f">>> early stop at epoch={epoch} (global_no_improve={global_no_improve})")
            break

        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    train()
