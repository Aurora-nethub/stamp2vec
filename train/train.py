#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import csv
import copy
import random
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .dataset import get_dataloaders
from .model import SealEmbeddingNet


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
    logger = logging.getLogger("SupConTrain-PKOnly-OnlineK+DiagK")
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
    def __init__(self, temperature: float = 0.10):
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
        groups.append({"params": bb_decay, "lr": lr_backbone, "weight_decay": weight_decay, "tag": "bb"})
    if bb_nodecay:
        groups.append({"params": bb_nodecay, "lr": lr_backbone, "weight_decay": 0.0, "tag": "bb"})

    if pe_decay:
        groups.append({"params": pe_decay, "lr": lr_backbone * lr_patch_embed_mult, "weight_decay": weight_decay, "tag": "pe"})
    if pe_nodecay:
        groups.append({"params": pe_nodecay, "lr": lr_backbone * lr_patch_embed_mult, "weight_decay": 0.0, "tag": "pe"})

    if hd_decay:
        groups.append({"params": hd_decay, "lr": lr_head, "weight_decay": weight_decay, "tag": "hd"})
    if hd_nodecay:
        groups.append({"params": hd_nodecay, "lr": lr_head, "weight_decay": 0.0, "tag": "hd"})

    if not groups:
        raise RuntimeError("No trainable params found.")
    return torch.optim.AdamW(groups)


def tag_optimizer_groups(optimizer: torch.optim.Optimizer, model: nn.Module) -> None:
    """
    Ensure param_groups have stable tags after loading old checkpoints.
    """
    p2n = {p: n for n, p in model.named_parameters()}
    for g in optimizer.param_groups:
        if "tag" in g:
            continue
        names = [p2n.get(p, "") for p in g.get("params", [])]
        tag = "bb"
        if any(("projector" in n and "backbone" not in n) or n.startswith("projector.") for n in names):
            tag = "hd"
        elif any("backbone.patch_embed" in n for n in names):
            tag = "pe"
        g["tag"] = tag


def optimizer_set_lrs(optimizer: torch.optim.Optimizer, lr_backbone: float, lr_head: float, lr_patch_embed_mult: float):
    for g in optimizer.param_groups:
        tag = g.get("tag", None)
        if tag == "bb":
            g["lr"] = lr_backbone
        elif tag == "pe":
            g["lr"] = lr_backbone * lr_patch_embed_mult
        elif tag == "hd":
            g["lr"] = lr_head


def supports_bf16() -> bool:
    fn = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            return False
    return False


# ---------------------------
# Dataloader wrapper (compat)
# ---------------------------
def _build_dataloaders_safe(**kwargs):
    try:
        return get_dataloaders(**kwargs)
    except TypeError:
        # compat for older dataset.py
        kwargs.pop("return_val_index", None)
        kwargs.pop("random_epoch_len", None)
        kwargs.pop("pk_epoch_len", None)
        return get_dataloaders(**kwargs)


# ---------------------------
# Validation: embedding helpers
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

    if feats_list:
        feats = torch.cat(feats_list, dim=0)
    else:
        feats = torch.empty(0, 768)

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


# ---------------------------
# Validation: core (ONLINE-K / DIAG-K both use this)
# ---------------------------
@torch.no_grad()
def validate_k_template_multisplit(
    model: nn.Module,
    val_loader,
    device: torch.device,
    template_k: int,
    agg: str = "max",
    embed_bs: int = 64,
    base_seed: int = 123,
    num_splits: int = 10,
    min_q_per_cls: int = 2,
    export_csv_path: Optional[str] = None,
    export_confuse_topn: int = 10,
    compute_thr_diag: bool = False,
) -> Dict[str, float]:
    """
    K-template multi-split evaluation with a crucial constraint:
    - Enforce at least min_q_per_cls queries per class (otherwise skip that class).
      This prevents the validation from degenerating into an easy leave-one-out game.

    If compute_thr_diag=True, also computes:
      pos_p05, neg_p95/neg_p99, tpr@FAR5, tpr@FAR1

    Returns metrics:
      tpl_acc_mean/min/p05_cls, margin_mean/p05, gap_mean/p05,
      confuse_rate, confuse_rate_max_cls, avg_q_per_cls,
      worst_confuse_pairs (string)
    """
    ds = getattr(val_loader, "dataset", None)
    if ds is None or not hasattr(ds, "__len__"):
        return {"queries": 0, "classes": 0, "classes_used": 0, "template_k": int(template_k), "splits": int(num_splits)}

    indices = list(range(len(ds)))
    feats, labels = _embed_indices(model, ds, indices, device=device, batch_size=embed_bs)
    if feats.numel() == 0 or labels.numel() < 2:
        return {"queries": 0, "classes": int(labels.unique().numel()) if labels.numel() else 0, "classes_used": 0,
                "template_k": int(template_k), "splits": int(num_splits)}

    uniq = sorted(labels.unique().tolist())
    C = len(uniq)
    y2c = {int(y): i for i, y in enumerate(uniq)}

    class_indices: List[List[int]] = [[] for _ in range(C)]
    for i in range(labels.numel()):
        class_indices[y2c[int(labels[i].item())]].append(i)

    total_queries = 0
    total_hits = 0
    margins_all: List[float] = []
    gaps_all: List[float] = []
    conf_all: List[int] = []

    pos_scores_all: List[float] = []
    neg_scores_all: List[float] = []

    cls_total = [0] * C
    cls_hit = [0] * C
    cls_conf = [0] * C

    skipped_classes_list: List[int] = []
    avg_q_per_cls_list: List[float] = []

    confuse_pair_cnt = defaultdict(int)

    writer = None
    f_csv = None
    if export_csv_path is not None:
        d = os.path.dirname(export_csv_path)
        os.makedirs(d if d else ".", exist_ok=True)
        f_csv = open(export_csv_path, "w", newline="", encoding="utf-8")
        fieldnames = [
            "split_id", "q_idx", "query",
            "true_label", "true_class_idx",
            "pred_class_idx", "hit",
            "top1_score", "top2_score", "margin",
            "true_score", "best_neg_score", "gap",
            "template_k", "agg",
        ]
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

    img_paths = getattr(ds, "image_paths", None)

    for s in range(int(num_splits)):
        rng = random.Random(int(base_seed) + s)

        gallery_idx: List[int] = []
        gallery_c: List[int] = []
        query_idx: List[int] = []
        query_c: List[int] = []

        skipped = 0
        used = 0
        q_counts = []

        for c in range(C):
            idxs = class_indices[c]
            n = len(idxs)

            # Need at least templates + min_q_per_cls
            if n < (min_q_per_cls + 1):
                skipped += 1
                continue

            idxs = idxs[:]  # copy
            rng.shuffle(idxs)

            # Ensure at least min_q_per_cls left for query
            k_eff = min(int(template_k), n - int(min_q_per_cls))
            if k_eff <= 0:
                skipped += 1
                continue

            tpl = idxs[:k_eff]
            qry = idxs[k_eff:]
            if len(qry) < min_q_per_cls:
                skipped += 1
                continue

            used += 1
            q_counts.append(len(qry))
            gallery_idx.extend(tpl)
            gallery_c.extend([c] * len(tpl))
            query_idx.extend(qry)
            query_c.extend([c] * len(qry))

        skipped_classes_list.append(skipped)
        if q_counts:
            avg_q_per_cls_list.append(sum(q_counts) / len(q_counts))

        Q = len(query_idx)
        G = len(gallery_idx)
        if Q == 0 or G == 0 or used == 0:
            continue

        qf = feats[query_idx]     # [Q, D]
        gf = feats[gallery_idx]   # [G, D]
        sim_qg = qf @ gf.t()      # [Q, G]
        gallery_c_t = torch.tensor(gallery_c, dtype=torch.long)  # [G]

        scores = torch.full((Q, C), -1e9, dtype=torch.float32)
        for c in range(C):
            cols = (gallery_c_t == c).nonzero(as_tuple=False).view(-1)
            if cols.numel() == 0:
                continue
            ss = sim_qg[:, cols]
            if agg == "max":
                scores[:, c] = ss.max(dim=1).values
            elif agg == "mean":
                scores[:, c] = ss.mean(dim=1)
            else:
                raise ValueError(f"Unknown agg={agg}, expected 'max' or 'mean'.")

        top2_vals, top2_idx = torch.topk(scores, k=2, dim=1)
        pred_c = top2_idx[:, 0]
        top1 = top2_vals[:, 0]
        top2 = top2_vals[:, 1]

        true_c = torch.tensor(query_c, dtype=torch.long)
        hit = (pred_c == true_c).to(torch.long)

        true_score = scores[torch.arange(Q), true_c]
        masked = scores.clone()
        masked[torch.arange(Q), true_c] = -1e9
        best_neg = masked.max(dim=1).values

        gap = (true_score - best_neg)
        conf = (gap < 0).to(torch.long)
        margin = (top1 - top2)

        total_queries += Q
        total_hits += int(hit.sum().item())

        margins_all.extend(margin.tolist())
        gaps_all.extend(gap.tolist())
        conf_all.extend(conf.tolist())

        if compute_thr_diag:
            pos_scores_all.extend(true_score.tolist())
            neg_scores_all.extend(best_neg.tolist())

        for qi in range(Q):
            c = int(true_c[qi].item())
            cls_total[c] += 1
            cls_hit[c] += int(hit[qi].item())
            cls_conf[c] += int(conf[qi].item())
            if int(hit[qi].item()) == 0:
                confuse_pair_cnt[(int(true_c[qi].item()), int(pred_c[qi].item()))] += 1

        if writer is not None:
            for qi in range(Q):
                global_i = query_idx[qi]
                qname = os.path.basename(img_paths[global_i]) if isinstance(img_paths, list) else str(global_i)
                tl = int(labels[global_i].item())
                writer.writerow({
                    "split_id": int(s),
                    "q_idx": int(global_i),
                    "query": qname,
                    "true_label": tl,
                    "true_class_idx": int(true_c[qi].item()),
                    "pred_class_idx": int(pred_c[qi].item()),
                    "hit": int(hit[qi].item()),
                    "top1_score": float(top1[qi].item()),
                    "top2_score": float(top2[qi].item()),
                    "margin": float(margin[qi].item()),
                    "true_score": float(true_score[qi].item()),
                    "best_neg_score": float(best_neg[qi].item()),
                    "gap": float(gap[qi].item()),
                    "template_k": int(template_k),
                    "agg": agg,
                })

    if f_csv is not None:
        try:
            f_csv.close()
        except Exception:
            pass

    if total_queries <= 0:
        return {
            "N": int(labels.numel()),
            "classes": int(C),
            "classes_used": 0,
            "queries": 0,
            "template_k": int(template_k),
            "splits": int(num_splits),
            "min_q_per_cls": int(min_q_per_cls),
            "tpl_acc_mean": 0.0,
            "tpl_acc_min": 0.0,
            "tpl_acc_p05_cls": 0.0,
            "margin_mean": 0.0,
            "margin_p05": 0.0,
            "gap_mean": 0.0,
            "gap_p05": 0.0,
            "confuse_rate": 0.0,
            "confuse_rate_max_cls": 0.0,
            "confuse_rate_p05_cls": 0.0,
            "avg_q_per_cls": 0.0,
            "skipped_classes_mean": float(sum(skipped_classes_list) / max(1, len(skipped_classes_list))) if skipped_classes_list else 0.0,
            "pos_p05": 0.0,
            "neg_p95": 0.0,
            "neg_p99": 0.0,
            "tpr_at_far5": 0.0,
            "tpr_at_far1": 0.0,
            "worst_confuse_pairs": "",
        }

    tpl_acc_mean = float(total_hits / max(1, total_queries))

    cls_acc = []
    cls_conf_rate = []
    classes_used_final = 0
    for c in range(C):
        if cls_total[c] > 0:
            classes_used_final += 1
            cls_acc.append(cls_hit[c] / cls_total[c])
            cls_conf_rate.append(cls_conf[c] / cls_total[c])

    tpl_acc_min = float(min(cls_acc)) if cls_acc else 0.0
    tpl_acc_p05_cls = _percentile(cls_acc, 0.05) if cls_acc else 0.0

    margin_mean = float(sum(margins_all) / len(margins_all)) if margins_all else 0.0
    margin_p05 = _percentile(margins_all, 0.05) if margins_all else 0.0

    gap_mean = float(sum(gaps_all) / len(gaps_all)) if gaps_all else 0.0
    gap_p05 = _percentile(gaps_all, 0.05) if gaps_all else 0.0

    confuse_rate = float(sum(conf_all) / max(1, len(conf_all)))
    confuse_rate_max_cls = float(max(cls_conf_rate)) if cls_conf_rate else 0.0
    confuse_rate_p05_cls = _percentile(cls_conf_rate, 0.05) if cls_conf_rate else 0.0

    skipped_mean = float(sum(skipped_classes_list) / max(1, len(skipped_classes_list))) if skipped_classes_list else 0.0
    avg_q_per_cls = float(sum(avg_q_per_cls_list) / max(1, len(avg_q_per_cls_list))) if avg_q_per_cls_list else 0.0

    # threshold diagnostics (optional)
    pos_p05 = neg_p95 = neg_p99 = tpr_at_far5 = tpr_at_far1 = 0.0
    if compute_thr_diag and pos_scores_all and neg_scores_all:
        pos_p05 = _percentile(pos_scores_all, 0.05)
        neg_p95 = _percentile(neg_scores_all, 0.95)
        neg_p99 = _percentile(neg_scores_all, 0.99)
        tpr_at_far5 = float(sum(1 for s in pos_scores_all if s > neg_p95) / max(1, len(pos_scores_all)))
        tpr_at_far1 = float(sum(1 for s in pos_scores_all if s > neg_p99) / max(1, len(pos_scores_all)))

    # worst confusion pairs
    worst_pairs = sorted(confuse_pair_cnt.items(), key=lambda kv: kv[1], reverse=True)[:max(0, int(export_confuse_topn))]
    worst_confuse_pairs = ""
    if worst_pairs:
        worst_confuse_pairs = " | ".join([f"{tp[0]}->{tp[1]}:{cnt}" for (tp, cnt) in worst_pairs])

    return {
        "N": int(labels.numel()),
        "classes": int(C),
        "classes_used": int(classes_used_final),
        "queries": int(total_queries),
        "template_k": int(template_k),
        "splits": int(num_splits),
        "min_q_per_cls": int(min_q_per_cls),

        "tpl_acc_mean": float(tpl_acc_mean),
        "tpl_acc_min": float(tpl_acc_min),
        "tpl_acc_p05_cls": float(tpl_acc_p05_cls),

        "margin_mean": float(margin_mean),
        "margin_p05": float(margin_p05),

        "gap_mean": float(gap_mean),
        "gap_p05": float(gap_p05),

        "confuse_rate": float(confuse_rate),
        "confuse_rate_max_cls": float(confuse_rate_max_cls),
        "confuse_rate_p05_cls": float(confuse_rate_p05_cls),

        "avg_q_per_cls": float(avg_q_per_cls),
        "skipped_classes_mean": float(skipped_mean),

        "pos_p05": float(pos_p05),
        "neg_p95": float(neg_p95),
        "neg_p99": float(neg_p99),
        "tpr_at_far5": float(tpr_at_far5),
        "tpr_at_far1": float(tpr_at_far1),

        "worst_confuse_pairs": worst_confuse_pairs,
    }


# ---------------------------
# Best / collapse criteria (ONLINE-K only)
# ---------------------------
def _is_better_online(snapshot: Dict[str, float], best: Dict[str, float]) -> bool:
    """
    Best selection (ONLINE-K only):
      1) tpl_acc_min           higher better (最差类优先拉满)
      2) tpl_acc_mean          higher better
      3) gap_p05               higher better (5% 最差 separation)
      4) -confuse_rate_max_cls higher better
      5) margin_p05            higher better
    """
    a = dict(snapshot)
    b = dict(best)
    a["neg_confuse_rate_max_cls"] = -float(snapshot.get("confuse_rate_max_cls", 0.0))
    b["neg_confuse_rate_max_cls"] = -float(best.get("confuse_rate_max_cls", 0.0))

    keys = ["tpl_acc_min", "tpl_acc_mean", "gap_p05", "neg_confuse_rate_max_cls", "margin_p05"]
    for k in keys:
        av = float(a.get(k, -1e9))
        bv = float(b.get(k, -1e9))
        if av > bv + 1e-12:
            return True
        if av < bv - 1e-12:
            return False
    return False


def _is_collapse_online(val_online: Dict[str, float]) -> bool:
    """
    Simple stability guard:
    - if acc_mean is extremely low OR gap_p05 very negative OR margin_p05 ~ 0
    """
    acc_mean = float(val_online.get("tpl_acc_mean", 0.0))
    gap_p05 = float(val_online.get("gap_p05", 0.0))
    margin_p05 = float(val_online.get("margin_p05", 0.0))
    if acc_mean < 0.20:
        return True
    if gap_p05 < -0.10:
        return True
    if margin_p05 < 1e-6:
        return True
    return False


# ---------------------------
# Main train (PK only)
# ---------------------------
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Paths ---
    ckpt_dir = "./checkpoints_v2"
    ckpt_path = os.path.join(ckpt_dir, "latest.pth")
    best_path = os.path.join(ckpt_dir, "best_model.pth")
    details_dir = os.path.join(ckpt_dir, "val_details")

    train_path = os.path.expanduser("/home/zhangjunjie/data_generation/output_dataset/train")
    val_path = os.path.expanduser("/home/zhangjunjie/stamp_cmp/stamps")

    # --- Data ---
    batch_size = 32
    img_size = 518
    num_workers = 4
    key_level = "bg_s_pad"

    # PK sampler config
    pk_P, pk_K = 8, 4
    pk_epoch_len = 800
    eff_bs = pk_P * pk_K

    # --- Training ---
    max_epochs = 200
    patience_global = 50  # 你要“全对”，就别太早停

    # LRs
    lr_head = 2e-4
    lr_backbone = 2e-6
    weight_decay = 1e-2

    # SupCon
    supcon_tau = 0.10

    # Stability
    clip_norm = 0.3
    max_bad_batches_per_epoch = 10

    # AMP
    use_amp = True
    prefer_bf16 = True
    disable_amp_first_epoch = False
    scaler_init_scale = 2 ** 8
    scaler_growth_interval = 2000
    scaler_backoff_factor = 0.5

    # patch_embed
    freeze_patch_embed = True
    lr_patch_embed_mult = 0.1

    # backbone warmup: 前 N 个 epoch 冻住 backbone 让 projector 先“贴合 val”
    backbone_freeze_warmup_epochs = 1

    # -------------------------
    # Validation configuration
    # -------------------------
    val_embed_bs = 64
    val_template_agg = "max"
    val_split_seed = 123

    # (A) ONLINE-K: THIS is your business KPI proxy, used for best selection.
    VAL_ONLINE_K = 3                  # <<< set to your real online K
    VAL_ONLINE_SPLITS = 20
    MIN_Q_PER_CLS = 2                 # <<< IMPORTANT: prevents trivial leave-one-out
    export_online_csv_each_epoch = True

    # (B) DIAG-K: log only, for "upper bound" + confusion debugging.
    DIAG_K = 999
    DIAG_SPLITS = 20
    export_diag_csv_each_epoch = True
    export_confuse_topn = 10

    logger = get_logger(ckpt_dir)

    # Build loaders
    train_loader, val_loader, num_classes = _build_dataloaders_safe(
        train_path=train_path,
        val_path=val_path,
        batch_size=batch_size,
        img_size=img_size,
        mode="pk",
        key_level=key_level,
        num_workers=num_workers,
        seed=123,
        pk_P=pk_P,
        pk_K=pk_K,
        pk_epoch_len=pk_epoch_len,
        return_val_index=True,
    )
    logger.info(f">>> num_classes={num_classes}")
    logger.info(f">>> TRAIN mode=pk-only (P={pk_P}, K={pk_K}, epoch_len={pk_epoch_len}, eff_bs={eff_bs})")
    logger.info(f">>> VAL ONLINE: K={VAL_ONLINE_K}, S={VAL_ONLINE_SPLITS}, min_q_per_cls={MIN_Q_PER_CLS}, agg={val_template_agg}")
    logger.info(f">>> VAL DIAG  : K={DIAG_K} (cap to n-min_q), S={DIAG_SPLITS}, min_q_per_cls={MIN_Q_PER_CLS}, agg={val_template_agg}")

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

    # Resume
    start_epoch = 0
    global_no_improve = 0
    best_snapshot_online: Dict[str, float] = {
        "tpl_acc_min": -1.0,
        "tpl_acc_mean": -1.0,
        "gap_p05": -1e9,
        "confuse_rate_max_cls": 1.0,
        "margin_p05": -1e9,
    }
    best_state_dict = copy.deepcopy(model.state_dict())
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    best_scaler_state = copy.deepcopy(scaler.state_dict()) if scaler is not None else None

    resume = False  # <<< set True to resume
    if resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e:
            logger.info(f">>> optimizer not loaded: {e}")
        tag_optimizer_groups(optimizer, model)

        if scaler is not None and ckpt.get("scaler") is not None:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception as e:
                logger.info(f">>> scaler not loaded: {e}")

        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_snapshot_online = ckpt.get("best_snapshot_online", best_snapshot_online) if isinstance(ckpt.get("best_snapshot_online", None), dict) else best_snapshot_online
        global_no_improve = int(ckpt.get("global_no_improve", global_no_improve))
        logger.info(f">>> resumed: start_epoch={start_epoch}, global_no_improve={global_no_improve}, best_snapshot_online={best_snapshot_online}")

    os.makedirs(details_dir, exist_ok=True)

    for epoch in range(start_epoch, max_epochs):
        # backbone warmup freeze
        if epoch < int(backbone_freeze_warmup_epochs):
            set_backbone_trainable(model, False)
            warm = "FREEZE_BB"
        else:
            set_backbone_trainable(model, True)
            warm = "UNFREEZE_BB"

        model.train()
        epoch_use_amp = (device.type == "cuda") and use_amp and not (disable_amp_first_epoch and epoch == 0)

        t_loss, steps_done = 0.0, 0
        skipped_no_pos, bad_batches = 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train:PK {warm}]", leave=False)
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

            # loss in fp32
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
                    f"gnorm={float(total_norm):.3e} scale={(scaler.get_scale() if scaler is not None else 'NA')} bad={bad_batches}"
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

        # -------------------------
        # Validate ONLINE-K (best KPI)
        # -------------------------
        export_online_csv = None
        if export_online_csv_each_epoch:
            export_online_csv = os.path.join(details_dir, f"ONLINE_K{VAL_ONLINE_K}_{val_template_agg}_S{VAL_ONLINE_SPLITS}_Qmin{MIN_Q_PER_CLS}_e{epoch:04d}.csv")

        val_online = validate_k_template_multisplit(
            model,
            val_loader,
            device=device,
            template_k=VAL_ONLINE_K,
            agg=val_template_agg,
            embed_bs=val_embed_bs,
            base_seed=val_split_seed,
            num_splits=VAL_ONLINE_SPLITS,
            min_q_per_cls=MIN_Q_PER_CLS,
            export_csv_path=export_online_csv,
            export_confuse_topn=export_confuse_topn,
            compute_thr_diag=False,
        )

        snapshot_online = {
            "tpl_acc_min": float(val_online.get("tpl_acc_min", 0.0)),
            "tpl_acc_mean": float(val_online.get("tpl_acc_mean", 0.0)),
            "gap_p05": float(val_online.get("gap_p05", 0.0)),
            "confuse_rate_max_cls": float(val_online.get("confuse_rate_max_cls", 0.0)),
            "margin_p05": float(val_online.get("margin_p05", 0.0)),
        }

        # -------------------------
        # Validate DIAG-K (log only)
        # -------------------------
        export_diag_csv = None
        if export_diag_csv_each_epoch:
            export_diag_csv = os.path.join(details_dir, f"DIAG_K{DIAG_K}_{val_template_agg}_S{DIAG_SPLITS}_Qmin{MIN_Q_PER_CLS}_e{epoch:04d}.csv")

        val_diag = validate_k_template_multisplit(
            model,
            val_loader,
            device=device,
            template_k=DIAG_K,
            agg=val_template_agg,
            embed_bs=val_embed_bs,
            base_seed=val_split_seed,
            num_splits=DIAG_SPLITS,
            min_q_per_cls=MIN_Q_PER_CLS,
            export_csv_path=export_diag_csv,
            export_confuse_topn=export_confuse_topn,
            compute_thr_diag=True,
        )

        # -------------------------
        # Log
        # -------------------------
        logger.info(
            f"Epoch {epoch}: TrainLoss={avg_loss:.4f} | TRAIN=PKOnly({warm}) | "
            f"ONLINE(K={VAL_ONLINE_K}, S={VAL_ONLINE_SPLITS}, Qmin={MIN_Q_PER_CLS}, agg={val_template_agg}, "
            f"Q={val_online.get('queries',0)}, used={val_online.get('classes_used',0)}/{val_online.get('classes',0)}, "
            f"avgQ/cls={val_online.get('avg_q_per_cls',0.0):.2f}) "
            f"ACC(mean={val_online.get('tpl_acc_mean',0.0):.4f}, min_cls={val_online.get('tpl_acc_min',0.0):.4f}, p05_cls={val_online.get('tpl_acc_p05_cls',0.0):.4f}) "
            f"| GAP(p05={val_online.get('gap_p05',0.0):.6f}, mean={val_online.get('gap_mean',0.0):.6f}) "
            f"| MARGIN(p05={val_online.get('margin_p05',0.0):.6f}, mean={val_online.get('margin_mean',0.0):.6f}) "
            f"| CONFUSE(max_cls={val_online.get('confuse_rate_max_cls',0.0):.4f}) "
            f"| worst_pairs={val_online.get('worst_confuse_pairs','')} "
            f"{('| export=' + export_online_csv) if export_online_csv else ''} "
            f"|| DIAG(K={DIAG_K}, S={DIAG_SPLITS}, Qmin={MIN_Q_PER_CLS}, "
            f"Q={val_diag.get('queries',0)}, used={val_diag.get('classes_used',0)}/{val_diag.get('classes',0)}, "
            f"avgQ/cls={val_diag.get('avg_q_per_cls',0.0):.2f}) "
            f"ACC(mean={val_diag.get('tpl_acc_mean',0.0):.4f}, min_cls={val_diag.get('tpl_acc_min',0.0):.4f}) "
            f"| THR(pos_p05={val_diag.get('pos_p05',0.0):.4f}, neg_p95={val_diag.get('neg_p95',0.0):.4f}, neg_p99={val_diag.get('neg_p99',0.0):.4f}, "
            f"tpr@FAR5={val_diag.get('tpr_at_far5',0.0):.4f}, tpr@FAR1={val_diag.get('tpr_at_far1',0.0):.4f}) "
            f"{('| export=' + export_diag_csv) if export_diag_csv else ''}"
        )

        # Save latest checkpoint
        latest = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "amp_mode": amp_mode,

            "val_online": val_online,
            "val_diag": val_diag,
            "best_snapshot_online": best_snapshot_online,
            "global_no_improve": global_no_improve,

            "paths": {"train_path": train_path, "val_path": val_path},
            "pk": {"P": pk_P, "K": pk_K, "epoch_len": pk_epoch_len},
            "lrs": {"lr_backbone": lr_backbone, "lr_head": lr_head, "weight_decay": weight_decay},
            "supcon_tau": supcon_tau,

            "val_cfg": {
                "agg": val_template_agg,
                "val_embed_bs": val_embed_bs,
                "val_split_seed": val_split_seed,
                "online": {"K": VAL_ONLINE_K, "splits": VAL_ONLINE_SPLITS, "min_q_per_cls": MIN_Q_PER_CLS},
                "diag": {"K": DIAG_K, "splits": DIAG_SPLITS, "min_q_per_cls": MIN_Q_PER_CLS},
            },
            "freeze_patch_embed": freeze_patch_embed,
            "backbone_freeze_warmup_epochs": backbone_freeze_warmup_epochs,
        }
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(latest, ckpt_path)

        # collapse => rollback + lr drop
        if _is_collapse_online(val_online):
            logger.info(
                f">>> COLLAPSE(ONLINE) detected! acc_mean={val_online.get('tpl_acc_mean',0.0):.4f}, "
                f"gap_p05={val_online.get('gap_p05',0.0):.6f}, margin_p05={val_online.get('margin_p05',0.0):.6f}"
            )
            model.load_state_dict(best_state_dict, strict=True)
            try:
                optimizer.load_state_dict(best_optimizer_state)
            except Exception as e:
                logger.info(f">>> optimizer rollback failed (rebuild): {e}")
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
                except Exception as e:
                    logger.info(f">>> scaler rollback failed: {e}")

            # LR drop
            lr_backbone = max(lr_backbone * 0.5, 1e-8)
            lr_head = max(lr_head * 0.5, 1e-7)
            optimizer_set_lrs(optimizer, lr_backbone, lr_head, (0.0 if freeze_patch_embed else lr_patch_embed_mult))
            logger.info(f">>> rollback done, LR -> bb={lr_backbone:.2e}, hd={lr_head:.2e}")

            global_no_improve += 1
            if global_no_improve >= patience_global:
                logger.info(f">>> early stop (global_no_improve={global_no_improve})")
                break

            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        # best logic (ONLINE only)
        if _is_better_online(snapshot_online, best_snapshot_online):
            best_snapshot_online = dict(snapshot_online)
            best_state_dict = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            best_scaler_state = copy.deepcopy(scaler.state_dict()) if scaler is not None else None

            global_no_improve = 0
            best_ckpt = dict(latest)
            best_ckpt["best_snapshot_online"] = best_snapshot_online
            torch.save(best_ckpt, best_path)
            logger.info(f"✨ New best(ONLINE) saved: best_snapshot_online={best_snapshot_online}")
        else:
            global_no_improve += 1
            logger.info(f">>> no improve(ONLINE): global_no_improve={global_no_improve}/{patience_global} best={best_snapshot_online}")

        if global_no_improve >= patience_global:
            logger.info(f">>> early stop (global_no_improve={global_no_improve})")
            break

        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    train()
