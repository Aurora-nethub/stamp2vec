#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
dataset.py (drop-in replacement)

- Train: returns (x, y)
- Val: can return (x, y, idx) when return_val_index=True
  so train.py can do repeated gallery/probe evaluation robustly.

Assumes naming:
  <seal_id>__{bg_type}__{bg_idx:04d}__s{sample:03d}__pad{pad:02d}__v{clean|def}.jpg
  (tolerates vdefect too)
"""

import os
import re
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Iterator

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler
from torchvision import transforms
import torchvision.transforms.functional as TF

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

_FNAME_RE = re.compile(
    r"^(?P<seal_id>.+?)__"
    r"(?P<bg_type>scan|photo)__"
    r"(?P<bg_idx>\d+)__"
    r"s(?P<s>\d+)__"
    r"pad(?P<pad>\d+)__"
    r"v(?P<variant>clean|def|defect)\."
    r"(?P<ext>[A-Za-z0-9]+)$"
)


# ---------------------------
# Transforms
# ---------------------------
class SquarePad:
    def __init__(self, fill: int = 255):
        self.fill = int(fill)

    def __call__(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        m = max(w, h)
        left = (m - w) // 2
        top = (m - h) // 2
        right = m - w - left
        bottom = m - h - top
        return TF.pad(image, (left, top, right, bottom), fill=self.fill, padding_mode="constant")


def build_default_transform(img_size: int) -> transforms.Compose:
    img_size = int(img_size)
    return transforms.Compose([
        SquarePad(fill=255),
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------
# Filename parsing
# ---------------------------
def parse_meta_from_filename(filename: str) -> Optional[dict]:
    base = os.path.basename(filename)
    m = _FNAME_RE.match(base)
    if not m:
        return None
    v = m.group("variant")
    if v == "defect":
        v = "def"
    return {
        "bg_type": m.group("bg_type"),
        "bg_idx": int(m.group("bg_idx")),
        "s": int(m.group("s")),
        "pad": int(m.group("pad")),
        "variant": v,  # "clean" | "def"
    }


def parse_group_key(filename: str, key_level: str) -> Optional[Tuple]:
    meta = parse_meta_from_filename(filename)
    if meta is None:
        return None

    if key_level == "bg_s":
        return (meta["bg_type"], meta["bg_idx"], meta["s"])
    if key_level == "bg_s_pad":
        return (meta["bg_type"], meta["bg_idx"], meta["s"], meta["pad"])
    if key_level == "bg_only":
        return (meta["bg_type"], meta["bg_idx"])
    raise ValueError(f"Unknown key_level: {key_level}")


# ---------------------------
# Dataset
# ---------------------------
class SealDataset(Dataset):
    def __init__(self, root_dir: str, img_size: int = 518, key_level: str = "bg_s", return_index: bool = False):
        self.root_dir = os.path.expanduser(root_dir)
        self.img_size = int(img_size)
        self.key_level = str(key_level)
        self.return_index = bool(return_index)

        if not os.path.isdir(self.root_dir):
            raise RuntimeError(f"Dataset root not found: {self.root_dir}")

        self.classes = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])
        if not self.classes:
            raise RuntimeError(f"No class folders found under: {self.root_dir}")

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        self.image_paths: List[str] = []
        self.labels: List[int] = []
        self.group_keys: List[Optional[Tuple]] = []
        self.variants: List[Optional[str]] = []

        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            for fn in os.listdir(cls_dir):
                ext = os.path.splitext(fn)[1].lower()
                if ext not in _IMG_EXTS:
                    continue
                p = os.path.join(cls_dir, fn)
                self.image_paths.append(p)
                y = self.class_to_idx[cls_name]
                self.labels.append(y)

                meta = parse_meta_from_filename(fn)
                self.variants.append(meta["variant"] if meta else None)
                self.group_keys.append(parse_group_key(fn, self.key_level))

        if not self.image_paths:
            raise RuntimeError(f"No images found under: {self.root_dir}")

        self.indices_by_label: Dict[int, List[int]] = defaultdict(list)
        for i, y in enumerate(self.labels):
            self.indices_by_label[y].append(i)

        self.transform = build_default_transform(self.img_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.return_index:
            return x, y, idx
        return x, y


# ---------------------------
# PK Sampler
# ---------------------------
class PKSampler(Sampler[List[int]]):
    def __init__(
        self,
        dataset: SealDataset,
        P: int,
        K: int,
        seed: int = 123,
        epoch_len: Optional[int] = None,
        max_retry_per_batch: int = 50,
        allow_repeat_if_insufficient: bool = True,
        prefer_diff_groups: bool = True,
        prefer_mix_variants: bool = True,
    ):
        self.dataset = dataset
        self.P = int(P)
        self.K = int(K)
        self.seed = int(seed)
        self.max_retry_per_batch = int(max_retry_per_batch)
        self.allow_repeat_if_insufficient = bool(allow_repeat_if_insufficient)
        self.prefer_diff_groups = bool(prefer_diff_groups)
        self.prefer_mix_variants = bool(prefer_mix_variants)

        if self.P <= 0 or self.K <= 0:
            raise ValueError("P and K must be positive.")

        self.indices_by_label: Dict[int, List[int]] = defaultdict(list)
        self.indices_by_label_by_group: Dict[int, Dict[Tuple, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.indices_by_label_by_variant: Dict[int, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))

        for i, y in enumerate(dataset.labels):
            self.indices_by_label[y].append(i)

            gk = dataset.group_keys[i]
            if gk is not None:
                self.indices_by_label_by_group[y][gk].append(i)

            v = dataset.variants[i]
            if v in ("clean", "def"):
                self.indices_by_label_by_variant[y][v].append(i)

        self.labels = sorted(self.indices_by_label.keys())
        if len(self.labels) < self.P:
            raise RuntimeError(f"Not enough classes for P={self.P}. Have {len(self.labels)}.")

        if epoch_len is not None:
            self._len = max(1, int(epoch_len))
        else:
            self._len = max(1, len(dataset) // (self.P * self.K))

        self._epoch = 0
        self._auto_epoch_counter = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[List[int]]:
        self._auto_epoch_counter += 1
        epoch_for_seed = self._epoch + (self._auto_epoch_counter - 1)
        rng = random.Random(self.seed + 1000003 * epoch_for_seed)

        labels = self.labels[:]
        rng.shuffle(labels)
        ptr = 0

        for _ in range(self._len):
            batch: Optional[List[int]] = None

            for _retry in range(self.max_retry_per_batch):
                if ptr + self.P > len(labels):
                    rng.shuffle(labels)
                    ptr = 0
                chosen = labels[ptr:ptr + self.P]
                ptr += self.P

                candidate: List[int] = []
                ok = True
                for y in chosen:
                    picked = self._pick_k(rng, y, self.K)
                    if picked is None or len(picked) != self.K:
                        ok = False
                        break
                    candidate.extend(picked)

                if ok and len(candidate) == self.P * self.K:
                    batch = candidate
                    break

            if batch is None:
                chosen = rng.sample(self.labels, self.P)
                batch = []
                for y in chosen:
                    batch.extend(self._pick_any_k(rng, y, self.K))
                while len(batch) < self.P * self.K:
                    batch.append(rng.randrange(len(self.dataset)))
                batch = batch[: self.P * self.K]

            yield batch

    def _pick_any_k(self, rng: random.Random, y: int, K: int) -> List[int]:
        idxs = self.indices_by_label[y]
        if len(idxs) >= K:
            return rng.sample(idxs, K)
        if not self.allow_repeat_if_insufficient:
            return []
        return [rng.choice(idxs) for _ in range(K)]

    def _pick_k(self, rng: random.Random, y: int, K: int) -> Optional[List[int]]:
        all_idxs = self.indices_by_label[y]
        if not all_idxs:
            return None

        picked: List[int] = []
        used = set()

        if self.prefer_mix_variants and K >= 2:
            vmap = self.indices_by_label_by_variant.get(y, {})
            clean = vmap.get("clean", [])
            defe = vmap.get("def", [])
            if clean and defe:
                a = rng.choice(clean)
                b = rng.choice(defe)
                if a == b and len(defe) > 1:
                    b = rng.choice([i for i in defe if i != a])
                picked.extend([a, b])
                used.update(picked)

        if self.prefer_diff_groups:
            groups = self.indices_by_label_by_group.get(y, {})
            gks = list(groups.keys())
            if gks:
                rng.shuffle(gks)
                for gk in gks:
                    if len(picked) >= K:
                        break
                    cand = [i for i in groups[gk] if i not in used]
                    if cand:
                        c = rng.choice(cand)
                        picked.append(c)
                        used.add(c)

        if len(picked) < K:
            remaining = [i for i in all_idxs if i not in used]
            rng.shuffle(remaining)
            for idx in remaining:
                if len(picked) >= K:
                    break
                picked.append(idx)
                used.add(idx)

        if len(picked) >= K:
            return picked[:K]

        if not self.allow_repeat_if_insufficient:
            return None

        while len(picked) < K:
            picked.append(rng.choice(all_idxs))
        return picked[:K]


# ---------------------------
# Dataloaders
# ---------------------------
def get_dataloaders(
    train_path: str,
    val_path: str,
    batch_size: int = 32,
    img_size: int = 518,
    mode: str = "pk",          # "pk" | "random"
    key_level: str = "bg_s",
    num_workers: int = 4,
    pk_P: int = 8,
    pk_K: int = 4,
    seed: int = 123,
    pk_epoch_len: Optional[int] = None,
    random_epoch_len: Optional[int] = None,     # cap random steps/epoch (in batches)
    return_val_index: bool = True,
):
    train_ds = SealDataset(train_path, img_size=img_size, key_level=key_level, return_index=False)
    val_ds = SealDataset(val_path, img_size=img_size, key_level=key_level, return_index=return_val_index)

    if mode == "pk":
        P = int(pk_P)
        K = int(pk_K)
        effective_bs = P * K
        if effective_bs != int(batch_size):
            print(f"[INFO] PK uses batch_size=P*K={effective_bs} (your batch_size={batch_size}).")

        sampler = PKSampler(
            dataset=train_ds,
            P=P,
            K=K,
            seed=seed,
            epoch_len=pk_epoch_len,
            allow_repeat_if_insufficient=True,
            prefer_diff_groups=False,
            prefer_mix_variants=True,
        )

        train_loader = DataLoader(
            train_ds,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )

    elif mode == "random":
        if random_epoch_len is not None:
            num_samples = int(random_epoch_len) * int(batch_size)
            g = torch.Generator()
            g.manual_seed(int(seed))

            sampler = RandomSampler(train_ds, replacement=True, num_samples=num_samples, generator=g)
            train_loader = DataLoader(
                train_ds,
                batch_size=int(batch_size),
                sampler=sampler,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
            )
            print(f"[INFO] RANDOM capped: epoch_len={int(random_epoch_len)} batches, num_samples={num_samples}, seed={seed}")
        else:
            train_loader = DataLoader(
                train_ds,
                batch_size=int(batch_size),
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
            )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    val_loader = DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, len(train_ds.classes)
