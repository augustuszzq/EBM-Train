#!/usr/bin/env python3
"""ImageNet-32 benchmark metadata, loading, and deterministic materialization."""

from __future__ import annotations

import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch as t

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    import datasets as hf_datasets
except Exception:  # pragma: no cover
    hf_datasets = None


IMAGENET32_HF_REPO = "ChocolateDave/imagenet-32"
IMAGENET32_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
}


@dataclass(frozen=True)
class BenchmarkDataSpec:
    name: str
    conditional: bool
    num_classes: int
    image_size: int
    train_split: str
    val_split: str


class MaterializedImageNet32Dataset(t.utils.data.Dataset):
    def __init__(self, split_dir: str | Path):
        self.split_dir = Path(split_dir)
        self.images_root = self.split_dir / "images"
        labels_csv = self.split_dir / "labels.csv"
        if not labels_csv.exists():
            raise RuntimeError(f"missing labels.csv in materialized ImageNet-32 split: {self.split_dir}")
        self.rows: List[Dict[str, str]] = []
        with labels_csv.open("r", newline="") as f:
            for row in csv.DictReader(f):
                self.rows.append(row)
        if not self.rows:
            raise RuntimeError(f"empty materialized ImageNet-32 split: {self.split_dir}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        if Image is None:
            raise RuntimeError("Pillow is required to read materialized ImageNet-32 samples")
        row = self.rows[int(index)]
        image_path = self.images_root / str(row["relpath"])
        image = Image.open(image_path).convert("RGB")
        arr = t.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        arr = (arr * 2.0 - 1.0).clamp(-1.0, 1.0)
        return arr, int(row["label"])


def get_benchmark_data_spec(name: str) -> BenchmarkDataSpec:
    key = str(name).strip().lower()
    if key == "imagenet32":
        return BenchmarkDataSpec(
            name="imagenet32",
            conditional=True,
            num_classes=1000,
            image_size=32,
            train_split="train",
            val_split="val",
        )
    if key == "cifar10":
        return BenchmarkDataSpec(
            name="cifar10",
            conditional=False,
            num_classes=10,
            image_size=32,
            train_split="train",
            val_split="test",
        )
    if key == "celeba64":
        return BenchmarkDataSpec(
            name="celeba64",
            conditional=False,
            num_classes=0,
            image_size=64,
            train_split="train",
            val_split="valid",
        )
    if key in {"celebahq256", "celeba_hq_256", "celeba-hq-256", "celeba256"}:
        return BenchmarkDataSpec(
            name="celebahq256",
            conditional=False,
            num_classes=0,
            image_size=256,
            train_split="train",
            val_split="validation",
        )
    if key in {"celebahq256_latent", "celeba_hq_256_latent", "celeba-hq-256-latent", "celeba256_latent"}:
        return BenchmarkDataSpec(
            name="celebahq256_latent",
            conditional=False,
            num_classes=0,
            image_size=32,
            train_split="train",
            val_split="validation",
        )
    raise ValueError(f"unsupported benchmark name: {name}")


def load_imagenet32_hf(split: str, cache_dir: str = "", streaming: bool = False):
    if hf_datasets is None:
        raise RuntimeError(
            "Hugging Face datasets is required to load %s; install `datasets` or materialize a local cache first"
            % IMAGENET32_HF_REPO
        )
    split_key = str(split).strip().lower()
    normalized_split = IMAGENET32_SPLIT_ALIASES.get(split_key, split)
    kwargs = {"split": normalized_split, "streaming": bool(streaming)}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return hf_datasets.load_dataset(IMAGENET32_HF_REPO, **kwargs)


def _save_rows(rows: Iterable[Dict], split_dir: Path) -> int:
    if Image is None:
        raise RuntimeError("Pillow is required to materialize ImageNet-32 image files")
    images_root = split_dir / "images"
    labels_csv = split_dir / "labels.csv"
    count = 0
    with labels_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "label", "relpath"])
        writer.writeheader()
        for idx, row in enumerate(rows):
            label = int(row["label"])
            rel = Path(str(label)) / f"{idx:08d}.png"
            out_path = images_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            image = row["image"]
            if not hasattr(image, "save"):
                raise RuntimeError("materialize_imagenet32 expects row['image'] to be a PIL-like image")
            image.save(out_path)
            writer.writerow({"index": idx, "label": label, "relpath": str(rel)})
            count += 1
    return count


def materialize_imagenet32(
    out_root: str | Path,
    split: str,
    cache_dir: str = "",
    streaming: bool = False,
    max_examples: int = 0,
) -> Dict[str, object]:
    spec = get_benchmark_data_spec("imagenet32")
    split_dir = Path(out_root) / str(split)
    success_path = split_dir / "_SUCCESS.json"
    if success_path.exists():
        payload = json.loads(success_path.read_text())
        payload["status"] = "cached"
        return payload

    split_dir.mkdir(parents=True, exist_ok=True)
    rows = load_imagenet32_hf(split=split, cache_dir=cache_dir, streaming=streaming)
    if int(max_examples) > 0:
        rows = itertools.islice(rows, int(max_examples))
    count = _save_rows(rows, split_dir)
    payload = {
        "benchmark": spec.name,
        "repo": IMAGENET32_HF_REPO,
        "split": split,
        "count": count,
        "max_examples": int(max_examples),
        "status": "written",
    }
    success_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload
