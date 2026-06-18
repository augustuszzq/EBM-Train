#!/usr/bin/env python3
"""CelebA-HQ 256 dataset helpers compatible with the NVAE LMDB layout."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch as t

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    from torchvision import transforms
except Exception:  # pragma: no cover
    transforms = None

try:
    import lmdb
except Exception:  # pragma: no cover
    lmdb = None

try:
    from imagenet32_data import BenchmarkDataSpec
except Exception:
    from polaris_ebm.scripts.current.imagenet32_data import BenchmarkDataSpec


CELEBAHQ256_ALIASES = {"celebahq256", "celeba_hq_256", "celeba-hq-256", "celeba256"}
CELEBAHQ256_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "validation",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_LMDB_ENV_CACHE = {}


def normalize_celebahq256_split(split: str) -> str:
    return CELEBAHQ256_SPLIT_ALIASES.get(str(split).strip().lower(), str(split).strip().lower())


def get_celebahq256_data_spec() -> BenchmarkDataSpec:
    return BenchmarkDataSpec(
        name="celebahq256",
        conditional=False,
        num_classes=0,
        image_size=256,
        train_split="train",
        val_split="validation",
    )


def _require_pillow() -> None:
    if Image is None:
        raise RuntimeError("Pillow is required to read CelebA-HQ 256 images")


def _require_transforms() -> None:
    if transforms is None:
        raise RuntimeError("torchvision is required for CelebA-HQ 256 transforms")


def make_celebahq256_transform(image_size: int = 256, train: bool = True):
    _require_transforms()
    ops = [transforms.Resize((int(image_size), int(image_size)))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transforms.Compose(ops)


def _list_images(split_root: Path) -> List[Path]:
    return sorted(path for path in split_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _open_readonly_lmdb(path: Path):
    key = str(path.resolve())
    env = _LMDB_ENV_CACHE.get(key)
    if env is not None:
        return env
    env = lmdb.open(
        str(path),
        readonly=True,
        max_readers=32,
        lock=False,
        readahead=False,
        meminit=False,
    )
    _LMDB_ENV_CACHE[key] = env
    return env


class CelebAHQ256ImageFolderDataset(t.utils.data.Dataset):
    """Unconditional CelebA-HQ 256 reader for local image-folder exports.

    The label is always zero so this can flow through training code that expects
    a `(x, y)` pair without making the benchmark conditional.
    """

    def __init__(self, root: str | Path, split: str = "train", image_size: int = 256, transform=None):
        _require_pillow()
        self.root = Path(root)
        self.split = normalize_celebahq256_split(split)
        self.split_root = self.root / self.split if (self.root / self.split).exists() else self.root
        self.transform = transform or make_celebahq256_transform(image_size=image_size, train=self.split == "train")
        self.images = _list_images(self.split_root)
        if not self.images:
            raise RuntimeError(f"no CelebA-HQ 256 image files found under {self.split_root}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = Image.open(self.images[int(index)]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, 0


class CelebAHQ256LMDBDataset(t.utils.data.Dataset):
    """Reader for NVAE-style `celeba-lmdb/{train,validation}.lmdb` files."""

    def __init__(self, root: str | Path, split: str = "train", image_size: int = 256, transform=None):
        _require_pillow()
        if lmdb is None:
            raise RuntimeError("lmdb is required to read NVAE-style CelebA-HQ 256 LMDB files")
        self.root = Path(root)
        self.split = normalize_celebahq256_split(split)
        self.lmdb_path = self.root / f"{self.split}.lmdb"
        if not self.lmdb_path.exists():
            raise RuntimeError(f"missing CelebA-HQ 256 LMDB split: {self.lmdb_path}")
        self.env = _open_readonly_lmdb(self.lmdb_path)
        self.length = int(self.env.stat().get("entries", 0))
        if self.length <= 0:
            raise RuntimeError(f"empty CelebA-HQ 256 LMDB split: {self.lmdb_path}")
        self.transform = transform or make_celebahq256_transform(image_size=image_size, train=self.split == "train")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        with self.env.begin(write=False, buffers=True) as txn:
            payload = txn.get(str(int(index)).encode())
        if payload is None:
            raise IndexError(f"missing CelebA-HQ 256 LMDB key {index}")
        arr = np.frombuffer(payload, dtype=np.uint8)
        size = int(math.sqrt(float(arr.size) / 3.0))
        if size * size * 3 != arr.size:
            raise RuntimeError(f"cannot infer RGB square image shape from LMDB item with {arr.size} bytes")
        arr = arr.reshape(size, size, 3)
        image = Image.fromarray(arr, mode="RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, 0


def build_celebahq256_dataset(root: str | Path, split: str = "train", image_size: int = 256):
    root_path = Path(root)
    normalized_split = normalize_celebahq256_split(split)
    lmdb_candidate = root_path / f"{normalized_split}.lmdb"
    if lmdb_candidate.exists():
        return CelebAHQ256LMDBDataset(root_path, split=normalized_split, image_size=image_size)
    return CelebAHQ256ImageFolderDataset(root_path, split=normalized_split, image_size=image_size)


def _iter_tfrecord_paths(tfr_path: Path, split: str) -> Iterable[Path]:
    split = normalize_celebahq256_split(split)
    num_shards = {"train": 120, "validation": 40}[split]
    for shard in range(num_shards):
        yield tfr_path / split / f"{split}-r08-s-{shard:04d}-of-{num_shards:04d}.tfrecords"


def _raw_tfrecord_image_bytes(value) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if t.is_tensor(value):
        return value.detach().cpu().numpy().tobytes()
    if isinstance(value, np.ndarray):
        return value.tobytes()
    if isinstance(value, list) and len(value) == 1:
        return _raw_tfrecord_image_bytes(value[0])
    raise TypeError(f"unsupported TFRecord image payload type: {type(value)!r}")


def convert_tfrecord_split_to_lmdb(
    tfr_path: str | Path,
    lmdb_root: str | Path,
    split: str,
    max_records: int = 0,
    force: bool = False,
    commit_every: int = 1000,
) -> dict:
    """Convert GLOW/NVAE CelebA-HQ TFRecords into NVAE-style LMDB."""

    if lmdb is None:
        raise RuntimeError("lmdb is required for TFRecord to LMDB conversion")
    try:
        from tfrecord.torch.dataset import TFRecordDataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("tfrecord is required for TFRecord to LMDB conversion; install `tfrecord`") from exc

    split = normalize_celebahq256_split(split)
    tfr_path = Path(tfr_path)
    lmdb_root = Path(lmdb_root)
    target = lmdb_root / f"{split}.lmdb"
    success = lmdb_root / f"{split}._SUCCESS.json"
    if success.exists() and not force:
        payload = json.loads(success.read_text())
        payload["status"] = "cached"
        return payload
    if target.exists() and not force:
        raise RuntimeError(f"{target} already exists without success marker; pass --force to rebuild")
    lmdb_root.mkdir(parents=True, exist_ok=True)
    if force and target.exists():
        import shutil

        shutil.rmtree(target)

    env = lmdb.open(str(target), map_size=int(1e12))
    description = {"shape": "int", "data": "byte", "label": "int"}
    count = 0
    txn = env.begin(write=True)
    try:
        for shard_path in _iter_tfrecord_paths(tfr_path, split):
            if not shard_path.exists():
                raise RuntimeError(f"missing CelebA-HQ TFRecord shard: {shard_path}")
            ds = TFRecordDataset(str(shard_path), None, description)
            for row in t.utils.data.DataLoader(ds, batch_size=1):
                payload = _raw_tfrecord_image_bytes(row["data"][0])
                txn.put(str(count).encode(), payload)
                count += 1
                if int(commit_every) > 0 and count % int(commit_every) == 0:
                    txn.commit()
                    print(f"[LMDB] {split}: wrote {count}", flush=True)
                    txn = env.begin(write=True)
                if int(max_records) > 0 and count >= int(max_records):
                    break
            if int(max_records) > 0 and count >= int(max_records):
                break
        txn.commit()
    except Exception:
        txn.abort()
        raise
    env.close()
    payload = {
        "benchmark": "celebahq256",
        "source": "GLOW/NVAE celeba-tfr",
        "split": split,
        "count": count,
        "lmdb_path": str(target),
        "status": "written",
    }
    success.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def inspect_celebahq256_root(root: str | Path) -> dict:
    root_path = Path(root)
    out = {
        "root": str(root_path),
        "exists": root_path.exists(),
        "lmdb_train": (root_path / "train.lmdb").exists(),
        "lmdb_validation": (root_path / "validation.lmdb").exists(),
        "train_image_count": 0,
        "validation_image_count": 0,
    }
    for split in ("train", "validation"):
        split_root = root_path / split
        if split_root.exists():
            out[f"{split}_image_count"] = len(_list_images(split_root))
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser("Prepare or verify CelebA-HQ 256 data")
    ap.add_argument("--data_root", default="./data/celeba/celeba-lmdb", help="NVAE-style LMDB root or image-folder root")
    ap.add_argument("--tfr_path", default="./data/celeba/celeba-tfr", help="GLOW/NVAE celeba-tfr directory")
    ap.add_argument("--lmdb_path", default="./data/celeba/celeba-lmdb", help="output LMDB root")
    ap.add_argument("--split", default="both", choices=["train", "validation", "both"])
    ap.add_argument("--convert", action="store_true", help="convert TFRecords into NVAE-style LMDB")
    ap.add_argument("--verify_only", action="store_true", help="only inspect the configured data_root")
    ap.add_argument("--max_records", type=int, default=0, help="debug conversion limit per split")
    ap.add_argument("--force", action="store_true", help="rebuild LMDB split if it already exists")
    ap.add_argument("--commit_every", type=int, default=1000, help="LMDB write transaction commit interval")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    spec = get_celebahq256_data_spec()
    if args.convert and not args.verify_only:
        splits = ["train", "validation"] if args.split == "both" else [args.split]
        results = [
            convert_tfrecord_split_to_lmdb(
                tfr_path=args.tfr_path,
                lmdb_root=args.lmdb_path,
                split=split,
                max_records=args.max_records,
                force=args.force,
                commit_every=args.commit_every,
            )
            for split in splits
        ]
        print(json.dumps({"spec": asdict(spec), "converted": results}, indent=2, sort_keys=True))
        return
    print(json.dumps({"spec": asdict(spec), "inspection": inspect_celebahq256_root(args.data_root)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
