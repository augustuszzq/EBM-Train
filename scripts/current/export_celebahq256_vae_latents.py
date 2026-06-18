#!/usr/bin/env python3
"""Materialize CelebA-HQ256 frozen-VAE latents for latent-space EBM training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch as t
from torch.utils.data import DataLoader

try:
    from celebahq256_data import (
        CelebAHQ256ImageFolderDataset,
        CelebAHQ256LMDBDataset,
        make_celebahq256_transform,
        normalize_celebahq256_split,
    )
    from vae_latent_codec import (
        DEFAULT_LATENT_SCALING_FACTOR,
        DEFAULT_VAE_MODEL,
        encode_images_to_latents,
        load_vae,
        resolve_torch_dtype,
    )
except Exception:
    from polaris_ebm.scripts.current.celebahq256_data import (
        CelebAHQ256ImageFolderDataset,
        CelebAHQ256LMDBDataset,
        make_celebahq256_transform,
        normalize_celebahq256_split,
    )
    from polaris_ebm.scripts.current.vae_latent_codec import (
        DEFAULT_LATENT_SCALING_FACTOR,
        DEFAULT_VAE_MODEL,
        encode_images_to_latents,
        load_vae,
        resolve_torch_dtype,
    )


def build_deterministic_source(root: Path, split: str):
    normalized = normalize_celebahq256_split(split)
    transform = make_celebahq256_transform(image_size=256, train=False)
    if (root / f"{normalized}.lmdb").exists():
        return CelebAHQ256LMDBDataset(root=root, split=normalized, image_size=256, transform=transform)
    return CelebAHQ256ImageFolderDataset(root=root, split=normalized, image_size=256, transform=transform)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Export CelebA-HQ256 VAE latents")
    ap.add_argument("--data-root", default="/eagle/lc-mpi/Zhiqing/polaris_ebm/data/celeba/celeba-lmdb")
    ap.add_argument(
        "--out-root",
        default="/eagle/lc-mpi/Zhiqing/polaris_ebm/data/celeba/celeba-latents/sd-vae-ft-ema-scale018215",
    )
    ap.add_argument("--splits", nargs="+", default=["train", "validation"])
    ap.add_argument("--vae", default=DEFAULT_VAE_MODEL)
    ap.add_argument("--latent-scaling-factor", type=float, default=DEFAULT_LATENT_SCALING_FACTOR)
    ap.add_argument("--encode-mode", choices=["sample", "mean"], default="sample")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--local-files-only", action="store_true")
    return ap.parse_args()


def export_split(args: argparse.Namespace, split: str, vae) -> dict:
    split = normalize_celebahq256_split(split)
    out_dir = Path(args.out_root) / split
    out_path = out_dir / "latents.pt"
    success_path = out_dir / "_SUCCESS.json"
    if out_path.exists() and success_path.exists() and not args.force:
        payload = json.loads(success_path.read_text())
        payload["status"] = "cached"
        return payload
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.force:
        raise RuntimeError(f"{out_path} exists without success marker; pass --force to overwrite")

    ds = build_deterministic_source(Path(args.data_root), split=split)
    if int(args.limit) > 0:
        ds = t.utils.data.Subset(ds, range(min(int(args.limit), len(ds))))
    loader = DataLoader(
        ds,
        batch_size=int(args.batch),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=False,
    )
    t.manual_seed(int(args.seed))
    if t.cuda.is_available():
        t.cuda.manual_seed_all(int(args.seed))

    chunks = []
    labels = []
    count = 0
    for xb, yb in loader:
        z = encode_images_to_latents(
            vae=vae,
            images_m11=xb,
            scaling_factor=float(args.latent_scaling_factor),
            mode=str(args.encode_mode),
        )
        chunks.append(z.cpu().half())
        labels.append(t.as_tensor(yb, dtype=t.long).cpu())
        count += int(z.size(0))
        print(f"[ENCODE] split={split} count={count}", flush=True)
    latents = t.cat(chunks, dim=0).contiguous()
    labels_t = t.cat(labels, dim=0).long().contiguous()
    payload = {
        "latents": latents,
        "labels": labels_t,
        "meta": {
            "benchmark": "celebahq256_latent",
            "source_data_root": str(args.data_root),
            "split": split,
            "vae": str(args.vae),
            "latent_scaling_factor": float(args.latent_scaling_factor),
            "encode_mode": str(args.encode_mode),
            "seed": int(args.seed),
            "source_image_size": 256,
            "latent_shape": [4, 32, 32],
            "num_samples": int(latents.size(0)),
            "dtype": str(latents.dtype),
            "mean": float(latents.float().mean().item()),
            "std": float(latents.float().std(unbiased=False).item()),
            "min": float(latents.float().min().item()),
            "max": float(latents.float().max().item()),
        },
    }
    tmp_path = out_path.with_suffix(".pt.tmp")
    t.save(payload, tmp_path)
    tmp_path.replace(out_path)
    success = dict(payload["meta"])
    success["status"] = "ok"
    success_path.write_text(json.dumps(success, indent=2, sort_keys=True))
    print(f"[DONE] split={split} wrote={out_path} samples={latents.size(0)}", flush=True)
    return success


def main() -> int:
    args = parse_args()
    dtype = resolve_torch_dtype(args.dtype)
    vae = load_vae(
        model_name=str(args.vae),
        device=str(args.device),
        dtype=dtype,
        local_files_only=bool(args.local_files_only),
    )
    summaries = [export_split(args, split, vae) for split in args.splits]
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
