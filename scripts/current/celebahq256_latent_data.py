#!/usr/bin/env python3
"""Materialized CelebA-HQ256 latent dataset helpers.

The latent benchmark stores frozen VAE latents with shape `[N, 4, 32, 32]`.
Training still uses the ordinary EBM objective; only the data representation
changes from pixel space to VAE latent space.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch as t

try:
    from celebahq256_data import normalize_celebahq256_split
except Exception:
    from polaris_ebm.scripts.current.celebahq256_data import normalize_celebahq256_split


LATENT_SCALING_FACTOR = 0.18215


class CelebAHQ256LatentDataset(t.utils.data.Dataset):
    """Read a materialized latent split from `<root>/<split>/latents.pt`."""

    def __init__(self, root: str | Path, split: str = "train"):
        self.root = Path(root)
        self.split = normalize_celebahq256_split(split)
        self.path = self.root / self.split / "latents.pt"
        if not self.path.exists():
            raise RuntimeError(
                "missing CelebA-HQ256 latent cache: %s. Run "
                "scripts/current/export_celebahq256_vae_latents.py first." % self.path
            )
        payload = t.load(self.path, map_location="cpu")
        if not isinstance(payload, dict) or "latents" not in payload:
            raise RuntimeError(f"{self.path} must contain a dict with key 'latents'")
        latents = payload["latents"]
        if not t.is_tensor(latents) or latents.dim() != 4:
            raise RuntimeError(f"latents must be a tensor [N,4,32,32], got {type(latents)!r}")
        if tuple(latents.shape[1:]) != (4, 32, 32):
            raise RuntimeError(f"latents must have shape [N,4,32,32], got {tuple(latents.shape)}")
        self.latents = latents.float().contiguous()
        labels = payload.get("labels")
        if labels is None:
            labels = t.zeros((self.latents.size(0),), dtype=t.long)
        if not t.is_tensor(labels) or labels.numel() != self.latents.size(0):
            raise RuntimeError("labels must be a tensor with one entry per latent")
        self.labels = labels.long().view(-1).contiguous()
        meta = payload.get("meta", {})
        self.meta: Dict = meta if isinstance(meta, dict) else {}

    def __len__(self) -> int:
        return int(self.latents.size(0))

    def __getitem__(self, index: int):
        idx = int(index)
        return self.latents[idx], int(self.labels[idx].item())


def build_celebahq256_latent_dataset(root: str | Path, split: str = "train"):
    return CelebAHQ256LatentDataset(root=root, split=split)
