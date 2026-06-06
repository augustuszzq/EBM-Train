#!/usr/bin/env python3
"""Minimal conditional EBM backbone compatible with current conv family."""

from __future__ import annotations

import torch as t
import torch.nn as nn


class ConditionalEnergyModel(nn.Module):
    def __init__(self, n_c: int = 3, n_f: int = 64, num_classes: int = 1000, image_size: int = 32, leak: float = 0.2):
        super().__init__()
        if int(num_classes) <= 0:
            raise ValueError("num_classes must be > 0 for conditional model")
        self.num_classes = int(num_classes)
        self.image_size = int(image_size)
        self.features = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),
            nn.LeakyReLU(leak, inplace=True),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),
            nn.LeakyReLU(leak, inplace=True),
            nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1),
            nn.LeakyReLU(leak, inplace=True),
            nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1),
            nn.LeakyReLU(leak, inplace=True),
        )
        final_hw = max(1, self.image_size // 8)
        feat_dim = (n_f * 8) * final_hw * final_hw
        self.proj = nn.Linear(feat_dim, n_f * 8)
        self.energy_head = nn.Linear(n_f * 8, 1)
        self.class_embed = nn.Embedding(self.num_classes, n_f * 8)

    def forward(self, x: t.Tensor, y: t.Tensor) -> t.Tensor:
        if y.dtype != t.long:
            raise ValueError("labels must be torch.long")
        if y.dim() != 1 or int(y.size(0)) != int(x.size(0)):
            raise ValueError("labels must be 1D and batch-aligned with x")
        h = self.features(x).flatten(start_dim=1)
        h = self.proj(h)
        label_vec = self.class_embed(y)
        base = self.energy_head(h).view(-1)
        proj_term = (h * label_vec).sum(dim=1) / float(h.size(1))
        return base + proj_term
