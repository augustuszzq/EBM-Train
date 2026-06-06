#!/usr/bin/env python3
"""DRL-style ResNet scalar energy backbone.

This module adapts only the network family used by Diffusion Recovery
Likelihood-style EBMs. It intentionally does not implement recovery likelihood,
diffusion timestep conditioning, or any diffusion training objective.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_tuple_int(values: Iterable[int] | None, default: Sequence[int]) -> tuple[int, ...]:
    if values is None:
        return tuple(int(v) for v in default)
    return tuple(int(v) for v in values)


def _maybe_spectral_norm(module: nn.Module, enabled: bool) -> nn.Module:
    return nn.utils.spectral_norm(module) if bool(enabled) else module


def _init_head_near_zero(module: nn.Module, *, exact_zero: bool) -> None:
    """Initialize a scalar energy head near zero without changing the backbone."""
    weight = getattr(module, "weight_orig", None)
    if weight is None:
        weight = getattr(module, "weight", None)
    bias = getattr(module, "bias", None)
    if weight is not None:
        if exact_zero:
            nn.init.zeros_(weight)
        else:
            nn.init.normal_(weight, mean=0.0, std=1e-4)
    if bias is not None:
        nn.init.zeros_(bias)


class ResBlock(nn.Module):
    """Residual block with optional average-pool downsampling."""

    def __init__(self, in_ch: int, out_ch: int, *, downsample: bool = False, spectral_norm: bool = False):
        super().__init__()
        self.downsample = bool(downsample)
        self.activation = nn.SiLU()
        self.conv1 = _maybe_spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1), spectral_norm)
        self.conv2 = _maybe_spectral_norm(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1), spectral_norm)
        if in_ch != out_ch:
            self.skip = _maybe_spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=1), spectral_norm)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.activation(x)
        h = self.conv1(h)
        h = self.activation(h)
        h = self.conv2(h)
        if self.downsample:
            h = F.avg_pool2d(h, kernel_size=2)
            residual = F.avg_pool2d(residual, kernel_size=2)
        return (h + residual) / math.sqrt(2.0)


class SelfAttention2d(nn.Module):
    """Lightweight non-local attention block for 16x16 feature maps."""

    def __init__(self, channels: int, *, spectral_norm: bool = False):
        super().__init__()
        attn_ch = max(1, channels // 8)
        self.query = _maybe_spectral_norm(nn.Conv2d(channels, attn_ch, kernel_size=1), spectral_norm)
        self.key = _maybe_spectral_norm(nn.Conv2d(channels, attn_ch, kernel_size=1), spectral_norm)
        self.value = _maybe_spectral_norm(nn.Conv2d(channels, channels, kernel_size=1), spectral_norm)
        self.proj = _maybe_spectral_norm(nn.Conv2d(channels, channels, kernel_size=1), spectral_norm)
        self.gamma = nn.Parameter(torch.zeros(()))
        self.scale = attn_ch**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        q = self.query(x).reshape(b, -1, n).transpose(1, 2)
        k = self.key(x).reshape(b, -1, n)
        attn = torch.softmax(torch.bmm(q, k) * self.scale, dim=-1)
        v = self.value(x).reshape(b, c, n).transpose(1, 2)
        out = torch.bmm(attn, v).transpose(1, 2).reshape(b, c, h, w)
        return x + self.gamma * self.proj(out)


class DRLResNetEnergy(nn.Module):
    """Scalar E(x) backbone for ImageNet-32 external validation.

    `y` is accepted for compatibility with conditional training plumbing but is
    deliberately ignored; this first DRL-backbone validation is unconditional.
    """

    def __init__(
        self,
        *,
        image_size: int = 32,
        in_channels: int = 3,
        ch: int = 128,
        ch_mult: Sequence[int] = (1, 2, 2, 2),
        num_res_blocks: int = 4,
        attention_resolutions: Sequence[int] = (),
        spectral_norm: bool = True,
        energy_sign: int | float = 1.0,
        energy_scale: float = 1.0,
        use_new_energy_head: bool = False,
        zero_init_energy_head: bool = False,
    ):
        super().__init__()
        if int(image_size) <= 0:
            raise ValueError("image_size must be > 0")
        if int(ch) <= 0:
            raise ValueError("ch must be > 0")
        if int(num_res_blocks) <= 0:
            raise ValueError("num_res_blocks must be > 0")
        if not ch_mult:
            raise ValueError("ch_mult must not be empty")
        if float(energy_sign) not in (-1.0, 1.0):
            raise ValueError("energy_sign must be +1 or -1")
        if not math.isfinite(float(energy_scale)):
            raise ValueError("energy_scale must be finite")

        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.ch = int(ch)
        self.ch_mult = tuple(int(v) for v in ch_mult)
        self.num_res_blocks = int(num_res_blocks)
        self.attention_resolutions = tuple(int(v) for v in attention_resolutions)
        self.spectral_norm = bool(spectral_norm)
        self.energy_sign = float(energy_sign)
        self.energy_scale = float(energy_scale)
        self.use_new_energy_head = bool(use_new_energy_head)
        self.zero_init_energy_head = bool(zero_init_energy_head)

        layers: list[nn.Module] = [
            _maybe_spectral_norm(nn.Conv2d(self.in_channels, self.ch, kernel_size=3, padding=1), self.spectral_norm)
        ]
        in_ch = self.ch
        resolution = self.image_size

        for level, mult in enumerate(self.ch_mult):
            out_ch = self.ch * int(mult)
            for _ in range(self.num_res_blocks):
                layers.append(ResBlock(in_ch, out_ch, spectral_norm=self.spectral_norm))
                in_ch = out_ch
            if resolution in self.attention_resolutions:
                layers.append(SelfAttention2d(in_ch, spectral_norm=self.spectral_norm))
            if level != len(self.ch_mult) - 1:
                next_out_ch = self.ch * int(self.ch_mult[level + 1])
                layers.append(ResBlock(in_ch, next_out_ch, downsample=True, spectral_norm=self.spectral_norm))
                in_ch = next_out_ch
                resolution = max(1, resolution // 2)

        self.net = nn.Sequential(*layers)
        self.activation = nn.SiLU()
        if self.use_new_energy_head:
            # The port-sanity head is a plain scalar linear layer. It deliberately
            # avoids spectral-norm wrapping and any final nonlinearity so the
            # ordinary EBM sign can be tested directly.
            self.head = nn.Linear(in_ch, 1)
        else:
            self.head = _maybe_spectral_norm(nn.Linear(in_ch, 1), self.spectral_norm)
        if self.zero_init_energy_head:
            _init_head_near_zero(self.head, exact_zero=self.use_new_energy_head)

    def raw_energy(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        del y
        h = self.net(x)
        h = self.activation(h)
        h = h.mean(dim=(2, 3))
        return self.head(h).squeeze(-1)

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        raw = self.raw_energy(x, y)
        return self.energy_scale * self.energy_sign * raw


def build_drl_resnet_energy_from_config(model_config: Mapping | None, *, image_size: int = 32) -> DRLResNetEnergy:
    cfg = dict(model_config or {})
    attention = cfg.get("attention_resolutions", ())
    if attention is None:
        attention = ()
    return DRLResNetEnergy(
        image_size=int(cfg.get("image_size", image_size)),
        in_channels=int(cfg.get("in_channels", 3)),
        ch=int(cfg.get("ch", 128)),
        ch_mult=_as_tuple_int(cfg.get("ch_mult"), (1, 2, 2, 2)),
        num_res_blocks=int(cfg.get("num_res_blocks", 4)),
        attention_resolutions=_as_tuple_int(attention, ()),
        spectral_norm=bool(cfg.get("spectral_norm", True)),
        energy_sign=float(cfg.get("energy_sign", 1.0)),
        energy_scale=float(cfg.get("energy_scale", 1.0)),
        use_new_energy_head=bool(cfg.get("use_new_energy_head", False)),
        zero_init_energy_head=bool(cfg.get("zero_init_energy_head", False)),
    )
