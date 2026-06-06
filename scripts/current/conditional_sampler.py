#!/usr/bin/env python3
"""Conditional Langevin sampler operating on E(x, y)."""

from __future__ import annotations

import torch as t
import torch.nn as nn


def langevin_sample_conditional(
    model: nn.Module,
    chain: t.Tensor,
    labels: t.Tensor,
    k_steps: int,
    langevin_sign: float,
    step_size: float,
    noise_std: float,
    clamp_x: bool,
    clamp_last_only: bool = False,
    drift_coeff: float | None = None,
) -> t.Tensor:
    if labels.dtype != t.long:
        raise ValueError("labels must be torch.long")
    if labels.dim() != 1 or int(labels.size(0)) != int(chain.size(0)):
        raise ValueError("labels must be 1D and batch-aligned with chain")
    x = chain.detach()
    labels_ref = labels.detach().clone()
    coeff = float(step_size) if drift_coeff is None else float(drift_coeff)
    for t_step in range(int(k_steps)):
        if not t.equal(labels, labels_ref):
            raise RuntimeError("labels changed during conditional Langevin sampling")
        with t.enable_grad():
            x = x.detach().requires_grad_(True)
            score = model(x, labels).sum()
            grad = t.autograd.grad(score, x, retain_graph=False, create_graph=False)[0]
        x = x.detach()
        x.add_(float(langevin_sign) * coeff * grad).add_(float(noise_std) * t.randn_like(x))
        do_clamp = bool(clamp_x) and ((not bool(clamp_last_only)) or (t_step == (int(k_steps) - 1)))
        if do_clamp:
            x.clamp_(-1.0, 1.0)
    chain.copy_(x)
    return x
