#!/usr/bin/env python3
"""Shared conditional chain-state contract used by larger benchmark paths."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

import torch as t


@dataclass(frozen=True)
class ChainState:
    x: t.Tensor
    y: t.Tensor
    chain_id: Optional[t.Tensor] = None
    steps_done: Optional[t.Tensor] = None
    stage_id: Optional[int] = None
    valid: Optional[t.Tensor] = None

    def _replace(self, **kwargs) -> "ChainState":
        return replace(self, **kwargs)

    @property
    def batch_size(self) -> int:
        return int(self.x.size(0))

    def validate(self) -> "ChainState":
        if not t.is_tensor(self.x) or self.x.dim() != 4:
            raise ValueError("x must be a 4D image tensor [B,C,H,W]")
        if not t.is_tensor(self.y):
            raise ValueError("y must be a tensor")
        if self.y.dtype != t.long:
            raise ValueError("labels must be torch.long")
        if self.y.dim() != 1:
            raise ValueError("y must be a 1D tensor with one label per batch element")
        if int(self.y.size(0)) != self.batch_size:
            raise ValueError("label batch size must match x batch size")
        if self.chain_id is not None:
            if self.chain_id.dtype != t.long or self.chain_id.dim() != 1 or int(self.chain_id.size(0)) != self.batch_size:
                raise ValueError("chain_id must be a 1D torch.long tensor aligned with batch size")
        if self.steps_done is not None:
            if self.steps_done.dtype != t.long or self.steps_done.dim() != 1 or int(self.steps_done.size(0)) != self.batch_size:
                raise ValueError("steps_done must be a 1D torch.long tensor aligned with batch size")
        if self.valid is not None:
            if self.valid.dtype != t.bool or self.valid.dim() != 1 or int(self.valid.size(0)) != self.batch_size:
                raise ValueError("valid must be a 1D torch.bool tensor aligned with batch size")
        return self

    def to(self, device: t.device | str) -> "ChainState":
        return ChainState(
            x=self.x.to(device),
            y=self.y.to(device),
            chain_id=None if self.chain_id is None else self.chain_id.to(device),
            steps_done=None if self.steps_done is None else self.steps_done.to(device),
            stage_id=self.stage_id,
            valid=None if self.valid is None else self.valid.to(device),
        )

    def to_payload(self) -> Dict[str, Any]:
        self.validate()
        return {
            "x": self.x,
            "y": self.y,
            "chain_id": self.chain_id,
            "steps_done": self.steps_done,
            "stage_id": self.stage_id,
            "valid": self.valid,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ChainState":
        state = cls(
            x=payload["x"],
            y=payload["y"],
            chain_id=payload.get("chain_id"),
            steps_done=payload.get("steps_done"),
            stage_id=payload.get("stage_id"),
            valid=payload.get("valid"),
        )
        return state.validate()
