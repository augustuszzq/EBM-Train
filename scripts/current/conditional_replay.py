#!/usr/bin/env python3
"""Label-bound replay buffer for conditional EBM chains."""

from __future__ import annotations

from typing import Dict, Tuple

import torch as t

try:
    from conditional_chain import ChainState
except Exception:
    from polaris_ebm.scripts.current.conditional_chain import ChainState


class LabelReplayBuffer:
    def __init__(self, capacity: int, image_shape: Tuple[int, int, int], num_classes: int):
        if int(capacity) <= 0:
            raise ValueError("capacity must be > 0")
        if int(num_classes) <= 0:
            raise ValueError("num_classes must be > 0")
        self.capacity = int(capacity)
        self.image_shape = tuple(int(x) for x in image_shape)
        self.num_classes = int(num_classes)
        self.x = t.zeros((self.capacity,) + self.image_shape, dtype=t.float32)
        self.y = t.full((self.capacity,), -1, dtype=t.long)
        self.steps_done = t.zeros((self.capacity,), dtype=t.long)
        self.valid = t.zeros((self.capacity,), dtype=t.bool)
        self._next_slot = 0

    def _fresh_noise(self, batch: int) -> t.Tensor:
        return t.empty((batch,) + self.image_shape, dtype=t.float32).uniform_(-1.0, 1.0)

    def _matching_slots(self, label: int) -> t.Tensor:
        mask = self.valid & (self.y == int(label))
        return mask.nonzero(as_tuple=False).view(-1)

    def sample(self, labels: t.Tensor) -> Tuple[ChainState, Dict[str, int]]:
        if labels.dtype != t.long or labels.dim() != 1:
            raise ValueError("labels must be a 1D torch.long tensor")
        batch = int(labels.size(0))
        out_x = self._fresh_noise(batch)
        out_y = labels.clone()
        out_steps = t.zeros(batch, dtype=t.long)
        chain_id = t.full((batch,), -1, dtype=t.long)
        valid = t.ones(batch, dtype=t.bool)
        matched_hits = 0
        restarts = 0

        for i, label in enumerate(labels.tolist()):
            slots = self._matching_slots(label)
            if int(slots.numel()) > 0:
                slot = int(slots[0].item())
                out_x[i].copy_(self.x[slot])
                out_steps[i] = self.steps_done[slot]
                chain_id[i] = slot
                matched_hits += 1
            else:
                slot = self._next_slot
                self._next_slot = (self._next_slot + 1) % self.capacity
                chain_id[i] = slot
                restarts += 1

        state = ChainState(
            x=out_x,
            y=out_y,
            chain_id=chain_id,
            steps_done=out_steps,
            valid=valid,
        ).validate()
        return state, {"matched_hits": matched_hits, "restarts": restarts}

    def update_slots(self, state: ChainState) -> None:
        state = state.validate()
        if state.chain_id is None:
            raise ValueError("chain_id is required to update replay slots")
        for i in range(state.batch_size):
            slot = int(state.chain_id[i].item())
            if slot < 0 or slot >= self.capacity:
                raise ValueError(f"invalid replay slot index: {slot}")
            self.x[slot].copy_(state.x[i].detach().cpu())
            self.y[slot] = state.y[i].detach().cpu()
            if state.steps_done is not None:
                self.steps_done[slot] = state.steps_done[i].detach().cpu()
            self.valid[slot] = bool(state.valid[i].item()) if state.valid is not None else True

    def state_dict(self) -> Dict[str, t.Tensor]:
        return {
            "capacity": t.tensor([self.capacity], dtype=t.long),
            "image_shape": t.tensor(list(self.image_shape), dtype=t.long),
            "num_classes": t.tensor([self.num_classes], dtype=t.long),
            "x": self.x.clone(),
            "y": self.y.clone(),
            "steps_done": self.steps_done.clone(),
            "valid": self.valid.clone(),
            "next_slot": t.tensor([self._next_slot], dtype=t.long),
        }

    def load_state_dict(self, state: Dict[str, t.Tensor]) -> None:
        self.x.copy_(state["x"])
        self.y.copy_(state["y"])
        self.steps_done.copy_(state["steps_done"])
        self.valid.copy_(state["valid"])
        self._next_slot = int(state["next_slot"][0].item())
