from pathlib import Path

import pytest
import torch

from polaris_ebm.scripts.current.conditional_chain import ChainState


def test_chain_state_requires_batch_aligned_long_labels():
    x = torch.zeros((4, 3, 32, 32), dtype=torch.float32)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    state = ChainState(x=x, y=y, steps_done=torch.zeros(4, dtype=torch.long), valid=torch.ones(4, dtype=torch.bool))
    state.validate()

    with pytest.raises(ValueError, match="labels must be torch.long"):
        ChainState(x=x, y=y.float()).validate()

    with pytest.raises(ValueError, match="batch size"):
        ChainState(x=x, y=torch.tensor([0, 1], dtype=torch.long)).validate()


def test_chain_state_payload_roundtrip_preserves_metadata(tmp_path: Path):
    x = torch.randn((2, 3, 8, 8))
    y = torch.tensor([5, 7], dtype=torch.long)
    chain_id = torch.tensor([11, 13], dtype=torch.long)
    steps_done = torch.tensor([25, 50], dtype=torch.long)
    valid = torch.tensor([True, False], dtype=torch.bool)

    state = ChainState(
        x=x,
        y=y,
        chain_id=chain_id,
        steps_done=steps_done,
        stage_id=3,
        valid=valid,
    )
    payload = state.to_payload()
    restored = ChainState.from_payload(payload)

    assert torch.equal(restored.x, x)
    assert torch.equal(restored.y, y)
    assert torch.equal(restored.chain_id, chain_id)
    assert torch.equal(restored.steps_done, steps_done)
    assert torch.equal(restored.valid, valid)
    assert restored.stage_id == 3

