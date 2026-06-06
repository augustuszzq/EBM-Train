from types import SimpleNamespace

import torch as t

from polaris_ebm.scripts.current.ebm_train_sync_mode_a import (
    chain_state_from_buffers,
    maybe_resume_from_checkpoint,
    rank_state_ckpt_path,
    restore_chain_state_into_buffers,
    state_to_cpu_payload,
)


def test_rank_state_resume_prefers_rank_local_extra_state(tmp_path):
    ckpt_path = tmp_path / "checkpoints" / "ckpt_step10.pt"
    ckpt_path.parent.mkdir(parents=True)

    model = t.nn.Linear(3, 2)
    optimizer = t.optim.Adam(model.parameters(), lr=1e-3)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": 10,
        "extra_state": {"source": "main"},
    }
    t.save(payload, ckpt_path)

    local_path = rank_state_ckpt_path(str(ckpt_path), rank=2)
    local_payload = dict(payload)
    local_payload["extra_state"] = {"source": "rank2", "value": 7}
    import os

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    t.save(local_payload, local_path)

    fresh_model = t.nn.Linear(3, 2)
    fresh_optimizer = t.optim.Adam(fresh_model.parameters(), lr=1e-3)
    extra = {}
    start_step = maybe_resume_from_checkpoint(
        args=SimpleNamespace(resume_ckpt=str(ckpt_path), steps=100),
        model=fresh_model,
        optimizer=fresh_optimizer,
        device=t.device("cpu"),
        rank=2,
        extra_state_out=extra,
    )
    assert start_step == 11
    assert extra["source"] == "rank2"
    assert extra["value"] == 7


def test_chain_state_buffer_restore_roundtrip():
    x = t.randn(2, 3, 8, 8)
    y = t.tensor([1, 4], dtype=t.long)
    chain_id = t.tensor([5, 6], dtype=t.long)
    steps_done = t.tensor([10, 20], dtype=t.long)
    valid = t.tensor([True, False], dtype=t.bool)

    payload = state_to_cpu_payload(
        chain_state_from_buffers(
            x=x,
            y=y,
            chain_id=chain_id,
            steps_done=steps_done,
            valid=valid,
            stage_id=3,
        )
    )

    dst_x = t.zeros_like(x)
    dst_y = t.zeros_like(y)
    dst_chain_id = t.zeros_like(chain_id)
    dst_steps = t.zeros_like(steps_done)
    dst_valid = t.zeros_like(valid)
    restored = restore_chain_state_into_buffers(
        payload,
        device=t.device("cpu"),
        chain_buf=dst_x,
        label_buf=dst_y,
        chain_id_buf=dst_chain_id,
        steps_done_buf=dst_steps,
        valid_buf=dst_valid,
    )
    assert restored is True
    assert t.equal(dst_x, x)
    assert t.equal(dst_y, y)
    assert t.equal(dst_chain_id, chain_id)
    assert t.equal(dst_steps, steps_done)
    assert t.equal(dst_valid, valid)
