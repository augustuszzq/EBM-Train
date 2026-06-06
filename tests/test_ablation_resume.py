from types import SimpleNamespace

import torch as t

from scripts.current.ebm_train_sync_mode_a import maybe_resume_from_checkpoint


def test_resume_helper_restores_model_optimizer_and_start_step(tmp_path):
    model = t.nn.Linear(3, 2)
    optimizer = t.optim.Adam(model.parameters(), lr=1e-3)
    inp = t.randn(4, 3)
    loss = model(inp).sum()
    loss.backward()
    optimizer.step()

    saved_model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    saved_optimizer_state = optimizer.state_dict()
    ckpt_path = tmp_path / "ckpt_step123.pt"
    t.save(
        {
            "model_state_dict": saved_model_state,
            "optimizer_state_dict": saved_optimizer_state,
            "step": 123,
        },
        ckpt_path,
    )

    fresh_model = t.nn.Linear(3, 2)
    fresh_optimizer = t.optim.Adam(fresh_model.parameters(), lr=1e-3)
    args = SimpleNamespace(resume_ckpt=str(ckpt_path), steps=500)

    start_step = maybe_resume_from_checkpoint(
        args=args,
        model=fresh_model,
        optimizer=fresh_optimizer,
        device=t.device("cpu"),
        rank=0,
    )

    assert start_step == 124
    for key, value in saved_model_state.items():
        assert t.equal(fresh_model.state_dict()[key], value)
    assert fresh_optimizer.state_dict()["state"]


def test_resume_helper_treats_none_extra_state_as_empty(tmp_path):
    model = t.nn.Linear(3, 2)
    optimizer = t.optim.Adam(model.parameters(), lr=1e-3)
    ckpt_path = tmp_path / "ckpt_step7.pt"
    t.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": 7,
            "extra_state": None,
        },
        ckpt_path,
    )

    fresh_model = t.nn.Linear(3, 2)
    fresh_optimizer = t.optim.Adam(fresh_model.parameters(), lr=1e-3)
    args = SimpleNamespace(resume_ckpt=str(ckpt_path), steps=20)
    extra_state = {"stale": 1}

    start_step = maybe_resume_from_checkpoint(
        args=args,
        model=fresh_model,
        optimizer=fresh_optimizer,
        device=t.device("cpu"),
        rank=0,
        extra_state_out=extra_state,
    )

    assert start_step == 8
    assert extra_state == {}
