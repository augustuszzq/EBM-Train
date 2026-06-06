import sys

from polaris_ebm.scripts.current.eval_generate import parse_args


def test_eval_generate_accepts_langevin_sign(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_generate.py",
            "--ckpt",
            "/tmp/mock.pt",
            "--out_dir",
            "/tmp/out",
            "--langevin_sign",
            "1",
        ],
    )
    args = parse_args()
    assert float(args.langevin_sign) == 1.0


def test_eval_generate_langevin_sign_default(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_generate.py", "--ckpt", "/tmp/mock.pt", "--out_dir", "/tmp/out"],
    )
    args = parse_args()
    assert float(args.langevin_sign) == -1.0


def test_eval_generate_accepts_clamp_last_only(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_generate.py",
            "--ckpt",
            "/tmp/mock.pt",
            "--out_dir",
            "/tmp/out",
            "--clamp_last_only",
        ],
    )
    args = parse_args()
    assert bool(args.clamp_last_only) is True
