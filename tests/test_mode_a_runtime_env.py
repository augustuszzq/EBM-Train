import sys

from polaris_ebm.scripts.current.ebm_train_sync_mode_a import normalize_local_rank, parse_args


def test_parse_args_defaults_from_environment(monkeypatch):
    monkeypatch.setenv("MODE", "pipeline")
    monkeypatch.setenv("SAMPLE_DTYPE", "bf16")
    monkeypatch.setattr(sys, "argv", ["ebm_train_sync_mode_a.py"])
    args = parse_args()
    assert args.mode == "pipeline"
    assert args.sample_dtype == "bf16"
    assert int(args.steps) == 20000
    assert int(args.vis_every) == 200
    assert str(args.stage_weight_mode) == "manual"


def test_normalize_local_rank_with_single_visible_device(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    local_rank, reason = normalize_local_rank(2)
    assert local_rank == 0
    assert reason == "single_visible_device"


def test_normalize_local_rank_wraps_out_of_range(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    local_rank, reason = normalize_local_rank(5)
    assert local_rank == 1
    assert reason == "local_rank_out_of_visible_range"


def test_parse_args_accepts_ddp_fullk_mode(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["ebm_train_sync_mode_a.py", "--mode", "ddp_fullk"])
    args = parse_args()
    assert args.mode == "ddp_fullk"


def test_parse_args_langevin_sign_defaults_to_negative(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["ebm_train_sync_mode_a.py"])
    args = parse_args()
    assert float(args.langevin_sign) == -1.0


def test_parse_args_accepts_custom_langevin_sign(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["ebm_train_sync_mode_a.py", "--langevin_sign", "1"])
    args = parse_args()
    assert float(args.langevin_sign) == 1.0


def test_parse_args_accepts_clamp_last_only(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["ebm_train_sync_mode_a.py", "--clamp_last_only"])
    args = parse_args()
    assert bool(args.clamp_last_only) is True


def test_parse_args_accepts_stage_weight_mode_and_vis(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ebm_train_sync_mode_a.py",
            "--stage_weight_mode",
            "power",
            "--stage_weight_gamma",
            "3.0",
            "--vis_every",
            "500",
            "--vis_num",
            "128",
            "--vis_nrow",
            "16",
            "--vis_stage",
            "3",
            "--vis_dir",
            "/tmp/vis",
        ],
    )
    args = parse_args()
    assert str(args.stage_weight_mode) == "power"
    assert float(args.stage_weight_gamma) == 3.0
    assert int(args.vis_every) == 500
    assert int(args.vis_num) == 128
    assert int(args.vis_nrow) == 16
    assert int(args.vis_stage) == 3
    assert str(args.vis_dir) == "/tmp/vis"


def test_parse_args_accepts_stage2_beta_warmup_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ebm_train_sync_mode_a.py",
            "--stage2_beta_warmup",
            "--stage2_beta_final",
            "0.02",
            "--stage2_beta_warmup_t0",
            "500",
            "--stage2_beta_warmup_t1",
            "1500",
        ],
    )
    args = parse_args()
    assert bool(args.stage2_beta_warmup) is True
    assert float(args.stage2_beta_final) == 0.02
    assert int(args.stage2_beta_warmup_t0) == 500
    assert int(args.stage2_beta_warmup_t1) == 1500
