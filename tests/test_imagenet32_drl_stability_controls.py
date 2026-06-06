import sys
from pathlib import Path

from polaris_ebm.scripts.current import ebm_train_sync_mode_a as sync_trainer
from polaris_ebm.scripts.current.imagenet32_data import get_benchmark_data_spec
from polaris_ebm.scripts.current.imagenet32_drl_stability_manifest import build_stability_rows


def test_pipeline_warmup_skip_only_before_full_diagonal():
    assert sync_trainer.should_skip_optimizer_for_pipeline_warmup(
        step=2,
        pipe_stages=4,
        enabled=True,
    )
    assert not sync_trainer.should_skip_optimizer_for_pipeline_warmup(
        step=3,
        pipe_stages=4,
        enabled=True,
    )
    assert not sync_trainer.should_skip_optimizer_for_pipeline_warmup(
        step=2,
        pipe_stages=4,
        enabled=False,
    )


def test_deep_only_weights_do_not_keep_pipeline_in_warmup_forever():
    alpha = sync_trainer.compute_completion_aware_alpha(
        step=10,
        pipe_stages=4,
        weight_mode="deep_only",
        last2_beta=0.01,
    )

    assert sync_trainer.active_stage_count_from_alpha(alpha) == 1
    assert not sync_trainer.should_skip_optimizer_for_pipeline_warmup(
        step=10,
        pipe_stages=4,
        enabled=True,
    )


def test_stability_crash_guard_reports_all_configured_reasons():
    reasons = sync_trainer.evaluate_stability_crash_guard(
        f_pos=1001.0,
        f_neg_total=-1002.0,
        max_abs_chain=3.0,
        clamp_x=False,
        grad_norm_theta=10001.0,
        crash_abs_energy=1000.0,
        crash_max_abs_chain_unclamped=2.5,
        crash_grad_norm=10000.0,
    )

    assert "abs(f_pos)>1000.0" in reasons
    assert "abs(f_neg_total)>1000.0" in reasons
    assert "max_abs_chain>2.5 with clamp_x=false" in reasons
    assert "grad_norm_theta>10000.0" in reasons


def test_stability_crash_guard_does_not_chain_guard_when_clamped():
    reasons = sync_trainer.evaluate_stability_crash_guard(
        f_pos=0.0,
        f_neg_total=0.0,
        max_abs_chain=3.0,
        clamp_x=True,
        grad_norm_theta=1.0,
        crash_abs_energy=1000.0,
        crash_max_abs_chain_unclamped=2.5,
        crash_grad_norm=10000.0,
    )

    assert reasons == []


def test_soft_stability_guard_marks_unhealthy_energy_and_preclip_grad():
    reasons = sync_trainer.evaluate_soft_stability_guard(
        f_pos=-25.0,
        grad_norm_pre_clip=582.0,
        grad_clip_active_fraction=0.5,
        soft_abs_energy=10.0,
        soft_grad_norm=100.0,
        hard_grad_norm=500.0,
        clip_active_fraction=0.25,
    )

    assert "soft_abs(f_pos)>10.0" in reasons
    assert "soft_grad_norm_pre_clip>100.0" in reasons
    assert "hard_grad_norm_pre_clip>500.0" in reasons
    assert "clip_active_fraction>0.25" in reasons


def test_diagnostic_log_cadence_forces_first_steps_then_respects_log_every():
    assert sync_trainer.should_diagnostic_log(
        step=199,
        debug_level=1,
        log_every=50,
        diagnostic_first_steps=200,
    )
    assert not sync_trainer.should_diagnostic_log(
        step=201,
        debug_level=1,
        log_every=50,
        diagnostic_first_steps=200,
    )
    assert sync_trainer.should_diagnostic_log(
        step=250,
        debug_level=1,
        log_every=50,
        diagnostic_first_steps=200,
    )


def test_sync_parser_accepts_stability_restart_controls(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--skip_optimizer_until_full_diagonal",
            "--grad_clip_norm",
            "10.0",
            "--crash_abs_energy",
            "1000.0",
            "--crash_max_abs_chain_unclamped",
            "2.5",
            "--crash_grad_norm",
            "10000.0",
            "--diagnostic_first_steps",
            "200",
            "--diagnostic_log_every",
            "50",
            "--lr_warmup_steps",
            "1000",
        ],
    )

    args = sync_trainer.parse_args()

    assert args.skip_optimizer_until_full_diagonal is True
    assert args.grad_clip_norm == 10.0
    assert args.crash_abs_energy == 1000.0
    assert args.crash_max_abs_chain_unclamped == 2.5
    assert args.crash_grad_norm == 10000.0
    assert args.diagnostic_first_steps == 200
    assert args.diagnostic_log_every == 50
    assert args.lr_warmup_steps == 1000


def test_sync_parser_accepts_sampler_scale_calibration_controls(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--sampler_prime",
            "50.0",
            "--sampler_temperature",
            "2.0",
            "--energy_loss_scale",
            "0.25",
        ],
    )

    args = sync_trainer.parse_args()

    assert args.sampler_prime == 50.0
    assert args.sampler_temperature == 2.0
    assert args.energy_loss_scale == 0.25


def test_langevin_calibration_defaults_to_existing_step_size():
    calibration = sync_trainer.resolve_langevin_calibration(
        step_size=5e-4,
        noise_std=5e-4,
        sampler_prime=0.0,
        sampler_temperature=1.0,
    )

    assert abs(calibration["sampler_prime"] - 4000.0) < 1e-9
    assert abs(calibration["effective_sampler_prime"] - 4000.0) < 1e-9
    assert abs(calibration["langevin_drift_coeff"] - 5e-4) < 1e-12


def test_langevin_calibration_uses_explicit_prime_and_temperature():
    calibration = sync_trainer.resolve_langevin_calibration(
        step_size=5e-4,
        noise_std=5e-4,
        sampler_prime=100.0,
        sampler_temperature=2.0,
    )

    assert calibration["sampler_prime"] == 100.0
    assert calibration["effective_sampler_prime"] == 50.0
    assert abs(calibration["langevin_drift_coeff"] - 6.25e-6) < 1e-12


def test_langevin_calibration_rejects_bad_temperature():
    try:
        sync_trainer.resolve_langevin_calibration(
            step_size=5e-4,
            noise_std=5e-4,
            sampler_prime=100.0,
            sampler_temperature=0.0,
        )
    except ValueError as exc:
        assert "sampler_temperature" in str(exc)
    else:
        raise AssertionError("expected sampler_temperature validation error")


def test_linear_lr_warmup_uses_global_step_plus_one():
    assert sync_trainer.lr_warmup_factor(step=0, lr_warmup_steps=1000) == 0.001
    assert sync_trainer.lr_warmup_factor(step=499, lr_warmup_steps=1000) == 0.5
    assert sync_trainer.lr_warmup_factor(step=999, lr_warmup_steps=1000) == 1.0
    assert sync_trainer.lr_warmup_factor(step=1000, lr_warmup_steps=1000) == 1.0
    assert sync_trainer.lr_warmup_factor(step=0, lr_warmup_steps=0) == 1.0


def test_apply_lr_warmup_updates_all_optimizer_groups():
    param_a = sync_trainer.t.nn.Parameter(sync_trainer.t.tensor([1.0]))
    param_b = sync_trainer.t.nn.Parameter(sync_trainer.t.tensor([2.0]))
    optimizer = sync_trainer.t.optim.Adam(
        [
            {"params": [param_a], "lr": 0.0},
            {"params": [param_b], "lr": 0.0},
        ],
        lr=0.0,
    )

    lr = sync_trainer.apply_lr_warmup(
        optimizer,
        base_lr=1e-5,
        step=499,
        lr_warmup_steps=1000,
    )

    assert lr == 5e-6
    assert [group["lr"] for group in optimizer.param_groups] == [5e-6, 5e-6]


def test_clamp_saturation_fraction_counts_boundary_values():
    x = sync_trainer.t.tensor([-1.0, -0.9999999, -0.5, 0.999, 1.0])

    frac = sync_trainer.clamp_saturation_fraction(x, eps=1e-6)

    assert abs(frac - (3.0 / 5.0)) < 1e-6


def test_drl_metrics_columns_include_10k_stability_diagnostics():
    columns = sync_trainer.build_metrics_columns(num_stage_slots=4)

    for name in [
        "clamp_sat_frac",
        "E_neg_total",
        "E_neg_minus_E_pos_gap",
        "wallclock_sec",
        "gpu_hours",
        "param_norm",
        "update_norm",
        "update_to_param_ratio",
        "energy_head_weight_norm",
        "energy_head_grad_norm",
        "backbone_grad_norm",
        "grad_norm_pre_clip",
        "grad_norm_post_clip",
        "clip_active",
        "clip_active_fraction",
        "sampler_prime",
        "sampler_temperature",
        "effective_sampler_prime",
        "langevin_drift_coeff",
        "energy_loss_scale",
        "soft_guard_reason",
    ]:
        assert name in columns

    for stage in range(4):
        assert "E_neg_stage%d" % stage in columns
        assert "clamp_sat_frac_stage%d" % stage in columns


def test_drl_metrics_gap_uses_defined_positive_energy_variable():
    source = Path("scripts/current/ebm_train_sync_mode_a.py").read_text()

    assert "loss_pos_val_pre" not in source
    assert "f_pos_total_pre = reduce_mean_scalar(f_pos_val_pre" in source


def test_drl_pbs_defaults_keep_chain_clamping_enabled():
    text = Path("scripts/current/pbs_run_imagenet32_drl_case.pbs").read_text()

    assert "--no_clamp_x" not in text
    assert "--skip_optimizer_until_full_diagonal" in text
    assert "--grad_clip_norm" in text


def test_drl_pbs_forwards_warmup_and_soft_guard_controls():
    text = Path("scripts/current/pbs_run_imagenet32_drl_case.pbs").read_text()

    for token in [
        "LR_WARMUP_STEPS",
        "--lr_warmup_steps",
        "SAMPLER_PRIME",
        "--sampler_prime",
        "SAMPLER_TEMPERATURE",
        "--sampler_temperature",
        "ENERGY_LOSS_SCALE",
        "--energy_loss_scale",
        "SOFT_GUARD_STOP",
        "--soft_guard_stop",
        "--soft_abs_energy",
        "--soft_grad_norm",
        "--hard_grad_norm",
        "--clip_active_fraction",
    ]:
        assert token in text


def test_stability_manifest_matches_short_diagnostic_matrix():
    rows = build_stability_rows()
    by_exp = {row["exp_id"]: row for row in rows}

    expected = {
        "imagenet32_drl_stability_A_ddp_moderate_s500",
        "imagenet32_drl_stability_B_pipe_p4_equal_moderate_s500",
        "imagenet32_drl_stability_C_pipe_p4_equal_conservative_s500",
        "imagenet32_drl_stability_D_pipe_p4_deepest_moderate_s500",
        "imagenet32_drl_stability_E_pipe_p4_equal_safest_s500",
    }
    assert expected.issubset(set(by_exp))

    b = by_exp["imagenet32_drl_stability_B_pipe_p4_equal_moderate_s500"]
    assert b["train_mode"] == "pipe_strict"
    assert b["pipe_stages"] == "4"
    assert b["weight_mode"] == "uniform"
    assert b["steps"] == "500"
    assert b["lr"] == "5e-5"
    assert b["step_size"] == "0.005"
    assert b["noise_std"] == "0.005"
    assert b["clamp_x"] == "true"
    assert b["skip_optimizer_until_full_diagonal"] == "true"
    assert b["submit"] == "no"

    for row in rows:
        assert "sampler_prime" in row
        assert "sampler_temperature" in row
        assert "energy_loss_scale" in row

    e = by_exp["imagenet32_drl_stability_E_pipe_p4_equal_safest_s500"]
    assert e["grad_clip_norm"] == "10.0"
    assert e["lr"] in {"1e-5", "2.5e-5"}


def test_cifar_s0_script_defines_all_four_sign_head_cases():
    text = Path("scripts/run_cifar_drl_s0.sh").read_text()

    assert "S0-A_old_head_sign_pos" in text
    assert "S0-B_old_head_sign_neg" in text
    assert "S0-C_new_head_sign_pos" in text
    assert "S0-D_new_head_sign_neg" in text
    assert "--lr_warmup_steps" in text


def test_imagenet32_drl_10k_wrappers_expose_optional_resume_ckpt():
    for script in [
        "scripts/run_imagenet32_drl_stability_ddp_10k.sh",
        "scripts/run_imagenet32_drl_stability_pipeline_10k.sh",
    ]:
        text = Path(script).read_text()
        assert "RESUME_CKPT" in text
        assert "--resume_ckpt" in text


def test_sync_parser_accepts_reset_optimizer_on_resume(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--resume_ckpt",
            "checkpoint.pt",
            "--resume_reset_optimizer",
        ],
    )

    args = sync_trainer.parse_args()

    assert args.resume_ckpt == "checkpoint.pt"
    assert args.resume_reset_optimizer is True


def test_imagenet32_drl_10k_wrappers_expose_reset_optimizer_resume():
    for script in [
        "scripts/run_imagenet32_drl_stability_ddp_10k.sh",
        "scripts/run_imagenet32_drl_stability_pipeline_10k.sh",
    ]:
        text = Path(script).read_text()
        assert "RESUME_RESET_OPTIMIZER" in text
        assert "--resume_reset_optimizer" in text
    assert "energy_sign" in text
    assert "use_new_energy_head" in text


def test_cifar10_is_supported_as_a_drl_sanity_benchmark():
    spec = get_benchmark_data_spec("cifar10")

    assert spec.name == "cifar10"
    assert spec.conditional is False
    assert spec.image_size == 32


def test_imagenet32_drl_10k_configs_record_ported_energy_head():
    for path in [
        Path("configs/imagenet32_drl_stability_ddp_10k.yaml"),
        Path("configs/imagenet32_drl_stability_pipeline_p4_equal_10k.yaml"),
    ]:
        text = path.read_text()
        assert "steps: 10000" in text
        assert "lr: 0.0000025" in text
        assert "lr_warmup_steps: 2000" in text
        assert "energy_sign: 1" in text
        assert "energy_scale: 1.0" in text
        assert "use_new_energy_head: true" in text
        assert "zero_init_energy_head: true" in text


def test_imagenet32_drl_10k_wrappers_record_resolved_config_and_registry():
    for path in [
        Path("scripts/run_imagenet32_drl_stability_ddp_10k.sh"),
        Path("scripts/run_imagenet32_drl_stability_pipeline_10k.sh"),
    ]:
        text = path.read_text()
        assert "config_resolved.json" in text
        assert "registry_entry.json" in text
        assert '"energy_sign"' in text
        assert '"energy_scale"' in text
        assert '"head_type"' in text
        assert '"head_init"' in text


def test_imagenet32_port_sanity_configs_use_s0_stable_head():
    ddp = Path("configs/imagenet32_drl_port_sanity_ddp_2k.yaml").read_text()
    pipe = Path("configs/imagenet32_drl_port_sanity_pipeline_p4_equal_2k.yaml").read_text()

    for text in (ddp, pipe):
        assert "use_new_energy_head: true" in text
        assert "zero_init_energy_head: true" in text
        assert "energy_sign: 1" in text
        assert "energy_scale: 1.0" in text
        assert "lr_warmup_steps: 1000" in text
        assert "lr: 0.00001" in text
        assert "step_size: 0.0005" in text
        assert "noise_std: 0.0005" in text
        assert "clamp_x: true" in text

    assert "pipe_stages: 1" in ddp
    assert "pipe_stages: 4" in pipe
    assert "weight_mode: uniform" in pipe


def test_imagenet32_port_sanity_scripts_are_isolated_from_old_smoke_defaults():
    ddp = Path("scripts/run_imagenet32_drl_port_sanity_ddp.sh").read_text()
    pipe = Path("scripts/run_imagenet32_drl_port_sanity_pipeline.sh").read_text()

    for text in (ddp, pipe):
        assert "imagenet32_drl_port_sanity" in text
        assert "--lr_warmup_steps" in text
        assert "--grad_clip_norm" in text
        assert "--langevin_sign \"${LANGEVIN_SIGN:--1.0}\"" in text
        assert "--step_size \"${STEP_SIZE:-5e-4}\"" in text
        assert "--noise_std \"${NOISE_STD:-5e-4}\"" in text
        assert "--lr \"${LR:-1e-5}\"" in text
        assert "--no_clamp_x" not in text

    assert "imagenet32_drl_port_sanity_ddp_2k.yaml" in ddp
    assert "imagenet32_drl_port_sanity_pipeline_p4_equal_2k.yaml" in pipe
