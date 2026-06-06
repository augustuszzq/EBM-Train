from pathlib import Path

from polaris_ebm.scripts.current.long_k_scaling_manifest import (
    build_long_k_rows,
    materialize_bundle,
)


def test_long_k_manifest_l0_has_six_submitted_smoke_runs():
    rows = build_long_k_rows()
    l0 = [row for row in rows if row["phase"] == "L0" and row["submit"] == "yes"]

    assert [row["exp_id"] for row in l0] == [
        "L0_ddp_fullk_K200_s20k_seed1",
        "L0_pipe_P8_equal_K200_s20k_seed1",
        "L0_ddp_fullk_K400_s20k_seed1",
        "L0_pipe_P8_equal_K400_s20k_seed1",
        "L0_single_fullk_K400_s20k_seed1",
        "L0_single_emul_P8_equal_K400_s20k_seed1",
    ]

    pipe_k400 = {row["exp_id"]: row for row in l0}["L0_pipe_P8_equal_K400_s20k_seed1"]
    assert pipe_k400["train_mode"] == "pipe_strict"
    assert pipe_k400["world_size"] == "8"
    assert pipe_k400["num_nodes"] == "2"
    assert pipe_k400["ppn"] == "4"
    assert pipe_k400["pipe_stages"] == "8"
    assert pipe_k400["K"] == "400"
    assert pipe_k400["stage_slices"] == "50;50;50;50;50;50;50;50"
    assert pipe_k400["step_size"] == "0.0025"
    assert pipe_k400["weight_mode"] == "uniform"


def test_long_k_manifest_prepares_l1_l2_without_submission():
    rows = build_long_k_rows()
    l1_l2 = [row for row in rows if row["phase"] in {"L1", "L2"}]

    assert l1_l2
    assert all(row["submit"] == "no" for row in l1_l2)
    assert any(row["exp_id"] == "L1_pipe_P8_equal_K400_s300k_seed3" for row in l1_l2)
    assert any(row["exp_id"] == "L2_ddp_fullk_K400_s500k_seed3" for row in l1_l2)


def test_long_k_bundle_materializes_configs_pbs_and_summary_headers(tmp_path):
    rows = build_long_k_rows()
    materialized = materialize_bundle(rows, bundle_root=tmp_path, project_dir=Path("/repo"))

    l0 = next(row for row in materialized if row["exp_id"] == "L0_ddp_fullk_K200_s20k_seed1")
    assert Path(l0["config_path"]).exists()
    assert Path(l0["pbs_path"]).exists()
    assert Path(l0["pbs_output_log"]).parent == tmp_path / "logs"

    cfg = Path(l0["config_path"]).read_text()
    assert "name: cifar10" in cfg
    assert "K: 200" in cfg
    assert "step_size: 0.005" in cfg

    pbs = Path(l0["pbs_path"]).read_text()
    assert "#PBS -q preemptable" in pbs
    assert "TRAIN_MODE=ddp_fullk" in pbs
    assert "CONFIG=" in pbs
    assert "MANIFEST_ROW_PATH=" in pbs
    assert "cp " + l0["config_path"] in pbs
    assert not any(line.startswith("cp ") and line.endswith("/manifest_row.json") for line in pbs.splitlines())
    assert "/repo/scripts/current/pbs_run_ablation_case.pbs" in pbs

    for name in [
        "summaries/long_k_master_registry.csv",
        "summaries/long_k_wallclock_summary.csv",
        "summaries/long_k_fid_vs_step.csv",
        "summaries/long_k_best_final_summary.csv",
        "summaries/long_k_phase_L0_report.md",
    ]:
        assert (tmp_path / name).exists()
