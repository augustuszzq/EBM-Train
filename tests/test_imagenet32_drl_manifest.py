from polaris_ebm.scripts.current.imagenet32_drl_manifest import build_drl_rows, imagenet32_materialized_success


def test_imagenet32_drl_manifest_contains_phase0_smoke_pair():
    rows = build_drl_rows()
    by_exp = {row["exp_id"]: row for row in rows}

    assert "imagenet32_drl_ddp_k100_s5k_seed1" in by_exp
    assert "imagenet32_drl_pipe_p4_equal_k100_s5k_seed1" in by_exp

    ddp = by_exp["imagenet32_drl_ddp_k100_s5k_seed1"]
    pipe = by_exp["imagenet32_drl_pipe_p4_equal_k100_s5k_seed1"]

    assert ddp["phase"] == "phase0_smoke"
    assert ddp["train_mode"] == "ddp_fullk"
    assert ddp["world_size"] == "4"
    assert ddp["config"].endswith("configs/imagenet32_drl_ddp_smoke.yaml")
    assert ddp["submit"] == "no"

    assert pipe["phase"] == "phase0_smoke"
    assert pipe["train_mode"] == "pipe_strict"
    assert pipe["pipe_stages"] == "4"
    assert pipe["weight_mode"] == "uniform"
    assert pipe["config"].endswith("configs/imagenet32_drl_pipeline_p4_equal_smoke.yaml")
    assert pipe["submit"] == "no"


def test_imagenet32_drl_manifest_keeps_paper_runs_prepared_but_not_submitted():
    rows = build_drl_rows()
    later = [row for row in rows if row["phase"] != "phase0_smoke"]

    assert later
    assert all(row["benchmark"] == "imagenet32" for row in rows)
    assert all(row["model_family"] == "drl_resnet_energy" for row in rows)
    assert all(row["submit"] == "no" for row in later)
    assert {"phase1_early", "phase2_300k", "phase3_500k"}.issubset({row["phase"] for row in rows})


def test_imagenet32_materialized_success_accepts_split_level_success_files(tmp_path):
    for split in ("train", "val"):
        split_dir = tmp_path / split
        split_dir.mkdir()
        (split_dir / "_SUCCESS.json").write_text("{}")

    assert imagenet32_materialized_success(tmp_path)
