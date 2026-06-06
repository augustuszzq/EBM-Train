from pathlib import Path


def test_overhead_attribution_rows_cover_overhead_hypotheses():
    from scripts.current.long_k_overhead_attribution_manifest import build_rows

    rows = build_rows()
    exp_ids = {row["exp_id"] for row in rows}

    assert exp_ids == {
        "A1_single_global_fullstep_K400_b256_s3k_seed1",
        "A1_single_local_fullstep_K400_b32_s3k_seed1",
        "A1_independent_fullstep_K400_b256_s3k_seed1",
        "A1_independent_sampling_K400_b256_s3k_seed1",
        "A1_sampling_barrier_K400_b256_s3k_seed1",
        "A1_ddp_train_only_K400_b256_s3k_seed1",
        "A1_ddp_fullstep_K400_b256_s3k_seed1",
        "A1_pipeline_stage_nocomm_P8_K400_b256_s3k_seed1",
        "A1_pipeline_stage_ring_P8_K400_b256_s3k_seed1",
        "A1_pipeline_fullstep_ring_P8_K400_b256_s3k_seed1",
    }
    assert {row["submit"] for row in rows} == {"yes"}


def test_overhead_attribution_rows_use_short_original_cifar_k400():
    from scripts.current.long_k_overhead_attribution_manifest import build_rows

    rows = build_rows()
    for row in rows:
        assert row["benchmark"] == "cifar10"
        assert row["K"] == "400"
        assert row["steps"] == "3000"
        assert row["uses_drl_backbone"] == "false"
        assert row["uses_imagenet32"] == "false"


def test_overhead_attribution_world_and_batch_are_explicit():
    from scripts.current.long_k_overhead_attribution_manifest import build_rows

    rows = {row["exp_id"]: row for row in build_rows()}

    assert rows["A1_single_global_fullstep_K400_b256_s3k_seed1"]["world_size"] == "1"
    assert rows["A1_single_global_fullstep_K400_b256_s3k_seed1"]["batch_size"] == "256"
    assert rows["A1_single_global_fullstep_K400_b256_s3k_seed1"]["local_batch"] == "256"
    assert rows["A1_single_local_fullstep_K400_b32_s3k_seed1"]["world_size"] == "1"
    assert rows["A1_single_local_fullstep_K400_b32_s3k_seed1"]["batch_size"] == "32"
    assert rows["A1_single_local_fullstep_K400_b32_s3k_seed1"]["local_batch"] == "32"
    assert rows["A1_ddp_fullstep_K400_b256_s3k_seed1"]["world_size"] == "8"
    assert rows["A1_ddp_fullstep_K400_b256_s3k_seed1"]["local_batch"] == "32"
    assert rows["A1_pipeline_fullstep_ring_P8_K400_b256_s3k_seed1"]["pipe_stages"] == "8"
    assert rows["A1_pipeline_fullstep_ring_P8_K400_b256_s3k_seed1"]["slice_steps"] == "50"


def test_overhead_attribution_materialize_writes_expected_files(tmp_path):
    from scripts.current.long_k_overhead_attribution_manifest import build_rows, materialize

    rows = materialize(build_rows(), tmp_path, Path("/repo"))

    assert (tmp_path / "summaries" / "overhead_attribution_master_registry.csv").exists()
    assert (tmp_path / "summaries" / "overhead_attribution_submitted.csv").exists() is False
    for row in rows:
        assert Path(row["config_path"]).exists()
        assert Path(row["pbs_path"]).exists()
        assert Path(row["manifest_row_path"]).exists()
        pbs = Path(row["pbs_path"]).read_text()
        assert "ebm_overhead_microbench.py" in pbs
        assert "MICROBENCH_MODE" in pbs
