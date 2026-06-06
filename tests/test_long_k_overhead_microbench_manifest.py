from pathlib import Path


def test_overhead_microbench_rows_cover_train_only_and_sampling_only():
    from scripts.current.long_k_overhead_microbench_manifest import build_rows

    rows = build_rows()
    exp_ids = {row["exp_id"] for row in rows}

    assert exp_ids == {
        "M0_ddp_train_only_K400_b64_s2k_seed1",
        "M0_sampling_only_K400_b64_s2k_seed1",
        "M0_ddp_train_only_K400_b256_s2k_seed1",
        "M0_sampling_only_K400_b256_s2k_seed1",
    }
    assert {row["microbench_mode"] for row in rows} == {"ddp_train_only", "sampling_only"}
    assert {row["submit"] for row in rows} == {"yes"}


def test_overhead_microbench_rows_use_original_cifar_k400_without_drl():
    from scripts.current.long_k_overhead_microbench_manifest import build_rows

    rows = build_rows()
    for row in rows:
        assert row["K"] == "400"
        assert row["benchmark"] == "cifar10"
        assert row["uses_drl_backbone"] == "false"
        assert row["uses_imagenet32"] == "false"
        assert row["world_size"] == "8"
        assert row["num_nodes"] == "2"
        assert row["ppn"] == "4"


def test_overhead_microbench_materialize_writes_expected_files(tmp_path):
    from scripts.current.long_k_overhead_microbench_manifest import build_rows, materialize

    rows = materialize(build_rows(), tmp_path, Path("/repo"))

    assert (tmp_path / "summaries" / "overhead_microbench_master_registry.csv").exists()
    assert (tmp_path / "summaries" / "overhead_microbench_submitted.csv").exists() is False
    for row in rows:
        assert Path(row["config_path"]).exists()
        assert Path(row["pbs_path"]).exists()
        assert Path(row["manifest_row_path"]).exists()
        pbs = Path(row["pbs_path"]).read_text()
        assert "ebm_overhead_microbench.py" in pbs
        assert "MICROBENCH_MODE" in pbs
