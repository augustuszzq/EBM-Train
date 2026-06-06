from polaris_ebm.scripts.current.imagenet32_phasea_manifest import build_phasea_rows


def test_imagenet32_phasea_manifest_covers_three_modes_and_three_horizons():
    rows = build_phasea_rows()

    assert len(rows) == 9

    by_exp_id = {row["exp_id"]: row for row in rows}
    assert "imagenet32_single_strict_k100_s1k" in by_exp_id
    assert "imagenet32_ddp_strict_k100_s5k" in by_exp_id
    assert "imagenet32_pipeline_strict_p4_k100_s20k" in by_exp_id

    horizons = sorted({int(row["steps"]) for row in rows})
    assert horizons == [1000, 5000, 20000]

    assert {row["train_mode"] for row in rows} == {"single_fullk", "ddp_fullk", "pipe_strict"}
    assert all(row["benchmark"] == "imagenet32" for row in rows)
    assert all(row["queue"] == "preemptable" for row in rows)


def test_imagenet32_phasea_manifest_points_pipeline_rows_at_pipeline_config():
    rows = build_phasea_rows()
    pipeline_rows = [row for row in rows if row["train_mode"] == "pipe_strict"]

    assert len(pipeline_rows) == 3
    assert all(row["config"].endswith("configs/imagenet32_pipeline_strict.yaml") for row in pipeline_rows)
    assert all(row["weight_mode"] == "last2_beta" for row in pipeline_rows)
    assert all(row["last2_beta"] == "0.01" for row in pipeline_rows)
