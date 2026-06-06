from polaris_ebm.scripts.current.ablation_500k_manifest import build_500k_rows


def test_build_500k_rows_returns_requested_seven_cases():
    rows = build_500k_rows()

    exp_ids = [row["exp_id"] for row in rows]
    assert exp_ids == [
        "single_fullk_K100_seed1_500k",
        "ddp_fullk_K100_seed1_500k",
        "ddp_fullk_K100_seed2_500k",
        "ddp_fullk_K100_seed3_500k",
        "pipe_strict_P4_K100_beta001_lr1e4_seed1_500k",
        "pipe_strict_P4_K100_beta001_lr1e4_seed2_500k",
        "pipe_strict_P4_K100_beta001_lr1e4_seed3_500k",
    ]
    assert all(row["steps"] == "500000" for row in rows)
    assert all(row["save_every"] == "5000" for row in rows)
    assert all(row["queue"] == "preemptable" for row in rows)
    assert [row["seed"] for row in rows] == ["1", "1", "2", "3", "1", "2", "3"]
