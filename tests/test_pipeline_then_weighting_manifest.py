import csv
from pathlib import Path


MANIFEST = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm/experiments/pipeline_then_weighting_manifest.csv")


def load_rows():
    with MANIFEST.open("r", newline="") as f:
        return list(csv.DictReader(f))


def test_pipeline_then_weighting_manifest_covers_e1_and_e2():
    rows = load_rows()
    assert len(rows) == 24
    assert {row["phase"] for row in rows} == {"E1", "E2"}


def test_e1_rows_are_submitted_and_e2_rows_are_prepared_only():
    rows = load_rows()
    e1 = [row for row in rows if row["phase"] == "E1"]
    e2 = [row for row in rows if row["phase"] == "E2"]
    assert len(e1) == 10
    assert len(e2) == 14
    assert all(row["submit"] == "yes" for row in e1)
    assert all(row["submit"] == "no" for row in e2)


def test_e1_manifest_covers_requested_stage_counts():
    rows = load_rows()
    single_p = sorted(
        int(row["pipe_stages"])
        for row in rows
        if row["exp_id"].startswith("E1_single_pipe_emul_")
    )
    pipe_p = sorted(
        int(row["pipe_stages"])
        for row in rows
        if row["exp_id"].startswith("E1_pipe_strict_")
    )
    assert single_p == [2, 4, 8, 16]
    assert pipe_p == [2, 4, 8, 16]
