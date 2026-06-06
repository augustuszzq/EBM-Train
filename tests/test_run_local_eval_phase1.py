import csv
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/current/run_local_eval_phase1.py")
SUBMITTED_TEMPLATE = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/experiments/ablation_manifest_phase1_submitted.csv")


def submitted_fieldnames():
    with SUBMITTED_TEMPLATE.open("r", newline="") as f:
        return list(csv.DictReader(f).fieldnames)


def write_submitted(path, rows):
    fieldnames = submitted_fieldnames()
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {name: "" for name in fieldnames}
            out.update(row)
            writer.writerow(out)


def test_local_eval_dry_run_selects_only_finished_runs_without_metrics(tmp_path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    (run_a / "eval").mkdir(parents=True)
    (run_b / "eval").mkdir(parents=True)
    (run_a / "job_info.json").write_text(json.dumps({"train_status": "succeeded"}))
    (run_b / "job_info.json").write_text(json.dumps({"train_status": "succeeded"}))
    (run_b / "eval" / "metrics_compare.json").write_text("{}")

    submitted = tmp_path / "submitted.csv"
    write_submitted(
        submitted,
        [
            {
                "submit": "1",
                "group": "A4",
                "exp_id": "needs_eval",
                "train_run_dir": str(run_a),
                "eval_run_dir": str(run_a / "eval"),
            },
            {
                "submit": "1",
                "group": "A4",
                "exp_id": "already_done",
                "train_run_dir": str(run_b),
                "eval_run_dir": str(run_b / "eval"),
            },
        ],
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--submitted-manifest",
            str(submitted),
            "--dry-run",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    assert result.returncode == 0, result.stderr
    assert "needs_eval" in result.stdout
    assert "already_done" not in result.stdout
