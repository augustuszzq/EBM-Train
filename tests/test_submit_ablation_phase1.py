import csv
import os
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")
SUBMIT_SCRIPT = PROJECT_DIR / "scripts" / "current" / "submit_ablation_phase1.sh"
MANIFEST_TEMPLATE = PROJECT_DIR / "experiments" / "ablation_manifest_phase1.csv"
SUBMITTED_TEMPLATE = PROJECT_DIR / "experiments" / "ablation_manifest_phase1_submitted.csv"


def load_template_rows():
    with MANIFEST_TEMPLATE.open("r", newline="") as f:
        return list(csv.DictReader(f))


def manifest_row(exp_id):
    for row in load_template_rows():
        if row["exp_id"] == exp_id:
            return dict(row)
    raise AssertionError(f"missing exp_id {exp_id}")


def submitted_fieldnames():
    with SUBMITTED_TEMPLATE.open("r", newline="") as f:
        return list(csv.DictReader(f).fieldnames)


def to_submitted_row(row, **overrides):
    out = {name: "" for name in submitted_fieldnames()}
    out.update(row)
    out.update(overrides)
    return out


def write_csv(path, rows):
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def install_fake_qsub(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    qsub = bin_dir / "qsub"
    qsub.write_text(
        """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

log_path = Path(os.environ["FAKE_QSUB_LOG"])
with log_path.open("a") as f:
    f.write(" ".join(sys.argv[1:]) + "\\n")

state_path = Path(os.environ["FAKE_QSUB_STATE"])
counter = int(state_path.read_text()) if state_path.exists() else 100
counter += 1
state_path.write_text(str(counter))

job_name = ""
if "-N" in sys.argv:
    job_name = sys.argv[sys.argv.index("-N") + 1]

if os.environ.get("FAKE_QSUB_FAIL_EVAL") == "1" and job_name.startswith("abev"):
    print("qsub: fake eval failure", file=sys.stderr)
    sys.exit(38)

print(f"{counter}.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov")
"""
    )
    qsub.chmod(0o755)
    return bin_dir


def run_submitter(tmp_path, manifest_path, submitted_path, extra_args=None, fail_eval=False):
    extra_args = extra_args or []
    bin_dir = install_fake_qsub(tmp_path)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["FAKE_QSUB_LOG"] = str(tmp_path / "qsub.log")
    env["FAKE_QSUB_STATE"] = str(tmp_path / "qsub_state.txt")
    if fail_eval:
        env["FAKE_QSUB_FAIL_EVAL"] = "1"
    cmd = [
        "bash",
        str(SUBMIT_SCRIPT),
        "--manifest",
        str(manifest_path),
        "--submitted-manifest",
        str(submitted_path),
    ] + list(extra_args)
    return subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)


def read_qsub_log(tmp_path):
    path = tmp_path / "qsub.log"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def test_submitter_eval_mode_local_preserves_existing_rows(tmp_path):
    rows = [
        manifest_row("A3_ddp_fullk_k100_s20k"),
        manifest_row("A3_pipe_last2b001_p4_k100_s20k"),
    ]
    manifest_path = tmp_path / "manifest.csv"
    write_csv(manifest_path, rows)

    submitted_rows = [
        to_submitted_row(
            rows[0],
            submitted_at="2026-03-29T00:00:00Z",
            train_run_dir="/tmp/existing_run",
            eval_run_dir="/tmp/existing_run/eval",
            train_job_id="111.polaris",
            eval_job_id="222.polaris",
        ),
        to_submitted_row(rows[1]),
    ]
    submitted_path = tmp_path / "submitted.csv"
    write_csv(submitted_path, submitted_rows)

    result = run_submitter(
        tmp_path=tmp_path,
        manifest_path=manifest_path,
        submitted_path=submitted_path,
        extra_args=["--eval-mode", "local"],
    )

    assert result.returncode == 0, result.stderr
    assert len(read_qsub_log(tmp_path)) == 1

    with submitted_path.open("r", newline="") as f:
        out_rows = list(csv.DictReader(f))

    assert out_rows[0]["train_job_id"] == "111.polaris"
    assert out_rows[0]["eval_job_id"] == "222.polaris"
    assert out_rows[1]["train_job_id"].endswith(".polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov")
    assert out_rows[1]["eval_job_id"] == ""


def test_submitter_persists_train_job_id_when_eval_submission_fails(tmp_path):
    rows = [manifest_row("A4_ddp_fullk_eps010_k100_s20k")]
    manifest_path = tmp_path / "manifest.csv"
    write_csv(manifest_path, rows)
    submitted_path = tmp_path / "submitted.csv"

    result = run_submitter(
        tmp_path=tmp_path,
        manifest_path=manifest_path,
        submitted_path=submitted_path,
        extra_args=["--eval-mode", "pbs"],
        fail_eval=True,
    )

    assert result.returncode != 0
    assert submitted_path.exists()

    with submitted_path.open("r", newline="") as f:
        out_rows = list(csv.DictReader(f))

    assert len(out_rows) == 1
    assert out_rows[0]["train_job_id"].endswith(".polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov")
    assert out_rows[0]["eval_job_id"] == ""
