import csv
import json
import subprocess
import sys
from pathlib import Path

from test_submit_ablation_phase1 import manifest_row, run_submitter, write_csv


PROJECT_DIR = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")
COLLECT_SCRIPT = PROJECT_DIR / "scripts" / "current" / "collect_ablation_phase1.py"
PHASE2_SUBMIT_WRAPPER = PROJECT_DIR / "scripts" / "current" / "submit_ablation_phase2.sh"
PHASE2_COLLECT_WRAPPER = PROJECT_DIR / "scripts" / "current" / "collect_ablation_phase2.py"
PHASE2_LOCAL_EVAL_WRAPPER = PROJECT_DIR / "scripts" / "current" / "run_local_eval_phase2.py"


def submitted_fieldnames():
    template = PROJECT_DIR / "experiments" / "ablation_manifest_phase1_submitted.csv"
    with template.open("r", newline="") as f:
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


def test_submitter_supports_custom_runs_root_and_payload_subdir(tmp_path):
    rows = [manifest_row("A3_ddp_fullk_k100_s20k")]
    manifest_path = tmp_path / "manifest.csv"
    submitted_path = tmp_path / "submitted.csv"
    runs_root = tmp_path / "runs_phase2"
    payload_subdir = "phase2_test_support"
    payload_path = PROJECT_DIR / "experiments" / "_submission_payloads" / payload_subdir / "01_A3_ddp_fullk_k100_s20k.json"
    if payload_path.exists():
        payload_path.unlink()
    write_csv(manifest_path, rows)

    result = run_submitter(
        tmp_path=tmp_path,
        manifest_path=manifest_path,
        submitted_path=submitted_path,
        extra_args=[
            "--eval-mode",
            "local",
            "--runs-root",
            str(runs_root),
            "--payload-subdir",
            payload_subdir,
        ],
    )

    assert result.returncode == 0, result.stderr
    assert payload_path.exists()

    with submitted_path.open("r", newline="") as f:
        out_rows = list(csv.DictReader(f))

    assert len(out_rows) == 1
    assert out_rows[0]["train_run_dir"].startswith(str(runs_root))
    assert out_rows[0]["eval_run_dir"].startswith(str(runs_root))


def test_collector_supports_custom_summary_prefix_and_renderer(tmp_path):
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True)
    (run_dir / "job_info.json").write_text(json.dumps({"train_status": "succeeded", "total_walltime_seconds": 12.0}))
    (eval_dir / "metrics_compare.json").write_text(
        json.dumps(
            {
                "fid_inception_baseline_vs_real": 12.34,
                "fid_feature_baseline_vs_real": 0.56,
                "unique_ratio": 0.78,
                "distinct_images": 3900,
                "diversity_trace": 1.23,
            }
        )
    )
    (eval_dir / "stats.json").write_text(
        json.dumps(
            {
                "max_f_neg_last_1k_mean": 1.0,
                "max_abs_chain_last_1k_mean": 2.0,
                "fneg_stage2_over_stage3_last_1k_mean": 0.5,
                "total_walltime_seconds": 34.0,
            }
        )
    )

    submitted = tmp_path / "submitted.csv"
    write_submitted(
        submitted,
        [
            {
                "submit": "1",
                "group": "M1",
                "exp_id": "phase2_case",
                "story": "phase2",
                "train_mode": "ddp_fullk",
                "world_size": "4",
                "pipe_stages": "0",
                "K": "100",
                "steps": "300000",
                "lr": "1e-4",
                "step_size": "1.0",
                "weight_mode": "uniform",
                "last2_beta": "0.01",
                "train_run_dir": str(run_dir),
                "eval_run_dir": str(eval_dir),
            }
        ],
    )

    render_script = tmp_path / "render.py"
    render_script.write_text(
        """#!/usr/bin/env python3
import argparse
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument('--summary_csv', required=True)
ap.add_argument('--out_md', required=True)
args = ap.parse_args()
Path(args.out_md).write_text('phase2 custom report\\n')
"""
    )

    out_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(COLLECT_SCRIPT),
            "--submitted_manifest",
            str(submitted),
            "--out_dir",
            str(out_dir),
            "--summary-prefix",
            "phase2_summary",
            "--render-script",
            str(render_script),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    assert result.returncode == 0, result.stderr
    assert (out_dir / "phase2_summary.csv").exists()
    assert (out_dir / "phase2_summary.json").exists()
    assert (out_dir / "phase2_summary.md").exists()
    assert "phase2 custom report" in (out_dir / "phase2_summary.md").read_text()


def test_phase2_wrappers_exist_and_point_to_phase2_defaults():
    submit_text = PHASE2_SUBMIT_WRAPPER.read_text()
    assert "ablation_manifest_phase2.csv" in submit_text
    assert "ablation_manifest_phase2_submitted.csv" in submit_text
    assert "--payload-subdir" in submit_text
    assert "phase2" in submit_text

    collect_text = PHASE2_COLLECT_WRAPPER.read_text()
    assert "phase2_summary" in collect_text
    assert "ablation_manifest_phase2_submitted.csv" in collect_text

    eval_text = PHASE2_LOCAL_EVAL_WRAPPER.read_text()
    assert "ablation_manifest_phase2_submitted.csv" in eval_text
