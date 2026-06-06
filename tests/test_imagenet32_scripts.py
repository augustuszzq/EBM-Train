import csv
import json
import subprocess
import sys
from pathlib import Path

from polaris_ebm.scripts.current.benchmark_config import load_benchmark_config


PROJECT_DIR = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")


def test_imagenet32_config_files_exist_and_load():
    for name in [
        "imagenet32_single_strict.yaml",
        "imagenet32_ddp_strict.yaml",
        "imagenet32_pipeline_strict.yaml",
        "imagenet32_ablation.yaml",
    ]:
        cfg = load_benchmark_config(PROJECT_DIR / "configs" / name)
        assert cfg.benchmark.name == "imagenet32"
        assert cfg.benchmark.conditional is True


def test_aggregate_imagenet32_results_writes_summary_files(tmp_path: Path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    (run_a / "eval").mkdir(parents=True)
    (run_b / "eval").mkdir(parents=True)
    (run_a / "eval" / "metrics_compare.json").write_text(
        json.dumps({"final_fid": 12.3, "best_fid": 10.1, "seed": 1, "exp_id": "a"})
    )
    (run_b / "eval" / "metrics_compare.json").write_text(
        json.dumps({"final_fid": 13.4, "best_fid": 11.2, "seed": 2, "exp_id": "a"})
    )
    (run_a / "eval" / "conditional_acc.json").write_text(
        json.dumps({"top1_conditional_acc": 0.7, "per_class_avg_acc": 0.6})
    )

    out_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_DIR / "scripts" / "aggregate_imagenet32_results.py"),
            "--runs-root",
            str(tmp_path),
            "--out-dir",
            str(out_dir),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )

    assert result.returncode == 0, result.stderr
    assert (out_dir / "imagenet32_summary.csv").exists()
    assert (out_dir / "imagenet32_summary.json").exists()
    with (out_dir / "imagenet32_summary.csv").open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["exp_id"] == "a"
    assert rows[0]["top1_conditional_acc"] == "0.7"
