#!/usr/bin/env python3
"""Evaluate FID trajectories across saved checkpoints for a single run."""

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_DIR = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")
DEFAULT_DATA_DIR = "/eagle/lc-mpi/Zhiqing/ebm/data/cifar10"
DEFAULT_EVAL_PYTHON = "/home/kevienzzq/.conda/envs/llm-env/bin/python"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Evaluate checkpoint FID trajectory for one run")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--step-interval", type=int, default=5000)
    ap.add_argument("--eval-images", type=int, default=5000)
    ap.add_argument("--eval-batch", type=int, default=256)
    ap.add_argument("--k-eval", type=int, default=100)
    ap.add_argument("--step-size", type=float, default=1.0)
    ap.add_argument("--noise-std", type=float, default=0.01)
    ap.add_argument("--langevin-sign", type=float, default=1.0)
    ap.add_argument("--eval-seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--skip-existing", action="store_true")
    return ap.parse_args()


def parse_step_from_name(path: Path) -> int:
    name = path.name
    if not name.startswith("ckpt_step") or not name.endswith(".pt"):
        raise ValueError("unsupported checkpoint name: %s" % name)
    return int(name[len("ckpt_step") : -len(".pt")])


def candidate_checkpoint_dirs(run_dir: Path) -> List[Path]:
    return [
        run_dir / "artifacts" / "checkpoints",
        run_dir / "checkpoints",
    ]


def discover_trajectory_checkpoints(run_dir: Path, step_interval: int = 5000) -> List[Dict]:
    if step_interval <= 0:
        raise ValueError("step_interval must be > 0")
    ckpt_dir = None
    for candidate in candidate_checkpoint_dirs(run_dir):
        if candidate.exists():
            ckpt_dir = candidate
            break
    if ckpt_dir is None:
        raise RuntimeError("no checkpoint directory found under %s" % run_dir)

    items = []
    for path in ckpt_dir.glob("ckpt_step*.pt"):
        step = parse_step_from_name(path)
        ordinal_step = step + 1
        if ordinal_step % step_interval != 0:
            continue
        items.append(
            {
                "step": step,
                "ordinal_step": ordinal_step,
                "ckpt_path": str(path),
            }
        )
    items.sort(key=lambda item: item["step"])
    return items


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r") as f:
        return json.load(f)


def resolve_eval_python() -> str:
    explicit = os.environ.get("EVAL_PYTHON", "").strip()
    if explicit:
        return explicit
    if os.path.exists(DEFAULT_EVAL_PYTHON):
        return DEFAULT_EVAL_PYTHON
    return sys.executable


def find_reference_final_fid(run_dir: Path) -> Optional[float]:
    metric_paths = sorted((run_dir / "eval").glob("metrics_compare*.json"))
    if not metric_paths:
        return None
    latest = metric_paths[-1]
    payload = load_json(latest, default={}) or {}
    return payload.get("fid_inception_baseline_vs_real")


def threshold_step(rows: List[Dict], threshold: float) -> Optional[int]:
    for row in rows:
        value = row.get("fid_inception")
        if value is not None and value < threshold:
            return row["ordinal_step"]
    return None


def summarize_trajectory_rows(rows: List[Dict], reference_final_fid: Optional[float] = None) -> Dict:
    if not rows:
        raise ValueError("rows must be non-empty")
    best = min(rows, key=lambda row: row["fid_inception"])
    final = rows[-1]
    summary = {
        "num_points": len(rows),
        "best_step": best["ordinal_step"],
        "best_fid": best["fid_inception"],
        "final_step": final["ordinal_step"],
        "final_fid": final["fid_inception"],
        "first_sub100_step": threshold_step(rows, 100.0),
        "first_sub70_step": threshold_step(rows, 70.0),
        "first_sub60_step": threshold_step(rows, 60.0),
        "reference_final_fid": reference_final_fid,
        "final_match_delta": None,
    }
    if reference_final_fid is not None:
        summary["final_match_delta"] = abs(float(final["fid_inception"]) - float(reference_final_fid))
    return summary


def run_cmd(cmd: List[str], env: Dict[str, str]) -> None:
    subprocess.run(cmd, check=True, env=env)


def eval_one_checkpoint(
    ckpt_path: Path,
    eval_dir: Path,
    env: Dict[str, str],
    args: argparse.Namespace,
) -> Dict:
    metrics_path = eval_dir / "metrics_compare.json"
    stats_path = eval_dir / "stats.json"
    grid_path = eval_dir / "grid.png"
    samples_path = eval_dir / "samples.pt"
    tmp_dir = eval_dir / "_tmp_generate"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing and metrics_path.exists() and stats_path.exists():
        metrics = load_json(metrics_path, default={}) or {}
        return {
            "fid_inception": metrics.get("fid_inception_baseline_vs_real"),
            "fid_feature": metrics.get("fid_feature_baseline_vs_real"),
            "unique_ratio": metrics.get("unique_ratio"),
            "metrics_path": str(metrics_path),
            "stats_path": str(stats_path),
            "grid_path": str(grid_path),
            "samples_path": str(samples_path),
            "eval_dir": str(eval_dir),
        }

    if tmp_dir.exists():
        shutil.rmtree(str(tmp_dir))
    tmp_dir.mkdir(parents=True)
    py = env["EVAL_PYTHON"]

    gen_cmd = [
        py,
        str(PROJECT_DIR / "scripts" / "current" / "eval_generate.py"),
        "--ckpt",
        str(ckpt_path),
        "--out_dir",
        str(tmp_dir),
        "--num_images",
        str(args.eval_images),
        "--batch",
        str(args.eval_batch),
        "--K_eval",
        str(args.k_eval),
        "--langevin_sign",
        str(args.langevin_sign),
        "--step_size",
        str(args.step_size),
        "--noise_std",
        str(args.noise_std),
        "--seed",
        str(args.eval_seed),
        "--device",
        str(args.device),
        "--no_clamp_x",
    ]
    metrics_cmd = [
        py,
        str(PROJECT_DIR / "scripts" / "current" / "eval_metrics.py"),
        "--baseline_samples",
        str(tmp_dir / "samples.pt"),
        "--pipeline_samples",
        str(tmp_dir / "samples.pt"),
        "--data_dir",
        str(args.data_dir),
        "--num_real",
        str(args.eval_images),
        "--batch",
        str(args.eval_batch),
        "--out_json",
        str(metrics_path),
        "--device",
        str(args.device),
    ]
    run_cmd(gen_cmd, env=env)
    shutil.copy2(str(tmp_dir / "grid.png"), str(grid_path))
    shutil.copy2(str(tmp_dir / "samples.pt"), str(samples_path))
    run_cmd(metrics_cmd, env=env)
    shutil.copy2(str(tmp_dir / "stats.json"), str(stats_path))
    shutil.rmtree(str(tmp_dir))

    metrics = load_json(metrics_path, default={}) or {}
    return {
        "fid_inception": metrics.get("fid_inception_baseline_vs_real"),
        "fid_feature": metrics.get("fid_feature_baseline_vs_real"),
        "unique_ratio": metrics.get("unique_ratio"),
        "metrics_path": str(metrics_path),
        "stats_path": str(stats_path),
        "grid_path": str(grid_path),
        "samples_path": str(samples_path),
        "eval_dir": str(eval_dir),
    }


def render_markdown(label: str, rows: List[Dict], summary: Dict) -> str:
    lines = [
        "# %s FID Trajectory" % label,
        "",
        "- points: %d" % summary["num_points"],
        "- best: step %s -> %.6f" % (summary["best_step"], summary["best_fid"]),
        "- final: step %s -> %.6f" % (summary["final_step"], summary["final_fid"]),
    ]
    if summary.get("reference_final_fid") is not None:
        lines.append(
            "- final_match_delta_vs_recorded: %.6f" % float(summary["final_match_delta"])
        )
    lines.extend(
        [
            "",
            "| step | fid_inception | fid_feature | unique_ratio |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| %d | %.6f | %.6f | %s |"
            % (
                row["ordinal_step"],
                float(row["fid_inception"]),
                float(row["fid_feature"]),
                "NA" if row.get("unique_ratio") is None else str(row.get("unique_ratio")),
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_rows_csv(path: Path, rows: List[Dict]) -> None:
    fieldnames = [
        "label",
        "step",
        "ordinal_step",
        "ckpt_path",
        "fid_inception",
        "fid_feature",
        "unique_ratio",
        "eval_dir",
        "metrics_path",
        "stats_path",
        "grid_path",
        "samples_path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    label = args.label or run_dir.name
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    project_parent = str(PROJECT_DIR.parent)
    env["PYTHONPATH"] = project_parent + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["EVAL_PYTHON"] = resolve_eval_python()

    ckpts = discover_trajectory_checkpoints(run_dir, step_interval=args.step_interval)
    reference_final_fid = find_reference_final_fid(run_dir)
    rows = []
    for item in ckpts:
        step_dir = out_dir / ("step%06d" % item["ordinal_step"])
        result = eval_one_checkpoint(Path(item["ckpt_path"]), step_dir, env, args)
        row = {
            "label": label,
            "step": item["step"],
            "ordinal_step": item["ordinal_step"],
            "ckpt_path": item["ckpt_path"],
        }
        row.update(result)
        rows.append(row)
        print(
            "[TRAJ] %s step=%d fid=%.6f"
            % (label, item["ordinal_step"], float(row["fid_inception"])),
            flush=True,
        )

    summary = summarize_trajectory_rows(rows, reference_final_fid=reference_final_fid)
    summary["label"] = label
    summary["run_dir"] = str(run_dir)
    summary["out_dir"] = str(out_dir)

    write_rows_csv(out_dir / "trajectory.csv", rows)
    with (out_dir / "trajectory.json").open("w") as f:
        json.dump(rows, f, indent=2)
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    (out_dir / "trajectory.md").write_text(render_markdown(label, rows, summary))
    print("[DONE] label=%s out_dir=%s" % (label, out_dir), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
