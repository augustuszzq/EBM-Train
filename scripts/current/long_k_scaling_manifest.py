#!/usr/bin/env python3
"""Prepare and optionally submit original-CIFAR Long-K scaling experiments."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    from ablation_common import compute_k_slices
except Exception:
    from polaris_ebm.scripts.current.ablation_common import compute_k_slices


PROJECT_DIR = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")
BUNDLE_ROOT = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_long_k_scaling")
DATA_DIR = Path("/eagle/lc-mpi/Zhiqing/ebm/data/cifar10")

FIELDS = [
    "phase",
    "group",
    "exp_id",
    "story",
    "train_mode",
    "world_size",
    "pipe_stages",
    "stage_slices",
    "K",
    "steps",
    "lr",
    "step_size",
    "noise_std",
    "weight_mode",
    "last2_beta",
    "seed",
    "num_nodes",
    "ppn",
    "batch_size",
    "save_every",
    "vis_every",
    "eval_images",
    "queue",
    "train_walltime",
    "data_dir",
    "submit",
    "paper_role",
    "notes",
    "config_path",
    "pbs_path",
    "run_dir",
    "pbs_output_log",
]


def _stage_slices(k: int, p: int) -> str:
    return ";".join(str(x) for x in compute_k_slices(k=int(k), pipe_stages=int(p)))


def _row(
    *,
    phase: str,
    exp_id: str,
    train_mode: str,
    k: int,
    steps: int,
    step_size: str,
    pipe_stages: int,
    world_size: int,
    num_nodes: int,
    ppn: int,
    seed: int,
    submit: bool,
    paper_role: str,
    walltime: str,
    notes: str = "",
) -> dict[str, str]:
    weight_mode = "uniform" if pipe_stages > 1 else "deep_only"
    return {
        "phase": phase,
        "group": "long_k_scaling",
        "exp_id": exp_id,
        "story": "original CIFAR EBM long-K scaling; no DRL backbone and no ImageNet-32",
        "train_mode": train_mode,
        "world_size": str(world_size),
        "pipe_stages": str(pipe_stages),
        "stage_slices": _stage_slices(k, pipe_stages),
        "K": str(k),
        "steps": str(steps),
        "lr": "1e-4",
        "step_size": step_size,
        "noise_std": "0.01",
        "weight_mode": weight_mode,
        "last2_beta": "0.01",
        "seed": str(seed),
        "num_nodes": str(num_nodes),
        "ppn": str(ppn),
        "batch_size": "64",
        "save_every": "5000",
        "vis_every": "5000",
        "eval_images": "5000",
        "queue": "preemptable",
        "train_walltime": walltime,
        "data_dir": str(DATA_DIR),
        "submit": "yes" if submit else "no",
        "paper_role": paper_role,
        "notes": notes,
        "config_path": "",
        "pbs_path": "",
        "run_dir": "",
        "pbs_output_log": "",
    }


def build_long_k_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    rows.extend(
        [
            _row(
                phase="L0",
                exp_id="L0_ddp_fullk_K200_s20k_seed1",
                train_mode="ddp_fullk",
                k=200,
                steps=20000,
                step_size="0.005",
                pipe_stages=1,
                world_size=8,
                num_nodes=2,
                ppn=4,
                seed=1,
                submit=True,
                paper_role="smoke",
                walltime="24:00:00",
            ),
            _row(
                phase="L0",
                exp_id="L0_pipe_P8_equal_K200_s20k_seed1",
                train_mode="pipe_strict",
                k=200,
                steps=20000,
                step_size="0.005",
                pipe_stages=8,
                world_size=8,
                num_nodes=2,
                ppn=4,
                seed=1,
                submit=True,
                paper_role="smoke",
                walltime="24:00:00",
            ),
            _row(
                phase="L0",
                exp_id="L0_ddp_fullk_K400_s20k_seed1",
                train_mode="ddp_fullk",
                k=400,
                steps=20000,
                step_size="0.0025",
                pipe_stages=1,
                world_size=8,
                num_nodes=2,
                ppn=4,
                seed=1,
                submit=True,
                paper_role="smoke",
                walltime="36:00:00",
            ),
            _row(
                phase="L0",
                exp_id="L0_pipe_P8_equal_K400_s20k_seed1",
                train_mode="pipe_strict",
                k=400,
                steps=20000,
                step_size="0.0025",
                pipe_stages=8,
                world_size=8,
                num_nodes=2,
                ppn=4,
                seed=1,
                submit=True,
                paper_role="smoke",
                walltime="36:00:00",
            ),
            _row(
                phase="L0",
                exp_id="L0_single_fullk_K400_s20k_seed1",
                train_mode="single_fullk",
                k=400,
                steps=20000,
                step_size="0.0025",
                pipe_stages=1,
                world_size=1,
                num_nodes=1,
                ppn=1,
                seed=1,
                submit=True,
                paper_role="smoke_control",
                walltime="36:00:00",
            ),
            _row(
                phase="L0",
                exp_id="L0_single_emul_P8_equal_K400_s20k_seed1",
                train_mode="single_pipe_emul",
                k=400,
                steps=20000,
                step_size="0.0025",
                pipe_stages=8,
                world_size=1,
                num_nodes=1,
                ppn=1,
                seed=1,
                submit=True,
                paper_role="smoke_control",
                walltime="36:00:00",
            ),
        ]
    )

    for k in (200, 400):
        for train_mode, p, label in (
            ("ddp_fullk", 1, "ddp_fullk"),
            ("pipe_strict", 8, "pipe_P8_equal"),
        ):
            rows.append(
                _row(
                    phase="L0_optional_same_step",
                    exp_id=f"L0opt_{label}_K{k}_stepsize001_s20k_seed1",
                    train_mode=train_mode,
                    k=k,
                    steps=20000,
                    step_size="0.01",
                    pipe_stages=p,
                    world_size=8,
                    num_nodes=2,
                    ppn=4,
                    seed=1,
                    submit=False,
                    paper_role="optional_smoke",
                    walltime="36:00:00",
                    notes="same-step-size smoke only; do not run long horizon without explicit approval",
                )
            )

    for seed in (1, 2, 3):
        rows.append(
            _row(
                phase="L1",
                exp_id=f"L1_ddp_fullk_K400_s300k_seed{seed}",
                train_mode="ddp_fullk",
                k=400,
                steps=300000,
                step_size="0.0025",
                pipe_stages=1,
                world_size=8,
                num_nodes=2,
                ppn=4,
                seed=seed,
                submit=False,
                paper_role="main_300k",
                walltime="36:00:00",
            )
        )
        rows.append(
            _row(
                phase="L1",
                exp_id=f"L1_pipe_P8_equal_K400_s300k_seed{seed}",
                train_mode="pipe_strict",
                k=400,
                steps=300000,
                step_size="0.0025",
                pipe_stages=8,
                world_size=8,
                num_nodes=2,
                ppn=4,
                seed=seed,
                submit=False,
                paper_role="main_300k",
                walltime="36:00:00",
            )
        )
    for train_mode, p, label in (
        ("single_fullk", 1, "single_fullk"),
        ("single_pipe_emul", 8, "single_emul_P8_equal"),
    ):
        rows.append(
            _row(
                phase="L1",
                exp_id=f"L1_{label}_K400_s300k_seed1",
                train_mode=train_mode,
                k=400,
                steps=300000,
                step_size="0.0025",
                pipe_stages=p,
                world_size=1,
                num_nodes=1,
                ppn=1,
                seed=1,
                submit=False,
                paper_role="main_300k_single_control",
                walltime="36:00:00",
            )
        )

    for seed in (1, 2, 3):
        rows.append(
            _row(
                phase="L2",
                exp_id=f"L2_ddp_fullk_K400_s500k_seed{seed}",
                train_mode="ddp_fullk",
                k=400,
                steps=500000,
                step_size="0.0025",
                pipe_stages=1,
                world_size=8,
                num_nodes=2,
                ppn=4,
                seed=seed,
                submit=False,
                paper_role="main_500k",
                walltime="36:00:00",
            )
        )
        rows.append(
            _row(
                phase="L2",
                exp_id=f"L2_pipe_P8_equal_K400_s500k_seed{seed}",
                train_mode="pipe_strict",
                k=400,
                steps=500000,
                step_size="0.0025",
                pipe_stages=8,
                world_size=8,
                num_nodes=2,
                ppn=4,
                seed=seed,
                submit=False,
                paper_role="main_500k",
                walltime="36:00:00",
            )
        )
    for train_mode, p, label in (
        ("single_fullk", 1, "single_fullk"),
        ("single_pipe_emul", 8, "single_emul_P8_equal"),
    ):
        rows.append(
            _row(
                phase="L2_optional",
                exp_id=f"L2opt_{label}_K400_s500k_seed1",
                train_mode=train_mode,
                k=400,
                steps=500000,
                step_size="0.0025",
                pipe_stages=p,
                world_size=1,
                num_nodes=1,
                ppn=1,
                seed=1,
                submit=False,
                paper_role="optional_500k_single_control",
                walltime="36:00:00",
            )
        )
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def config_text(row: dict[str, str]) -> str:
    stage_weights = "[%s]" % ", ".join(["1.0"] * int(row["pipe_stages"]))
    return "\n".join(
        [
            "benchmark:",
            "  name: cifar10",
            "  conditional: false",
            "  num_classes: 0",
            "  image_size: 32",
            f"  data_root: {row['data_dir']}",
            "  split_train: train",
            "  split_val: train",
            "train:",
            f"  steps: {row['steps']}",
            f"  batch_size: {row['batch_size']}",
            f"  K: {row['K']}",
            f"  lr: {row['lr']}",
            "pipeline:",
            f"  pipe_stages: {row['pipe_stages']}",
            f"  weight_mode: {row['weight_mode']}",
            f"  last2_beta: {row['last2_beta']}",
            f"  stage_weights: {stage_weights}",
            "long_k_scaling:",
            f"  exp_id: {row['exp_id']}",
            f"  stage_slices: [{', '.join(row['stage_slices'].split(';'))}]",
            f"  step_size: {row['step_size']}",
            f"  noise_std: {row['noise_std']}",
            "  method: original_cifar_completion_aware_diagonal_objective",
            "  uses_drl_backbone: false",
            "  uses_imagenet32: false",
            "",
        ]
    )


def pbs_text(row: dict[str, str], project_dir: Path) -> str:
    env = {
        "PROJECT_DIR": str(project_dir),
        "GROUP": row["group"],
        "EXP_ID": row["exp_id"],
        "STORY": row["story"],
        "TRAIN_MODE": row["train_mode"],
        "WORLD_SIZE": row["world_size"],
        "PIPE_STAGES": row["pipe_stages"],
        "K": row["K"],
        "STEPS": row["steps"],
        "LR": row["lr"],
        "STEP_SIZE": row["step_size"],
        "WEIGHT_MODE": row["weight_mode"],
        "LAST2_BETA": row["last2_beta"],
        "SEED": row["seed"],
        "NUM_NODES": row["num_nodes"],
        "PPN": row["ppn"],
        "EVAL_IMAGES": row["eval_images"],
        "NOTES": row["notes"],
        "RUN_DIR": row["run_dir"],
        "MANIFEST_ROW_PATH": row["manifest_row_path"],
        "SAVE_EVERY": row["save_every"],
        "VIS_EVERY": row["vis_every"],
        "BATCH_SIZE": row["batch_size"],
        "NOISE_STD": row["noise_std"],
        "LANGEVIN_SIGN": "1.0",
        "WEIGHT_DECAY": "0.0",
        "MAX_GRAD_NORM": "0.0",
        "DATA_DIR": row["data_dir"],
        "CONFIG": row["config_path"],
        "SYNC_FRESH_INIT": "1",
        "DEBUG_LEVEL": "1",
        "LOG_EVERY": "10",
    }
    exports = "\n".join(f"export {key}={shlex.quote(value)}" for key, value in env.items())
    job_name = row["exp_id"].replace("_", "")[:15]
    return "\n".join(
        [
            "#!/bin/bash",
            "#PBS -A lc-mpi",
            f"#PBS -l select={row['num_nodes']}:system=polaris",
            f"#PBS -l walltime={row['train_walltime']}",
            "#PBS -l filesystems=home:eagle",
            f"#PBS -q {row['queue']}",
            f"#PBS -N {job_name}",
            "#PBS -j oe",
            f"#PBS -o {row['pbs_output_log']}",
            "",
            "set -euo pipefail",
            exports,
            f"cp {shlex.quote(row['config_path'])} {shlex.quote(str(Path(row['run_dir']) / 'experiment_config.yaml'))}",
            f"exec /usr/bin/bash {shlex.quote(str(project_dir / 'scripts' / 'current' / 'pbs_run_ablation_case.pbs'))}",
            "",
        ]
    )


def materialize_bundle(
    rows: list[dict[str, str]],
    *,
    bundle_root: Path = BUNDLE_ROOT,
    project_dir: Path = PROJECT_DIR,
) -> list[dict[str, str]]:
    bundle_root = Path(bundle_root)
    for subdir in ("configs", "pbs", "logs", "summaries", "plots"):
        (bundle_root / subdir).mkdir(parents=True, exist_ok=True)

    materialized: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        item["config_path"] = str(bundle_root / "configs" / f"{item['exp_id']}.yaml")
        item["pbs_path"] = str(bundle_root / "pbs" / f"{item['exp_id']}.pbs")
        item["manifest_row_path"] = str(bundle_root / "configs" / f"{item['exp_id']}.manifest_row.json")
        item["run_dir"] = str(bundle_root / "runs" / item["phase"] / item["exp_id"])
        item["pbs_output_log"] = str(bundle_root / "logs" / f"{item['exp_id']}.pbs.out")
        Path(item["run_dir"]).mkdir(parents=True, exist_ok=True)
        Path(item["config_path"]).write_text(config_text(item))
        Path(item["manifest_row_path"]).write_text(json.dumps(item, indent=2, sort_keys=True))
        Path(item["pbs_path"]).write_text(pbs_text(item, project_dir=project_dir))
        materialized.append(item)

    summaries = bundle_root / "summaries"
    write_csv(summaries / "long_k_master_registry.csv", materialized, FIELDS)
    write_csv(
        summaries / "long_k_wallclock_summary.csv",
        [],
        ["phase", "exp_id", "train_mode", "K", "steps", "seed", "wallclock_sec", "gpu_hours", "step_time_sec", "throughput"],
    )
    write_csv(
        summaries / "long_k_fid_vs_step.csv",
        [],
        ["phase", "exp_id", "train_mode", "K", "seed", "step", "fid_inception", "fid_feature", "unique_ratio", "metrics_path"],
    )
    write_csv(
        summaries / "long_k_best_final_summary.csv",
        [],
        ["phase", "exp_id", "train_mode", "K", "seed", "best_step", "best_fid", "final_step", "final_fid", "status"],
    )
    (summaries / "long_k_phase_L0_report.md").write_text(render_l0_report(materialized, submitted=[]))
    return materialized


def render_l0_report(rows: list[dict[str, str]], submitted: list[dict[str, str]]) -> str:
    submitted_by_exp = {row["exp_id"]: row for row in submitted}
    lines = [
        "# Long-K Scaling Phase L0 Report",
        "",
        "Scope: original CIFAR EBM backbone/dataset only. No DRL backbone. No ImageNet-32.",
        "",
        "| exp_id | mode | K | P | step_size | submit | job_id | status |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in rows:
        if row["phase"] != "L0":
            continue
        sub = submitted_by_exp.get(row["exp_id"], {})
        lines.append(
            "| {exp_id} | {mode} | {K} | {P} | {step_size} | {submit} | {job} | {status} |".format(
                exp_id=row["exp_id"],
                mode=row["train_mode"],
                K=row["K"],
                P=row["pipe_stages"],
                step_size=row["step_size"],
                submit=row["submit"],
                job=sub.get("train_job_id", ""),
                status=sub.get("status", "prepared"),
            )
        )
    lines.extend(
        [
            "",
            "Success criteria: finite loss, no crash, at least one FID point after eval/backfill, expected max_abs_chain, wall-clock and GPU-hour logging.",
            "",
        ]
    )
    return "\n".join(lines)


def submit_l0(rows: list[dict[str, str]], bundle_root: Path, dry_run: bool = False) -> list[dict[str, str]]:
    submitted = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for row in rows:
        if not (row["phase"] == "L0" and row["submit"] == "yes"):
            continue
        job_id = ""
        if not dry_run:
            job_id = subprocess.check_output(["qsub", row["pbs_path"]], text=True).strip().split()[0]
        submitted.append(
            {
                "phase": row["phase"],
                "exp_id": row["exp_id"],
                "train_job_id": job_id if not dry_run else f"DRYRUN-{row['exp_id']}",
                "config_path": row["config_path"],
                "pbs_path": row["pbs_path"],
                "train_run_dir": row["run_dir"],
                "pbs_output_log": row["pbs_output_log"],
                "submitted_at": now,
                "status": "submitted" if not dry_run else "dry_run",
            }
        )
    write_csv(
        bundle_root / "summaries" / "long_k_l0_submitted.csv",
        submitted,
        ["phase", "exp_id", "train_job_id", "config_path", "pbs_path", "train_run_dir", "pbs_output_log", "submitted_at", "status"],
    )
    (bundle_root / "summaries" / "long_k_phase_L0_report.md").write_text(render_l0_report(rows, submitted=submitted))
    return submitted


def main() -> int:
    parser = argparse.ArgumentParser("Prepare and submit Long-K scaling experiments")
    parser.add_argument("--bundle-root", default=str(BUNDLE_ROOT))
    parser.add_argument("--project-dir", default=str(PROJECT_DIR))
    parser.add_argument("--submit-l0", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bundle_root = Path(args.bundle_root)
    project_dir = Path(args.project_dir)
    rows = materialize_bundle(build_long_k_rows(), bundle_root=bundle_root, project_dir=project_dir)
    submitted = submit_l0(rows, bundle_root=bundle_root, dry_run=args.dry_run) if args.submit_l0 else []
    print(f"bundle_root={bundle_root}")
    print(f"master_registry={bundle_root / 'summaries' / 'long_k_master_registry.csv'}")
    print(f"submitted={bundle_root / 'summaries' / 'long_k_l0_submitted.csv' if submitted else ''}")
    for row in submitted:
        print(f"{row['exp_id']}: {row['train_job_id']} {row['pbs_path']} {row['pbs_output_log']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
