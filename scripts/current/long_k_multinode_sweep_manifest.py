#!/usr/bin/env python3
"""Prepare and submit original-CIFAR multi-node long-K scaling experiments.

This bundle extends the stable CIFAR EBM setting, not the DRL/ImageNet path.
It screens whether larger global batch and more pipeline stages expose a
wall-clock advantage when the Langevin budget grows to K=400/800/1600.
"""

import argparse
import csv
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")
BUNDLE_ROOT = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_long_k_multinode_sweep")
DATA_DIR = Path("/eagle/lc-mpi/Zhiqing/ebm/data/cifar10")

FIELDS = [
    "phase",
    "scale_group",
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
    "local_batch",
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
    "manifest_row_path",
    "run_dir",
    "pbs_output_log",
]


def k_slices(k, p):
    base = int(k) // int(p)
    rem = int(k) % int(p)
    return [base + (1 if i < rem else 0) for i in range(int(p))]


def stage_slices_text(k, p):
    return ";".join(str(x) for x in k_slices(k, p))


def step_size_for_k(k):
    return "%.8g" % (0.01 * 100.0 / float(k))


def walltime_for(k, train_mode):
    # Polaris preemptable allows 72h; long-K runs otherwise fragment too often.
    return "72:00:00"


def make_row(scale, train_mode, k, steps, seed):
    if scale == "S16":
        world_size = 16 if train_mode != "single_fullk" else 1
        pipe_stages = 16 if train_mode == "pipe_strict" else 1
        num_nodes = 4 if train_mode != "single_fullk" else 1
        batch_size = 512
    elif scale == "S32":
        world_size = 32 if train_mode != "single_fullk" else 1
        pipe_stages = 32 if train_mode == "pipe_strict" else 1
        num_nodes = 8 if train_mode != "single_fullk" else 1
        batch_size = 1024
    else:
        raise ValueError("unknown scale %s" % scale)

    ppn = 4 if train_mode != "single_fullk" else 1
    if batch_size % world_size != 0:
        raise ValueError("batch_size must be divisible by world_size")
    local_batch = batch_size // world_size
    if train_mode == "ddp_fullk":
        mode_label = "ddp_fullk"
        family = "distributed_full_k"
        paper_role = "distributed_control"
    elif train_mode == "pipe_strict":
        mode_label = "pipe_P%s_equal" % pipe_stages
        family = "strict_pipeline"
        paper_role = "pipeline_main"
    elif train_mode == "single_fullk":
        mode_label = "single_fullk"
        family = "single_full_k"
        paper_role = "single_control"
    else:
        raise ValueError("unsupported train_mode %s" % train_mode)

    exp_id = "%s_%s_K%s_b%s_s%dk_seed%s" % (
        scale,
        mode_label,
        int(k),
        int(batch_size),
        int(steps) // 1000,
        int(seed),
    )
    notes = (
        "original CIFAR model/backbone; completion-aware objective unchanged; "
        "K-sweep screen for multi-node scaling"
    )
    if train_mode == "single_fullk" and int(k) >= 800:
        notes += "; high-batch single-GPU stress test, likely needs resume"

    return {
        "phase": "M0",
        "scale_group": scale,
        "group": "long_k_multinode_sweep",
        "exp_id": exp_id,
        "story": (
            "original CIFAR long-K scaling for single/DDP/pipeline; tests "
            "K=400/800/1600 and 16/32-GPU multi-node scaling"
        ),
        "train_mode": train_mode,
        "world_size": str(world_size),
        "pipe_stages": str(pipe_stages),
        "stage_slices": stage_slices_text(k, pipe_stages),
        "K": str(k),
        "steps": str(steps),
        "lr": "1e-4",
        "step_size": step_size_for_k(k),
        "noise_std": "0.01",
        "weight_mode": "uniform" if train_mode == "pipe_strict" else "deep_only",
        "last2_beta": "0.01",
        "seed": str(seed),
        "num_nodes": str(num_nodes),
        "ppn": str(ppn),
        "batch_size": str(batch_size),
        "local_batch": str(local_batch),
        "save_every": "5000",
        "vis_every": "5000",
        "eval_images": "5000",
        "queue": "preemptable",
        "train_walltime": walltime_for(k, train_mode),
        "data_dir": str(DATA_DIR),
        "submit": "yes",
        "paper_role": paper_role,
        "notes": "%s; family=%s" % (notes, family),
        "config_path": "",
        "pbs_path": "",
        "manifest_row_path": "",
        "run_dir": "",
        "pbs_output_log": "",
    }


def build_rows():
    rows = []
    for scale in ("S16", "S32"):
        for k in (400, 800, 1600):
            for train_mode in ("single_fullk", "ddp_fullk", "pipe_strict"):
                rows.append(make_row(scale, train_mode, k, steps=20000, seed=1))
    return rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def config_text(row):
    slices = ", ".join(row["stage_slices"].split(";"))
    return "\n".join(
        [
            "benchmark:",
            "  name: cifar10",
            "  conditional: false",
            "  num_classes: 0",
            "  image_size: 32",
            "  data_root: %s" % row["data_dir"],
            "  split_train: train",
            "  split_val: train",
            "train:",
            "  steps: %s" % row["steps"],
            "  batch_size: %s" % row["batch_size"],
            "  local_batch: %s" % row["local_batch"],
            "  K: %s" % row["K"],
            "  lr: %s" % row["lr"],
            "  step_size: %s" % row["step_size"],
            "  noise_std: %s" % row["noise_std"],
            "pipeline:",
            "  pipe_stages: %s" % row["pipe_stages"],
            "  weight_mode: %s" % row["weight_mode"],
            "  last2_beta: %s" % row["last2_beta"],
            "  stage_slices: [%s]" % slices,
            "multinode_k_sweep:",
            "  exp_id: %s" % row["exp_id"],
            "  scale_group: %s" % row["scale_group"],
            "  world_size: %s" % row["world_size"],
            "  num_nodes: %s" % row["num_nodes"],
            "  ppn: %s" % row["ppn"],
            "  method: original_cifar_completion_aware_diagonal_objective",
            "  uses_drl_backbone: false",
            "  uses_imagenet32: false",
            "  objective_note: weights are applied to negative energy scalars, not images",
            "",
        ]
    )


def pbs_text(row, project_dir):
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
        "RESUME_CKPT": "auto",
    }
    exports = "\n".join("export %s=%s" % (key, shlex.quote(value)) for key, value in env.items())
    job_name = row["exp_id"].replace("_", "")[:15]
    return "\n".join(
        [
            "#!/bin/bash",
            "#PBS -A lc-mpi",
            "#PBS -l select=%s:system=polaris" % row["num_nodes"],
            "#PBS -l walltime=%s" % row["train_walltime"],
            "#PBS -l filesystems=home:eagle",
            "#PBS -q %s" % row["queue"],
            "#PBS -N %s" % job_name,
            "#PBS -j oe",
            "#PBS -o %s" % row["pbs_output_log"],
            "",
            "set -euo pipefail",
            exports,
            "cp %s %s" % (
                shlex.quote(row["config_path"]),
                shlex.quote(str(Path(row["run_dir"]) / "experiment_config.yaml")),
            ),
            "exec /usr/bin/bash %s" % shlex.quote(
                str(project_dir / "scripts" / "current" / "pbs_run_ablation_case.pbs")
            ),
            "",
        ]
    )


def render_plan(rows, submitted):
    submitted_by_exp = dict((row["exp_id"], row) for row in submitted)
    lines = [
        "# Multi-Node Long-K Sweep",
        "",
        "Scope: original CIFAR EBM only; no DRL backbone and no ImageNet-32.",
        "",
        "Objective semantics are unchanged: pipeline weights are applied to negative energy scalars, never to image tensors.",
        "",
        "This first phase is a 20k screen for K=400/800/1600 at 16-GPU and 32-GPU scale.",
        "",
        "| exp_id | mode | scale | nodes | world | batch | local | K | P | step_size | walltime | job_id |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        sub = submitted_by_exp.get(row["exp_id"], {})
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                row["exp_id"],
                row["train_mode"],
                row["scale_group"],
                row["num_nodes"],
                row["world_size"],
                row["batch_size"],
                row["local_batch"],
                row["K"],
                row["pipe_stages"],
                row["step_size"],
                row["train_walltime"],
                sub.get("train_job_id", ""),
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- S16 uses 4 nodes / 16 GPUs for DDP and pipeline; S32 uses 8 nodes / 32 GPUs.",
            "- Single runs use the corresponding global batch on one GPU and may require resume for K=800/1600.",
            "- Step size follows 0.01 * 100 / K.",
            "",
        ]
    )
    return "\n".join(lines)


def materialize(rows, bundle_root, project_dir):
    for subdir in ("configs", "pbs", "logs", "summaries", "plots", "runs"):
        (bundle_root / subdir).mkdir(parents=True, exist_ok=True)

    out = []
    for row in rows:
        item = dict(row)
        item["config_path"] = str(bundle_root / "configs" / ("%s.yaml" % item["exp_id"]))
        item["pbs_path"] = str(bundle_root / "pbs" / ("%s.pbs" % item["exp_id"]))
        item["manifest_row_path"] = str(bundle_root / "configs" / ("%s.manifest_row.json" % item["exp_id"]))
        item["run_dir"] = str(bundle_root / "runs" / item["phase"] / item["scale_group"] / item["exp_id"])
        item["pbs_output_log"] = str(bundle_root / "logs" / ("%s.pbs.out" % item["exp_id"]))
        Path(item["run_dir"]).mkdir(parents=True, exist_ok=True)
        Path(item["config_path"]).write_text(config_text(item))
        Path(item["manifest_row_path"]).write_text(json.dumps(item, indent=2, sort_keys=True))
        Path(item["pbs_path"]).write_text(pbs_text(item, project_dir))
        out.append(item)

    write_csv(bundle_root / "summaries" / "multinode_k_sweep_master_registry.csv", out, FIELDS)
    write_csv(
        bundle_root / "summaries" / "multinode_k_sweep_wallclock_summary.csv",
        [],
        [
            "phase",
            "scale_group",
            "exp_id",
            "train_mode",
            "world_size",
            "batch_size",
            "local_batch",
            "K",
            "steps",
            "seed",
            "wallclock_sec",
            "gpu_hours",
            "step_time_sec",
            "throughput",
        ],
    )
    write_csv(
        bundle_root / "summaries" / "multinode_k_sweep_fid_vs_step.csv",
        [],
        [
            "phase",
            "scale_group",
            "exp_id",
            "train_mode",
            "world_size",
            "batch_size",
            "local_batch",
            "K",
            "seed",
            "step",
            "fid_inception",
            "fid_feature",
            "unique_ratio",
            "metrics_path",
        ],
    )
    (bundle_root / "summaries" / "multinode_k_sweep_plan.md").write_text(render_plan(out, submitted=[]))
    return out


def submit_rows(rows, bundle_root, dry_run):
    submitted = []
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    for row in rows:
        if row["submit"] != "yes":
            continue
        status = "submitted"
        error = ""
        if dry_run:
            job_id = "DRYRUN-%s" % row["exp_id"]
            status = "dry_run"
        else:
            try:
                job_id = subprocess.check_output(
                    ["qsub", row["pbs_path"]],
                    universal_newlines=True,
                    stderr=subprocess.STDOUT,
                ).strip().split()[0]
            except subprocess.CalledProcessError as exc:
                job_id = ""
                status = "qsub_failed"
                error = exc.output.strip()
        submitted.append(
            {
                "phase": row["phase"],
                "scale_group": row["scale_group"],
                "exp_id": row["exp_id"],
                "train_job_id": job_id,
                "config_path": row["config_path"],
                "pbs_path": row["pbs_path"],
                "train_run_dir": row["run_dir"],
                "pbs_output_log": row["pbs_output_log"],
                "submitted_at": now,
                "status": status,
                "error": error,
            }
        )
    write_csv(
        bundle_root / "summaries" / "multinode_k_sweep_submitted.csv",
        submitted,
        [
            "phase",
            "scale_group",
            "exp_id",
            "train_job_id",
            "config_path",
            "pbs_path",
            "train_run_dir",
            "pbs_output_log",
            "submitted_at",
            "status",
            "error",
        ],
    )
    (bundle_root / "summaries" / "multinode_k_sweep_plan.md").write_text(render_plan(rows, submitted=submitted))
    return submitted


def main():
    parser = argparse.ArgumentParser("Prepare and submit multi-node long-K sweep")
    parser.add_argument("--bundle-root", default=str(BUNDLE_ROOT))
    parser.add_argument("--project-dir", default=str(PROJECT_DIR))
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    bundle_root = Path(args.bundle_root)
    project_dir = Path(args.project_dir)
    rows = materialize(build_rows(), bundle_root, project_dir)
    submitted = submit_rows(rows, bundle_root, dry_run=args.dry_run) if args.submit else []
    print("bundle_root=%s" % bundle_root)
    print("master_registry=%s" % (bundle_root / "summaries" / "multinode_k_sweep_master_registry.csv"))
    if submitted:
        print("submitted=%s" % (bundle_root / "summaries" / "multinode_k_sweep_submitted.csv"))
        for row in submitted:
            msg = "%s: %s %s" % (row["exp_id"], row["train_job_id"] or row["status"], row["pbs_output_log"])
            if row.get("error"):
                msg += " error=%s" % row["error"]
            print(msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
