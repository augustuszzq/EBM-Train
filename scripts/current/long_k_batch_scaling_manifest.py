#!/usr/bin/env python3
"""Prepare and submit original-CIFAR long-K batch-scaling experiments.

This bundle keeps the previous K=400 / P=8 long-chain setting, but increases
the global batch so the 8-GPU runs have a less pathological local batch.
"""

import argparse
import csv
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")
BUNDLE_ROOT = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_long_k_batch_scaling")
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


def make_row(
    phase,
    exp_id,
    train_mode,
    pipe_stages,
    world_size,
    num_nodes,
    ppn,
    batch_size,
    steps,
    seed,
    submit,
    paper_role,
    walltime,
    notes="",
):
    local_batch = int(batch_size) // int(world_size)
    if int(batch_size) % int(world_size) != 0:
        raise ValueError("batch_size must be divisible by world_size")
    weight_mode = "uniform" if int(pipe_stages) > 1 else "deep_only"
    return {
        "phase": phase,
        "group": "long_k_batch_scaling",
        "exp_id": exp_id,
        "story": (
            "original CIFAR EBM K400 batch scaling; tests whether larger "
            "global/local batch exposes strict pipeline wall-clock advantage"
        ),
        "train_mode": train_mode,
        "world_size": str(world_size),
        "pipe_stages": str(pipe_stages),
        "stage_slices": stage_slices_text(400, pipe_stages),
        "K": "400",
        "steps": str(steps),
        "lr": "1e-4",
        "step_size": "0.0025",
        "noise_std": "0.01",
        "weight_mode": weight_mode,
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
        "train_walltime": walltime,
        "data_dir": str(DATA_DIR),
        "submit": "yes" if submit else "no",
        "paper_role": paper_role,
        "notes": notes,
        "config_path": "",
        "pbs_path": "",
        "manifest_row_path": "",
        "run_dir": "",
        "pbs_output_log": "",
    }


def build_rows():
    rows = []
    # Primary smoke: global batch 256 gives local batch 32 on 8 GPUs.
    rows.extend(
        [
            make_row(
                phase="B0",
                exp_id="B0_single_fullk_K400_b256_s20k_seed1",
                train_mode="single_fullk",
                pipe_stages=1,
                world_size=1,
                num_nodes=1,
                ppn=1,
                batch_size=256,
                steps=20000,
                seed=1,
                submit=True,
                paper_role="smoke_single_control",
                walltime="16:00:00",
                notes="single-GPU full-K control at larger batch",
            ),
            make_row(
                phase="B0",
                exp_id="B0_single_emul_P8_equal_K400_b256_s20k_seed1",
                train_mode="single_pipe_emul",
                pipe_stages=8,
                world_size=1,
                num_nodes=1,
                ppn=1,
                batch_size=256,
                steps=20000,
                seed=1,
                submit=True,
                paper_role="smoke_single_control",
                walltime="16:00:00",
                notes="single-GPU serialized P8 equal objective at larger batch",
            ),
            make_row(
                phase="B0",
                exp_id="B0_ddp_fullk_K400_b256_s20k_seed1",
                train_mode="ddp_fullk",
                pipe_stages=1,
                world_size=8,
                num_nodes=2,
                ppn=4,
                batch_size=256,
                steps=20000,
                seed=1,
                submit=True,
                paper_role="smoke_distributed_control",
                walltime="24:00:00",
                notes="8-GPU DDP full-K control; local_batch=32",
            ),
            make_row(
                phase="B0",
                exp_id="B0_pipe_P8_equal_K400_b256_s20k_seed1",
                train_mode="pipe_strict",
                pipe_stages=8,
                world_size=8,
                num_nodes=2,
                ppn=4,
                batch_size=256,
                steps=20000,
                seed=1,
                submit=True,
                paper_role="smoke_pipeline_main",
                walltime="24:00:00",
                notes="8-GPU strict pipeline P8 equal; local_batch=32",
            ),
        ]
    )

    # Prepared but not submitted: nearby batch points for the next decision.
    for batch_size in (128, 512):
        for train_mode, pipe_stages, world_size, num_nodes, ppn, label, role in (
            ("ddp_fullk", 1, 8, 2, 4, "ddp_fullk", "optional_batch_control"),
            ("pipe_strict", 8, 8, 2, 4, "pipe_P8_equal", "optional_batch_pipeline"),
        ):
            rows.append(
                make_row(
                    phase="B0_optional",
                    exp_id="B0opt_%s_K400_b%s_s20k_seed1" % (label, batch_size),
                    train_mode=train_mode,
                    pipe_stages=pipe_stages,
                    world_size=world_size,
                    num_nodes=num_nodes,
                    ppn=ppn,
                    batch_size=batch_size,
                    steps=20000,
                    seed=1,
                    submit=False,
                    paper_role=role,
                    walltime="24:00:00",
                    notes="prepared only; submit after b256 memory/runtime review",
                )
            )
    return rows


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def config_text(row):
    stage_weights = "[%s]" % ", ".join(["1.0"] * int(row["pipe_stages"]))
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
            "pipeline:",
            "  pipe_stages: %s" % row["pipe_stages"],
            "  weight_mode: %s" % row["weight_mode"],
            "  last2_beta: %s" % row["last2_beta"],
            "  stage_weights: %s" % stage_weights,
            "long_k_batch_scaling:",
            "  exp_id: %s" % row["exp_id"],
            "  stage_slices: [%s]" % slices,
            "  step_size: %s" % row["step_size"],
            "  noise_std: %s" % row["noise_std"],
            "  method: original_cifar_completion_aware_diagonal_objective",
            "  uses_drl_backbone: false",
            "  uses_imagenet32: false",
            "  motivation: increase local batch to reduce communication dominated pipeline regime",
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
            "exec /usr/bin/bash %s" % shlex.quote(str(project_dir / "scripts" / "current" / "pbs_run_ablation_case.pbs")),
            "",
        ]
    )


def render_plan(rows, submitted):
    submitted_by_exp = {}
    for row in submitted:
        submitted_by_exp[row["exp_id"]] = row
    lines = [
        "# Long-K Batch-Scaling Plan",
        "",
        "Scope: original CIFAR model/dataset only; no DRL backbone and no ImageNet-32.",
        "",
        "Motivation: the previous K400 pipeline used global batch 64 on 8 GPUs, giving local batch 8. This likely made the run communication/launch-overhead dominated. This bundle first tests global batch 256, giving local batch 32 on 8 GPUs.",
        "",
        "| exp_id | mode | world | global batch | local batch | K | P | steps | submit | job_id |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        sub = submitted_by_exp.get(row["exp_id"], {})
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                row["exp_id"],
                row["train_mode"],
                row["world_size"],
                row["batch_size"],
                row["local_batch"],
                row["K"],
                row["pipe_stages"],
                row["steps"],
                row["submit"],
                sub.get("train_job_id", ""),
            )
        )
    lines.extend(
        [
            "",
            "Gate for next phase: compare step time, throughput, finite loss, max_abs_chain, and at least one FID point. If b256 is stable and pipeline overhead improves, extend the same rows to 300k.",
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
        item["run_dir"] = str(bundle_root / "runs" / item["phase"] / item["exp_id"])
        item["pbs_output_log"] = str(bundle_root / "logs" / ("%s.pbs.out" % item["exp_id"]))
        Path(item["run_dir"]).mkdir(parents=True, exist_ok=True)
        Path(item["config_path"]).write_text(config_text(item))
        Path(item["manifest_row_path"]).write_text(json.dumps(item, indent=2, sort_keys=True))
        Path(item["pbs_path"]).write_text(pbs_text(item, project_dir))
        out.append(item)

    write_csv(bundle_root / "summaries" / "batch_scaling_master_registry.csv", out, FIELDS)
    write_csv(
        bundle_root / "summaries" / "batch_scaling_wallclock_summary.csv",
        [],
        ["phase", "exp_id", "train_mode", "batch_size", "local_batch", "K", "steps", "seed", "wallclock_sec", "gpu_hours", "step_time_sec", "throughput"],
    )
    write_csv(
        bundle_root / "summaries" / "batch_scaling_fid_vs_step.csv",
        [],
        ["phase", "exp_id", "train_mode", "batch_size", "local_batch", "K", "seed", "step", "fid_inception", "fid_feature", "unique_ratio", "metrics_path"],
    )
    (bundle_root / "summaries" / "batch_scaling_plan.md").write_text(render_plan(out, submitted=[]))
    return out


def submit_rows(rows, bundle_root, dry_run):
    submitted = []
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    for row in rows:
        if row["submit"] != "yes":
            continue
        if dry_run:
            job_id = "DRYRUN-%s" % row["exp_id"]
        else:
            job_id = subprocess.check_output(["qsub", row["pbs_path"]], text=True).strip().split()[0]
        submitted.append(
            {
                "phase": row["phase"],
                "exp_id": row["exp_id"],
                "train_job_id": job_id,
                "config_path": row["config_path"],
                "pbs_path": row["pbs_path"],
                "train_run_dir": row["run_dir"],
                "pbs_output_log": row["pbs_output_log"],
                "submitted_at": now,
                "status": "dry_run" if dry_run else "submitted",
            }
        )
    write_csv(
        bundle_root / "summaries" / "batch_scaling_submitted.csv",
        submitted,
        ["phase", "exp_id", "train_job_id", "config_path", "pbs_path", "train_run_dir", "pbs_output_log", "submitted_at", "status"],
    )
    (bundle_root / "summaries" / "batch_scaling_plan.md").write_text(render_plan(rows, submitted=submitted))
    return submitted


def main():
    parser = argparse.ArgumentParser("Prepare and submit K400 batch-scaling experiments")
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
    print("master_registry=%s" % (bundle_root / "summaries" / "batch_scaling_master_registry.csv"))
    if submitted:
        print("submitted=%s" % (bundle_root / "summaries" / "batch_scaling_submitted.csv"))
        for row in submitted:
            print("%s: %s %s" % (row["exp_id"], row["train_job_id"], row["pbs_output_log"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
