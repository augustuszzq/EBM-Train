#!/usr/bin/env python3
"""Prepare and submit K400 overhead microbenchmarks.

The bundle isolates whether K400 end-to-end behavior is dominated by
DDP training/all-reduce or by the K-step Langevin sampler.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


PROJECT_DIR = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")
BUNDLE_ROOT = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_long_k_overhead_microbench")

FIELDS = [
    "phase",
    "group",
    "exp_id",
    "story",
    "benchmark",
    "microbench_mode",
    "world_size",
    "num_nodes",
    "ppn",
    "K",
    "steps",
    "warmup_steps",
    "batch_size",
    "local_batch",
    "lr",
    "step_size",
    "noise_std",
    "langevin_sign",
    "seed",
    "queue",
    "walltime",
    "submit",
    "paper_role",
    "uses_drl_backbone",
    "uses_imagenet32",
    "notes",
    "config_path",
    "pbs_path",
    "manifest_row_path",
    "run_dir",
    "pbs_output_log",
]


def make_row(microbench_mode: str, batch_size: int) -> dict:
    if batch_size % 8 != 0:
        raise ValueError("batch_size must divide world_size=8")
    label = "ddp_train_only" if microbench_mode == "ddp_train_only" else "sampling_only"
    return {
        "phase": "M0",
        "group": "long_k_overhead_microbench",
        "exp_id": "M0_%s_K400_b%s_s2k_seed1" % (label, batch_size),
        "story": "K400 overhead split: isolate DDP train-only versus Langevin sampling-only cost",
        "benchmark": "cifar10",
        "microbench_mode": microbench_mode,
        "world_size": "8",
        "num_nodes": "2",
        "ppn": "4",
        "K": "400",
        "steps": "2000",
        "warmup_steps": "50",
        "batch_size": str(batch_size),
        "local_batch": str(batch_size // 8),
        "lr": "1e-4",
        "step_size": "0.0025",
        "noise_std": "0.01",
        "langevin_sign": "1.0",
        "seed": "1",
        "queue": "preemptable",
        "walltime": "06:00:00" if microbench_mode == "sampling_only" else "02:00:00",
        "submit": "yes",
        "paper_role": "overhead_microbench",
        "uses_drl_backbone": "false",
        "uses_imagenet32": "false",
        "notes": (
            "No paper training; synthetic tensors only. Measures %s at K400 with global batch %s."
            % (microbench_mode, batch_size)
        ),
        "config_path": "",
        "pbs_path": "",
        "manifest_row_path": "",
        "run_dir": "",
        "pbs_output_log": "",
    }


def build_rows() -> list[dict]:
    rows: list[dict] = []
    for batch_size in (64, 256):
        rows.append(make_row("ddp_train_only", batch_size))
        rows.append(make_row("sampling_only", batch_size))
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def config_text(row: dict) -> str:
    return "\n".join(
        [
            "benchmark:",
            "  name: cifar10",
            "  data_mode: synthetic",
            "  uses_drl_backbone: false",
            "  uses_imagenet32: false",
            "microbench:",
            "  mode: %s" % row["microbench_mode"],
            "  K: %s" % row["K"],
            "  steps: %s" % row["steps"],
            "  warmup_steps: %s" % row["warmup_steps"],
            "  batch_size: %s" % row["batch_size"],
            "  local_batch: %s" % row["local_batch"],
            "  step_size: %s" % row["step_size"],
            "  noise_std: %s" % row["noise_std"],
            "  langevin_sign: %s" % row["langevin_sign"],
            "  purpose: split K400 end-to-end time into DDP train-only and sampling-only components",
            "",
        ]
    )


def pbs_text(row: dict, project_dir: Path) -> str:
    env = {
        "PROJECT_DIR": str(project_dir),
        "GROUP": row["group"],
        "EXP_ID": row["exp_id"],
        "MICROBENCH_MODE": row["microbench_mode"],
        "WORLD_SIZE": row["world_size"],
        "NUM_NODES": row["num_nodes"],
        "PPN": row["ppn"],
        "K": row["K"],
        "STEPS": row["steps"],
        "WARMUP_STEPS": row["warmup_steps"],
        "BATCH_SIZE": row["batch_size"],
        "LR": row["lr"],
        "STEP_SIZE": row["step_size"],
        "NOISE_STD": row["noise_std"],
        "LANGEVIN_SIGN": row["langevin_sign"],
        "SEED": row["seed"],
        "RUN_DIR": row["run_dir"],
        "MANIFEST_ROW_PATH": row["manifest_row_path"],
        "CONFIG": row["config_path"],
    }
    exports = "\n".join("export %s=%s" % (key, shlex.quote(value)) for key, value in env.items())
    job_name = row["exp_id"].replace("_", "")[:15]
    return "\n".join(
        [
            "#!/bin/bash",
            "#PBS -A lc-mpi",
            "#PBS -l select=%s:system=polaris" % row["num_nodes"],
            "#PBS -l walltime=%s" % row["walltime"],
            "#PBS -l filesystems=home:eagle",
            "#PBS -q %s" % row["queue"],
            "#PBS -N %s" % job_name,
            "#PBS -j oe",
            "#PBS -o %s" % row["pbs_output_log"],
            "",
            "set -euo pipefail",
            exports,
            "mkdir -p ${RUN_DIR}",
            "cp ${MANIFEST_ROW_PATH} ${RUN_DIR}/manifest_row.json",
            "cp ${CONFIG} ${RUN_DIR}/experiment_config.yaml",
            "cd ${PROJECT_DIR}",
            "module use /soft/modulefiles",
            "module load conda",
            "conda activate /home/kevienzzq/.conda/envs/llm-env",
            "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
            "export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}",
            "PY=/home/kevienzzq/.conda/envs/llm-env/bin/python",
            "TORCHRUN=/home/kevienzzq/.conda/envs/llm-env/bin/torchrun",
            "TRAIN_LOG=${RUN_DIR}/train.log",
            "echo \"[RUN] ${EXP_ID} mode=${MICROBENCH_MODE} batch=${BATCH_SIZE}\" | tee -a ${TRAIN_LOG}",
            "python3 - <<'PY'",
            "import json, os, time",
            "payload = {k.lower(): os.environ[k] for k in ['GROUP','EXP_ID','MICROBENCH_MODE','WORLD_SIZE','NUM_NODES','PPN','K','STEPS','BATCH_SIZE','RUN_DIR','CONFIG']}",
            "payload['pbs_jobid'] = os.environ.get('PBS_JOBID', '')",
            "payload['started_at_iso'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())",
            "with open(os.path.join(os.environ['RUN_DIR'], 'job_info.json'), 'w') as f: json.dump(payload, f, indent=2, sort_keys=True)",
            "PY",
            "if [[ -n \"${PBS_NODEFILE:-}\" && -f \"${PBS_NODEFILE}\" ]]; then",
            "  mapfile -t NODES < <(sort -u \"${PBS_NODEFILE}\")",
            "else",
            "  NODES=(\"$(hostname)\")",
            "fi",
            "NNODES=${#NODES[@]}",
            "MASTER_ADDR=${NODES[0]}",
            "JID_NUM=${PBS_JOBID%%.*}",
            "MASTER_PORT=$((29500 + (JID_NUM % 1000)))",
            "COMMON_ARGS=\"--mode ${MICROBENCH_MODE} --run_dir ${RUN_DIR} --steps ${STEPS} --warmup_steps ${WARMUP_STEPS} --batch_size ${BATCH_SIZE} --K ${K} --lr ${LR} --step_size ${STEP_SIZE} --noise_std ${NOISE_STD} --langevin_sign ${LANGEVIN_SIGN} --seed ${SEED} --no_clamp_x --log_every 10\"",
            "if [[ \"${NNODES}\" -eq 1 ]]; then",
            "  CMD=\"${TORCHRUN} --standalone --nproc_per_node=${WORLD_SIZE} ${PROJECT_DIR}/scripts/current/ebm_overhead_microbench.py ${COMMON_ARGS}\"",
            "  echo \"[CMD] ${CMD}\" | tee -a ${TRAIN_LOG}",
            "  eval \"${CMD}\" 2>&1 | tee -a ${TRAIN_LOG}",
            "else",
            "  CMD=\"${TORCHRUN} --nnodes=${NNODES} --node_rank=\\${NODE_RANK} --nproc_per_node=${PPN} --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} ${PROJECT_DIR}/scripts/current/ebm_overhead_microbench.py ${COMMON_ARGS}\"",
            "  echo \"[CMD] ${CMD}\" | tee -a ${TRAIN_LOG}",
            "  mpiexec -n ${NNODES} -ppn 1 --hostfile ${PBS_NODEFILE} bash -lc 'set -euo pipefail; module use /soft/modulefiles; module load conda; conda activate /home/kevienzzq/.conda/envs/llm-env; cd \"'\"${PROJECT_DIR}\"'\"; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}; export NODE_RANK=${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-${PALS_RANKID:-0}}}}; eval \"'\"${CMD}\"'\"' 2>&1 | tee -a ${TRAIN_LOG}",
            "fi",
            "echo \"[DONE] ${EXP_ID}\" | tee -a ${TRAIN_LOG}",
            "",
        ]
    )


def materialize(rows: list[dict], bundle_root: Path, project_dir: Path) -> list[dict]:
    for subdir in ("configs", "pbs", "logs", "summaries", "runs"):
        (bundle_root / subdir).mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
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
    write_csv(bundle_root / "summaries" / "overhead_microbench_master_registry.csv", out, FIELDS)
    return out


def submit_rows(rows: list[dict], bundle_root: Path, dry_run: bool) -> list[dict]:
    submitted: list[dict] = []
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
                "microbench_mode": row["microbench_mode"],
                "batch_size": row["batch_size"],
                "train_job_id": job_id,
                "config_path": row["config_path"],
                "pbs_path": row["pbs_path"],
                "run_dir": row["run_dir"],
                "pbs_output_log": row["pbs_output_log"],
                "submitted_at": now,
                "status": "dry_run" if dry_run else "submitted",
            }
        )
    write_csv(
        bundle_root / "summaries" / "overhead_microbench_submitted.csv",
        submitted,
        [
            "phase",
            "exp_id",
            "microbench_mode",
            "batch_size",
            "train_job_id",
            "config_path",
            "pbs_path",
            "run_dir",
            "pbs_output_log",
            "submitted_at",
            "status",
        ],
    )
    return submitted


def main() -> int:
    ap = argparse.ArgumentParser("Prepare K400 overhead microbenchmarks")
    ap.add_argument("--bundle-root", default=str(BUNDLE_ROOT))
    ap.add_argument("--project-dir", default=str(PROJECT_DIR))
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    bundle_root = Path(args.bundle_root)
    rows = materialize(build_rows(), bundle_root, Path(args.project_dir))
    submitted = submit_rows(rows, bundle_root, bool(args.dry_run)) if args.submit else []
    print("bundle_root=%s" % bundle_root)
    print("master_registry=%s" % (bundle_root / "summaries" / "overhead_microbench_master_registry.csv"))
    if submitted:
        print("submitted=%s" % (bundle_root / "summaries" / "overhead_microbench_submitted.csv"))
        for row in submitted:
            print("%s: %s %s" % (row["exp_id"], row["train_job_id"], row["pbs_output_log"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
