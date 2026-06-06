#!/usr/bin/env python3
"""Prepare and submit short K400 overhead attribution microbenchmarks.

This bundle does not change the paper method and does not train a paper model.
It uses synthetic CIFAR-shaped tensors to isolate why DDP full-K is slower than
single full-K even though both execute K Langevin steps.
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
BUNDLE_ROOT = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_long_k_overhead_attribution")

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
    "pipe_stages",
    "slice_steps",
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


def slice_steps(k: int, pipe_stages: int, rank: int = 0) -> int:
    base = int(k) // int(pipe_stages)
    rem = int(k) % int(pipe_stages)
    return base + (1 if int(rank) < rem else 0)


def make_row(
    *,
    exp_id: str,
    microbench_mode: str,
    world_size: int,
    num_nodes: int,
    ppn: int,
    batch_size: int,
    pipe_stages: int = 1,
    paper_role: str,
    notes: str,
    walltime: str,
) -> dict:
    if int(batch_size) % int(world_size) != 0:
        raise ValueError("batch_size must be divisible by world_size")
    return {
        "phase": "A1",
        "group": "long_k_overhead_attribution",
        "exp_id": exp_id,
        "story": "K400 overhead attribution: separate sampling critical path, local-batch utilization, DDP sync, and pipeline ring costs",
        "benchmark": "cifar10",
        "microbench_mode": microbench_mode,
        "world_size": str(world_size),
        "num_nodes": str(num_nodes),
        "ppn": str(ppn),
        "pipe_stages": str(pipe_stages),
        "slice_steps": str(slice_steps(400, pipe_stages)),
        "K": "400",
        "steps": "3000",
        "warmup_steps": "100",
        "batch_size": str(batch_size),
        "local_batch": str(int(batch_size) // int(world_size)),
        "lr": "1e-4",
        "step_size": "0.0025",
        "noise_std": "0.01",
        "langevin_sign": "1.0",
        "seed": "1",
        "queue": "preemptable",
        "walltime": walltime,
        "submit": "yes",
        "paper_role": paper_role,
        "uses_drl_backbone": "false",
        "uses_imagenet32": "false",
        "notes": notes,
        "config_path": "",
        "pbs_path": "",
        "manifest_row_path": "",
        "run_dir": "",
        "pbs_output_log": "",
    }


def build_rows() -> list[dict]:
    return [
        make_row(
            exp_id="A1_single_global_fullstep_K400_b256_s3k_seed1",
            microbench_mode="single_full_step",
            world_size=1,
            num_nodes=1,
            ppn=1,
            batch_size=256,
            paper_role="single_global_fullk_reference",
            walltime="04:00:00",
            notes="One GPU, full synthetic update, global batch 256; matches single full-K batch.",
        ),
        make_row(
            exp_id="A1_single_local_fullstep_K400_b32_s3k_seed1",
            microbench_mode="single_full_step",
            world_size=1,
            num_nodes=1,
            ppn=1,
            batch_size=32,
            paper_role="single_local_batch_reference",
            walltime="04:00:00",
            notes="One GPU, local batch 32; matches per-rank local batch of 8-GPU batch256 DDP.",
        ),
        make_row(
            exp_id="A1_independent_fullstep_K400_b256_s3k_seed1",
            microbench_mode="independent_full_step",
            world_size=8,
            num_nodes=2,
            ppn=4,
            batch_size=256,
            paper_role="no_ddp_multigpu_reference",
            walltime="04:00:00",
            notes="Eight independent processes, local batch 32, full K sampling and local optimizer, no DDP all-reduce.",
        ),
        make_row(
            exp_id="A1_independent_sampling_K400_b256_s3k_seed1",
            microbench_mode="independent_sampling_only",
            world_size=8,
            num_nodes=2,
            ppn=4,
            batch_size=256,
            paper_role="sampling_no_sync_reference",
            walltime="04:00:00",
            notes="Eight independent processes, local batch 32, K-step sampling only, no per-step synchronization.",
        ),
        make_row(
            exp_id="A1_sampling_barrier_K400_b256_s3k_seed1",
            microbench_mode="sampling_only_barrier",
            world_size=8,
            num_nodes=2,
            ppn=4,
            batch_size=256,
            paper_role="sampling_sync_reference",
            walltime="04:00:00",
            notes="Distributed K-step sampling only plus a per-step barrier to expose straggler/synchronization cost.",
        ),
        make_row(
            exp_id="A1_ddp_train_only_K400_b256_s3k_seed1",
            microbench_mode="ddp_train_only",
            world_size=8,
            num_nodes=2,
            ppn=4,
            batch_size=256,
            paper_role="ordinary_ddp_allreduce_reference",
            walltime="02:00:00",
            notes="DDP forward/backward/optimizer only, no Langevin sampler.",
        ),
        make_row(
            exp_id="A1_ddp_fullstep_K400_b256_s3k_seed1",
            microbench_mode="ddp_full_step",
            world_size=8,
            num_nodes=2,
            ppn=4,
            batch_size=256,
            paper_role="synthetic_ddp_fullk_reference",
            walltime="04:00:00",
            notes="Synthetic DDP full update: K-step sampling plus DDP train update, no dataloader/replay.",
        ),
        make_row(
            exp_id="A1_pipeline_stage_nocomm_P8_K400_b256_s3k_seed1",
            microbench_mode="pipeline_stage_nocomm",
            world_size=8,
            num_nodes=2,
            ppn=4,
            batch_size=256,
            pipe_stages=8,
            paper_role="pipeline_stage_sampling_reference",
            walltime="02:00:00",
            notes="Each stage runs K/P=50 sampling steps, no ring handoff and no train update.",
        ),
        make_row(
            exp_id="A1_pipeline_stage_ring_P8_K400_b256_s3k_seed1",
            microbench_mode="pipeline_stage_ring",
            world_size=8,
            num_nodes=2,
            ppn=4,
            batch_size=256,
            pipe_stages=8,
            paper_role="pipeline_ring_comm_reference",
            walltime="02:00:00",
            notes="Each stage runs K/P=50 sampling steps plus image-state ring handoff.",
        ),
        make_row(
            exp_id="A1_pipeline_fullstep_ring_P8_K400_b256_s3k_seed1",
            microbench_mode="pipeline_full_step_ring",
            world_size=8,
            num_nodes=2,
            ppn=4,
            batch_size=256,
            pipe_stages=8,
            paper_role="synthetic_pipeline_fullstep_reference",
            walltime="04:00:00",
            notes="Synthetic pipeline step: K/P sampling, ring handoff, and DDP train update.",
        ),
    ]


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
            "  pipe_stages: %s" % row["pipe_stages"],
            "  slice_steps: %s" % row["slice_steps"],
            "  steps: %s" % row["steps"],
            "  warmup_steps: %s" % row["warmup_steps"],
            "  batch_size: %s" % row["batch_size"],
            "  local_batch: %s" % row["local_batch"],
            "  step_size: %s" % row["step_size"],
            "  noise_std: %s" % row["noise_std"],
            "  purpose: overhead attribution for K400 sampling/DDP/pipeline critical path",
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
        "PIPE_STAGES": row["pipe_stages"],
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
            "echo \"[RUN] ${EXP_ID} mode=${MICROBENCH_MODE} batch=${BATCH_SIZE} world=${WORLD_SIZE}\" | tee -a ${TRAIN_LOG}",
            "python3 - <<'PY'",
            "import json, os, time",
            "payload = {k.lower(): os.environ[k] for k in ['GROUP','EXP_ID','MICROBENCH_MODE','WORLD_SIZE','NUM_NODES','PPN','PIPE_STAGES','K','STEPS','BATCH_SIZE','RUN_DIR','CONFIG']}",
            "payload['pbs_jobid'] = os.environ.get('PBS_JOBID', '')",
            "payload['started_at_iso'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())",
            "with open(os.path.join(os.environ['RUN_DIR'], 'job_info.json'), 'w') as f: json.dump(payload, f, indent=2, sort_keys=True)",
            "PY",
            "COMMON_ARGS=\"--mode ${MICROBENCH_MODE} --run_dir ${RUN_DIR} --steps ${STEPS} --warmup_steps ${WARMUP_STEPS} --batch_size ${BATCH_SIZE} --K ${K} --pipe_stages ${PIPE_STAGES} --lr ${LR} --step_size ${STEP_SIZE} --noise_std ${NOISE_STD} --langevin_sign ${LANGEVIN_SIGN} --seed ${SEED} --no_clamp_x --log_every 10\"",
            "if [[ \"${WORLD_SIZE}\" -eq 1 ]]; then",
            "  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}",
            "  CMD=\"${PY} -u ${PROJECT_DIR}/scripts/current/ebm_overhead_microbench.py ${COMMON_ARGS}\"",
            "  echo \"[CMD] ${CMD}\" | tee -a ${TRAIN_LOG}",
            "  eval \"${CMD}\" 2>&1 | tee -a ${TRAIN_LOG}",
            "else",
            "  if [[ -n \"${PBS_NODEFILE:-}\" && -f \"${PBS_NODEFILE}\" ]]; then",
            "    mapfile -t NODES < <(sort -u \"${PBS_NODEFILE}\")",
            "  else",
            "    NODES=(\"$(hostname)\")",
            "  fi",
            "  NNODES=${#NODES[@]}",
            "  MASTER_ADDR=${NODES[0]}",
            "  JID_NUM=${PBS_JOBID%%.*}",
            "  MASTER_PORT=$((29500 + (JID_NUM % 1000)))",
            "  CMD=\"${TORCHRUN} --nnodes=${NNODES} --node_rank=\\${NODE_RANK} --nproc_per_node=${PPN} --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} ${PROJECT_DIR}/scripts/current/ebm_overhead_microbench.py ${COMMON_ARGS}\"",
            "  echo \"[CMD] ${CMD}\" | tee -a ${TRAIN_LOG}",
            "  mpiexec -n ${NNODES} -ppn 1 --hostfile ${PBS_NODEFILE} bash -lc 'set -euo pipefail; module use /soft/modulefiles; module load conda; conda activate /home/kevienzzq/.conda/envs/llm-env; cd \"'\"${PROJECT_DIR}\"'\"; export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}; export MASTER_ADDR=\"'\"${MASTER_ADDR}\"'\"; export MASTER_PORT=\"'\"${MASTER_PORT}\"'\"; export NODE_RANK=${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-${PALS_RANKID:-0}}}}; eval \"'\"${CMD}\"'\"' 2>&1 | tee -a ${TRAIN_LOG}",
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
    write_csv(bundle_root / "summaries" / "overhead_attribution_master_registry.csv", out, FIELDS)
    return out


def submit_rows(rows: list[dict], bundle_root: Path, dry_run: bool) -> list[dict]:
    submitted: list[dict] = []
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    for row in rows:
        if row["submit"] != "yes":
            continue
        qsub_error = ""
        if dry_run:
            job_id = "DRYRUN-%s" % row["exp_id"]
            status = "dry_run"
        else:
            try:
                job_id = subprocess.check_output(["qsub", row["pbs_path"]], stderr=subprocess.STDOUT, text=True).strip().split()[0]
                status = "submitted"
            except subprocess.CalledProcessError as exc:
                job_id = ""
                status = "qsub_failed"
                qsub_error = (exc.output or str(exc)).strip()
        submitted.append(
            {
                "phase": row["phase"],
                "exp_id": row["exp_id"],
                "microbench_mode": row["microbench_mode"],
                "batch_size": row["batch_size"],
                "local_batch": row["local_batch"],
                "train_job_id": job_id,
                "config_path": row["config_path"],
                "pbs_path": row["pbs_path"],
                "run_dir": row["run_dir"],
                "pbs_output_log": row["pbs_output_log"],
                "submitted_at": now,
                "status": status,
                "qsub_error": qsub_error,
            }
        )
    write_csv(
        bundle_root / "summaries" / "overhead_attribution_submitted.csv",
        submitted,
        [
            "phase",
            "exp_id",
            "microbench_mode",
            "batch_size",
            "local_batch",
            "train_job_id",
            "config_path",
            "pbs_path",
            "run_dir",
            "pbs_output_log",
            "submitted_at",
            "status",
            "qsub_error",
        ],
    )
    return submitted


def load_previous_submissions(bundle_root: Path) -> dict[str, dict]:
    path = bundle_root / "summaries" / "overhead_attribution_submitted.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row["exp_id"]: row for row in csv.DictReader(f)}


def submit_rows_idempotent(rows: list[dict], bundle_root: Path, dry_run: bool) -> list[dict]:
    previous = load_previous_submissions(bundle_root)
    merged: list[dict] = []
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    for row in rows:
        prev = previous.get(row["exp_id"], {})
        prev_job = prev.get("train_job_id", "")
        prev_status = prev.get("status", "")
        if prev_job and prev_status in {"submitted", "already_submitted"}:
            merged.append(
                {
                    "phase": row["phase"],
                    "exp_id": row["exp_id"],
                    "microbench_mode": row["microbench_mode"],
                    "batch_size": row["batch_size"],
                    "local_batch": row["local_batch"],
                    "train_job_id": prev_job,
                    "config_path": row["config_path"],
                    "pbs_path": row["pbs_path"],
                    "run_dir": row["run_dir"],
                    "pbs_output_log": row["pbs_output_log"],
                    "submitted_at": prev.get("submitted_at", ""),
                    "status": "already_submitted",
                    "qsub_error": "",
                }
            )
            continue
        qsub_error = ""
        if dry_run:
            job_id = "DRYRUN-%s" % row["exp_id"]
            status = "dry_run"
        else:
            try:
                job_id = subprocess.check_output(["qsub", row["pbs_path"]], stderr=subprocess.STDOUT, text=True).strip().split()[0]
                status = "submitted"
            except subprocess.CalledProcessError as exc:
                job_id = ""
                status = "qsub_failed"
                qsub_error = (exc.output or str(exc)).strip()
        merged.append(
            {
                "phase": row["phase"],
                "exp_id": row["exp_id"],
                "microbench_mode": row["microbench_mode"],
                "batch_size": row["batch_size"],
                "local_batch": row["local_batch"],
                "train_job_id": job_id,
                "config_path": row["config_path"],
                "pbs_path": row["pbs_path"],
                "run_dir": row["run_dir"],
                "pbs_output_log": row["pbs_output_log"],
                "submitted_at": now,
                "status": status,
                "qsub_error": qsub_error,
            }
        )
    write_csv(
        bundle_root / "summaries" / "overhead_attribution_submitted.csv",
        merged,
        [
            "phase",
            "exp_id",
            "microbench_mode",
            "batch_size",
            "local_batch",
            "train_job_id",
            "config_path",
            "pbs_path",
            "run_dir",
            "pbs_output_log",
            "submitted_at",
            "status",
            "qsub_error",
        ],
    )
    return merged


def main() -> int:
    ap = argparse.ArgumentParser("Prepare K400 overhead attribution microbenchmarks")
    ap.add_argument("--bundle-root", default=str(BUNDLE_ROOT))
    ap.add_argument("--project-dir", default=str(PROJECT_DIR))
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--idempotent", action="store_true", help="do not resubmit rows that already have a job id")
    args = ap.parse_args()
    bundle_root = Path(args.bundle_root)
    rows = materialize(build_rows(), bundle_root, Path(args.project_dir))
    if args.submit and args.idempotent:
        submitted = submit_rows_idempotent(rows, bundle_root, bool(args.dry_run))
    elif args.submit:
        submitted = submit_rows(rows, bundle_root, bool(args.dry_run))
    else:
        submitted = []
    print("bundle_root=%s" % bundle_root)
    print("master_registry=%s" % (bundle_root / "summaries" / "overhead_attribution_master_registry.csv"))
    if submitted:
        print("submitted=%s" % (bundle_root / "summaries" / "overhead_attribution_submitted.csv"))
        for row in submitted:
            print("%s: %s %s" % (row["exp_id"], row["train_job_id"], row["pbs_output_log"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
