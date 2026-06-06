#!/usr/bin/env python3
"""Audit experiment runs and reconcile them with the final paper taxonomy.

This script produces four outputs under ``runs_final_bundle``:

1. master_experiment_registry.csv
2. scheduler_attempts.csv
3. paper_manifest_current.md
4. manifest_diff_vs_claimed_plan.md

The registry keeps three counting units separate:
- family
- logical run = family x seed x horizon
- scheduler/local execution attempt
"""

import csv
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple


ROOT = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm")
CODE_ROOT = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")
EXPERIMENTS = ROOT / "experiments"
BUNDLE = ROOT / "runs_final_bundle"

PHASE1_SUMMARY = BUNDLE / "phase1_summary.csv"
PHASE2_SUMMARY = BUNDLE / "phase2_summary.csv"
ABLATION_500K_SUMMARY = BUNDLE / "ablation_500k_summary.csv"
PIPELINE_THEN_WEIGHTING_CURRENT = BUNDLE / "pipeline_then_weighting_current.csv"
IMAGENET32_PHASEA_CURRENT = BUNDLE / "imagenet32_phasea_current.csv"

PHASE1_SUBMITTED = EXPERIMENTS / "ablation_manifest_phase1_submitted.csv"
PHASE2_SUBMITTED = EXPERIMENTS / "ablation_manifest_phase2_submitted.csv"
ABLATION_500K_SUBMITTED = EXPERIMENTS / "ablation_manifest_500k_submitted.csv"
PTW_SUBMITTED = EXPERIMENTS / "pipeline_then_weighting_manifest_submitted.csv"
IMAGENET32_PHASEA_SUBMITTED = EXPERIMENTS / "imagenet32_phasea_submitted.csv"
OBJECTIVE_MANIFEST = EXPERIMENTS / "objective_first_followup_manifest.csv"
OBJECTIVE_SUBMITTED = EXPERIMENTS / "objective_first_followup_manifest_submitted.csv"

MASTER_OUT = BUNDLE / "master_experiment_registry.csv"
ATTEMPTS_OUT = BUNDLE / "scheduler_attempts.csv"
PAPER_MANIFEST_MD = BUNDLE / "paper_manifest_current.md"
MANIFEST_DIFF_MD = BUNDLE / "manifest_diff_vs_claimed_plan.md"


CURRENT_PRIORITY_FAMILIES = {
    "single_P1_terminal",
    "ddp_P1_terminal",
    "single_P2_equal",
    "single_P2_deepest",
    "pipe_P4_equal",
}
CURRENT_OPTIONAL_CONTROL_FAMILIES = {
    "pipe_P2_equal",
    "pipe_P2_deepest",
    "single_P4_equal",
}


RUN_DIR_PACKAGE_MAP = {
    "runs_ablation": "phase1_screening",
    "runs_ablation_phase2": "phase2_screening",
    "runs_ablation_500k": "mainline_500k",
    "runs_pipeline_then_weighting": "pipeline_then_weighting",
    "runs_objective_first_followup": "objective_first_followup",
    "runs_imagenet32_phasea": "imagenet32_phasea",
}


MASTER_FIELDS = [
    "ledger",
    "phase_group",
    "family_id",
    "logical_run_id",
    "scheduler_attempt_count",
    "benchmark",
    "regime",
    "stage_count",
    "weighting",
    "K",
    "horizon_steps",
    "seed",
    "status",
    "paper_role",
    "canonical_run_dir",
    "latest_checkpoint",
    "latest_eval_step",
    "notes",
]


ATTEMPT_FIELDS = [
    "ledger",
    "phase_group",
    "family_id",
    "logical_run_id",
    "scheduler_attempt_id",
    "exp_id",
    "benchmark",
    "regime",
    "stage_count",
    "weighting",
    "K",
    "horizon_steps",
    "seed",
    "attempt_kind",
    "attempt_status",
    "scheduler_state",
    "exit_status",
    "train_job_id",
    "submitted_at",
    "started_at",
    "finished_at",
    "run_dir",
    "latest_checkpoint",
    "latest_eval_step",
    "notes",
]


def clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [{k: clean(v) for k, v in row.items()} for row in csv.DictReader(handle)]


def write_csv(path: Path, fields: List[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: clean(row.get(field, "")) for field in fields})


def normalize_regime(train_mode: str) -> str:
    return {
        "single_fullk": "single_strict",
        "ddp_fullk": "ddp_strict",
        "single_pipe_emul": "single_emulation",
        "pipe_strict": "strict_pipeline",
    }.get(train_mode, train_mode or "unknown")


def infer_benchmark(row: Mapping[str, str]) -> str:
    benchmark = clean(row.get("benchmark"))
    if benchmark:
        return benchmark
    data_dir = clean(row.get("data_dir")).lower()
    if "imagenet32" in data_dir:
        return "imagenet32"
    if "cifar10" in data_dir:
        return "cifar10"
    return "unknown"


def normalize_weighting(train_mode: str, weight_mode: str, last2_beta: str, stage_count: str) -> str:
    stage = clean(stage_count) or clean(weight_mode)
    if train_mode in {"single_fullk", "ddp_fullk"} and clean(stage_count) in {"", "0", "1"}:
        return "terminal"
    if weight_mode == "uniform":
        return "equal"
    if weight_mode == "deep_only":
        return "deepest"
    if weight_mode == "last2_beta":
        beta = clean(last2_beta)
        return f"late_stage_beta_{beta}" if beta else "late_stage_beta"
    if weight_mode:
        return "other"
    if stage == "1":
        return "terminal"
    return "other"


def extract_seed(row: Mapping[str, str]) -> str:
    seed = clean(row.get("seed"))
    if seed:
        return seed
    exp_id = clean(row.get("exp_id"))
    match = re.search(r"seed(\d+)", exp_id)
    if match:
        return match.group(1)
    return "1"


def family_id_for(row: Mapping[str, str], ledger: str, phase_group: str) -> str:
    train_mode = clean(row.get("train_mode"))
    stage_count = clean(row.get("pipe_stages")) or ("1" if train_mode in {"single_fullk", "ddp_fullk"} else "")
    weighting = normalize_weighting(train_mode, clean(row.get("weight_mode")), clean(row.get("last2_beta")), stage_count)
    k = clean(row.get("K"))

    if train_mode == "single_fullk":
        family = "single_P1_terminal"
    elif train_mode == "ddp_fullk":
        family = "ddp_P1_terminal"
    elif train_mode == "single_pipe_emul":
        family = f"single_P{stage_count}_{weighting}"
    elif train_mode == "pipe_strict":
        family = f"pipe_P{stage_count}_{weighting}"
    else:
        family = f"other_{train_mode or 'unknown'}"

    if phase_group == "O5":
        family = f"{family}_K{k}"
    elif ledger == "historical_exploratory" and k and k != "100":
        family = f"{family}_K{k}"
    return family


def logical_run_id_for(
    ledger: str,
    experiment_package: str,
    benchmark: str,
    family_id: str,
    seed: str,
    horizon_steps: str,
) -> str:
    # For historical ledgers preserve package context to avoid collapsing
    # semantically different superseded runs that happen to share normalized names.
    prefix = f"{ledger}:{experiment_package}" if ledger == "historical_exploratory" else f"{ledger}:{benchmark}"
    return f"{prefix}:{family_id}:seed{seed}:h{horizon_steps}"


def paper_role_for(ledger: str, phase_group: str, family_id: str) -> str:
    if ledger == "historical_exploratory":
        return "deprecated"
    if ledger == "external_benchmark":
        return "external"
    if phase_group in {"O3", "O4"}:
        if family_id in CURRENT_PRIORITY_FAMILIES:
            return "headline"
        if family_id in CURRENT_OPTIONAL_CONTROL_FAMILIES or "deepest" in family_id:
            return "control"
    if phase_group == "O5":
        return "control"
    if phase_group in {"O1", "O2"}:
        stage_match = re.search(r"_P(\d+)_", family_id)
        if stage_match and int(stage_match.group(1)) >= 8:
            return "appendix"
        return "control"
    return "appendix"


def status_from_summary(value: str) -> str:
    value = clean(value)
    if value in {"reference", "done", "done_local", "F:0"}:
        return "completed"
    if value in {"R"}:
        return "running"
    if value in {"Q", "H"}:
        return "queued"
    if value.startswith("F:"):
        return "failed"
    return "missing"


def latest_checkpoint(run_dir: str) -> str:
    if not run_dir:
        return ""
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return ""
    ckpts = sorted(ckpt_dir.glob("ckpt_step*.pt"))
    return str(ckpts[-1]) if ckpts else ""


def checkpoint_step_from_path(path: str) -> int:
    match = re.search(r"ckpt_step(\d+)\.pt$", clean(path))
    if not match:
        return -1
    return int(match.group(1))


def latest_eval_step(run_dir: str) -> str:
    if not run_dir:
        return ""
    run_path = Path(run_dir)
    traj_dir = run_path / "trajectory"
    if traj_dir.exists():
        steps = []
        for child in traj_dir.iterdir():
            match = re.match(r"step(\d+)$", child.name)
            if match and (child / "metrics_compare.json").exists():
                steps.append(int(match.group(1)))
        if steps:
            return str(max(steps))
    eval_metrics = run_path / "eval" / "metrics_compare.json"
    if eval_metrics.exists():
        ckpt = latest_checkpoint(run_dir)
        if ckpt:
            match = re.search(r"ckpt_step(\d+)\.pt$", ckpt)
            if match:
                return str(int(match.group(1)) + 1)
    return ""


def load_qstat_cache(job_ids: Iterable[str]) -> Dict[str, Dict[str, str]]:
    cache: Dict[str, Dict[str, str]] = {}
    for job_id in sorted({clean(x) for x in job_ids if clean(x)}):
        try:
            proc = subprocess.run(
                ["qstat", "-xf", job_id],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
        except FileNotFoundError:
            continue
        if proc.returncode != 0 or not proc.stdout:
            continue
        data: Dict[str, str] = {}
        for raw_line in proc.stdout.splitlines():
            line = raw_line.strip()
            if " = " not in line:
                continue
            key, value = line.split(" = ", 1)
            data[key.strip()] = value.strip()
        cache[job_id] = data
    return cache


def job_state_to_status(job_state: str, exit_status: str) -> str:
    if job_state == "R":
        return "running"
    if job_state in {"Q", "H", "B"}:
        return "queued"
    if job_state == "F":
        return "completed" if clean(exit_status) == "0" else "failed"
    return "missing"


def scan_job_info_attempts() -> List[Dict[str, str]]:
    attempts: List[Dict[str, str]] = []
    for runs_dir_name, package in RUN_DIR_PACKAGE_MAP.items():
        runs_dir = ROOT / runs_dir_name
        if not runs_dir.exists():
            continue
        for job_info_path in sorted(runs_dir.rglob("job_info.json")):
            try:
                info = json.loads(job_info_path.read_text())
            except json.JSONDecodeError:
                continue
            run_dir = clean(info.get("run_dir")) or str(job_info_path.parent)
            train_job_id = clean(info.get("train_job_id"))
            attempt_id = train_job_id or f"local::{Path(run_dir).name}"
            attempts.append(
                {
                    "experiment_package": package,
                    "exp_id": clean(info.get("exp_id")),
                    "group": clean(info.get("group")),
                    "train_mode": clean(info.get("train_mode")),
                    "run_dir": run_dir,
                    "train_job_id": train_job_id,
                    "scheduler_attempt_id": attempt_id,
                    "attempt_kind": "scheduler" if train_job_id else "local",
                    "submitted_at": clean(info.get("train_submitted_at_iso")),
                    "started_at": clean(info.get("train_started_at_iso")),
                    "finished_at": clean(info.get("train_finished_at_iso")),
                    "job_info_status": clean(info.get("train_status")),
                    "latest_checkpoint": latest_checkpoint(run_dir),
                    "latest_eval_step": latest_eval_step(run_dir),
                    "notes": "",
                }
            )
    return attempts


def scan_manifest_attempts() -> List[Dict[str, str]]:
    manifest_specs = [
        (PHASE1_SUBMITTED, "phase1_screening"),
        (PHASE2_SUBMITTED, "phase2_screening"),
        (ABLATION_500K_SUBMITTED, "mainline_500k"),
        (PTW_SUBMITTED, "pipeline_then_weighting"),
        (IMAGENET32_PHASEA_SUBMITTED, "imagenet32_phasea"),
        (OBJECTIVE_SUBMITTED, "objective_first_followup"),
    ]
    attempts: List[Dict[str, str]] = []
    for manifest_path, package in manifest_specs:
        for row in read_csv_rows(manifest_path):
            train_job_id = clean(row.get("train_job_id"))
            if not train_job_id:
                continue
            attempts.append(
                {
                    "experiment_package": package,
                    "exp_id": clean(row.get("exp_id")),
                    "group": clean(row.get("group")) or clean(row.get("phase")),
                    "train_mode": clean(row.get("train_mode")),
                    "run_dir": clean(row.get("train_run_dir")),
                    "train_job_id": train_job_id,
                    "scheduler_attempt_id": train_job_id,
                    "attempt_kind": "scheduler",
                    "submitted_at": clean(row.get("submitted_at")),
                    "started_at": "",
                    "finished_at": "",
                    "job_info_status": "",
                    "latest_checkpoint": latest_checkpoint(clean(row.get("train_run_dir"))),
                    "latest_eval_step": latest_eval_step(clean(row.get("train_run_dir"))),
                    "notes": "",
                }
            )
    return attempts


def dedupe_attempts(raw_attempts: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    merged: Dict[str, Dict[str, str]] = {}
    for attempt in raw_attempts:
        attempt_id = attempt["scheduler_attempt_id"]
        if attempt_id not in merged:
            merged[attempt_id] = dict(attempt)
            continue
        current = merged[attempt_id]
        for key, value in attempt.items():
            if clean(current.get(key)):
                continue
            current[key] = value
    return merged


def build_current_cifar_rows() -> List[Dict[str, str]]:
    submitted_by_exp = {row["exp_id"]: row for row in read_csv_rows(OBJECTIVE_SUBMITTED)}
    rows: List[Dict[str, str]] = []
    for row in read_csv_rows(OBJECTIVE_MANIFEST):
        submitted = submitted_by_exp.get(row["exp_id"], {})
        merged = dict(row)
        merged.update({k: v for k, v in submitted.items() if v})
        benchmark = infer_benchmark(merged)
        phase_group = clean(merged.get("phase")) or "current"
        family_id = family_id_for(merged, "current_cifar_paper", phase_group)
        logical_run_id = logical_run_id_for(
            "current_cifar_paper",
            "objective_first_followup",
            benchmark,
            family_id,
            extract_seed(merged),
            clean(merged.get("steps")),
        )
        rows.append(
            {
                "ledger": "current_cifar_paper",
                "phase_group": phase_group,
                "experiment_package": "objective_first_followup",
                "exp_id": merged["exp_id"],
                "family_id": family_id,
                "logical_run_id": logical_run_id,
                "benchmark": benchmark,
                "regime": normalize_regime(clean(merged.get("train_mode"))),
                "stage_count": clean(merged.get("pipe_stages")) or "1",
                "weighting": normalize_weighting(
                    clean(merged.get("train_mode")),
                    clean(merged.get("weight_mode")),
                    clean(merged.get("last2_beta")),
                    clean(merged.get("pipe_stages")) or "1",
                ),
                "K": clean(merged.get("K")),
                "horizon_steps": clean(merged.get("steps")),
                "seed": extract_seed(merged),
                "paper_role": paper_role_for("current_cifar_paper", phase_group, family_id),
                "planned_run_dir": clean(merged.get("train_run_dir")),
                "notes": clean(merged.get("notes")),
            }
        )
    return rows


def build_external_rows() -> List[Dict[str, str]]:
    submitted_by_exp = {row["exp_id"]: row for row in read_csv_rows(IMAGENET32_PHASEA_SUBMITTED)}
    rows: List[Dict[str, str]] = []
    for row in read_csv_rows(IMAGENET32_PHASEA_CURRENT):
        submitted = submitted_by_exp.get(row["exp_id"], {})
        merged = dict(row)
        merged.update({k: v for k, v in submitted.items() if v})
        family_id = family_id_for(merged, "external_benchmark", "PhaseA")
        logical_run_id = logical_run_id_for(
            "external_benchmark",
            "imagenet32_phasea",
            "imagenet32",
            family_id,
            extract_seed(merged),
            clean(merged.get("steps")),
        )
        rows.append(
            {
                "ledger": "external_benchmark",
                "phase_group": "PhaseA",
                "experiment_package": "imagenet32_phasea",
                "exp_id": merged["exp_id"],
                "family_id": family_id,
                "logical_run_id": logical_run_id,
                "benchmark": "imagenet32",
                "regime": normalize_regime(clean(merged.get("train_mode"))),
                "stage_count": clean(merged.get("pipe_stages")) or ("1" if clean(merged.get("train_mode")) in {"single_fullk", "ddp_fullk"} else ""),
                "weighting": normalize_weighting(
                    clean(merged.get("train_mode")),
                    clean(merged.get("weight_mode")) or "uniform",
                    clean(merged.get("last2_beta")),
                    clean(merged.get("pipe_stages")) or "1",
                ),
                "K": clean(merged.get("K")) or "100",
                "horizon_steps": clean(merged.get("steps")),
                "seed": extract_seed(merged),
                "paper_role": "external",
                "planned_run_dir": clean(merged.get("run_dir")) or clean(merged.get("train_run_dir")),
                "notes": "External conditional benchmark infrastructure validation.",
            }
        )
    return rows


def build_historical_rows_from_summary(path: Path, package: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for row in read_csv_rows(path):
        benchmark = infer_benchmark(row)
        if benchmark == "unknown":
            benchmark = "cifar10"
        phase_group = clean(row.get("group")) or package
        family_id = family_id_for(row, "historical_exploratory", phase_group)
        seed = extract_seed(row)
        logical_run_id = (
            f"historical_exploratory:{package}:{row['exp_id']}:{family_id}:seed{seed}:h{clean(row.get('steps'))}"
        )
        rows.append(
            {
                "ledger": "historical_exploratory",
                "phase_group": phase_group,
                "experiment_package": package,
                "exp_id": row["exp_id"],
                "family_id": family_id,
                "logical_run_id": logical_run_id,
                "benchmark": benchmark,
                "regime": normalize_regime(clean(row.get("train_mode"))),
                "stage_count": clean(row.get("pipe_stages")) or "1",
                "weighting": normalize_weighting(
                    clean(row.get("train_mode")),
                    clean(row.get("weight_mode")),
                    clean(row.get("last2_beta")),
                    clean(row.get("pipe_stages")) or "1",
                ),
                "K": clean(row.get("K")),
                "horizon_steps": clean(row.get("steps")),
                "seed": seed,
                "paper_role": "deprecated",
                "planned_run_dir": clean(row.get("run_dir")),
                "notes": f"Historical exploratory package {package}; exclude from current paper counts.",
            }
        )
    return rows


def build_all_logical_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    rows.extend(build_current_cifar_rows())
    rows.extend(build_external_rows())
    rows.extend(build_historical_rows_from_summary(PHASE1_SUMMARY, "phase1_screening"))
    rows.extend(build_historical_rows_from_summary(PHASE2_SUMMARY, "phase2_screening"))
    rows.extend(build_historical_rows_from_summary(ABLATION_500K_SUMMARY, "mainline_500k"))
    rows.extend(build_historical_rows_from_summary(PIPELINE_THEN_WEIGHTING_CURRENT, "pipeline_then_weighting"))
    return rows


def build_logical_index(logical_rows: Iterable[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    return {(row["experiment_package"], row["exp_id"]): row for row in logical_rows}


def infer_package_from_run_dir(run_dir: str) -> str:
    for prefix, package in RUN_DIR_PACKAGE_MAP.items():
        if f"/{prefix}/" in run_dir:
            return package
    return ""


def finalize_attempts(
    attempts: Dict[str, Dict[str, str]],
    logical_index: Dict[Tuple[str, str], Dict[str, str]],
) -> Dict[str, Dict[str, str]]:
    qstat_cache = load_qstat_cache(attempt["train_job_id"] for attempt in attempts.values())
    finalized: Dict[str, Dict[str, str]] = {}
    for attempt_id, attempt in attempts.items():
        package = attempt.get("experiment_package") or infer_package_from_run_dir(attempt.get("run_dir", ""))
        exp_id = attempt.get("exp_id", "")
        logical = logical_index.get((package, exp_id))
        if not logical:
            # Ignore execution fragments that do not belong to the canonical ledgers.
            continue
        qinfo = qstat_cache.get(clean(attempt.get("train_job_id")), {})
        job_state = clean(qinfo.get("job_state"))
        exit_status = clean(qinfo.get("Exit_status"))
        if job_state:
            attempt_status = job_state_to_status(job_state, exit_status)
        else:
            job_info_status = clean(attempt.get("job_info_status"))
            attempt_status = {
                "succeeded": "completed",
                "completed": "completed",
                "running": "running",
                "queued": "queued",
            }.get(job_info_status, "failed" if job_info_status else "missing")
        notes = attempt.get("notes", "")
        if clean(qinfo.get("comment")):
            notes = f"{notes}; {clean(qinfo.get('comment'))}".strip("; ")
        finalized[attempt_id] = {
            **attempt,
            "experiment_package": package,
            "ledger": logical["ledger"],
            "phase_group": logical["phase_group"],
            "family_id": logical["family_id"],
            "logical_run_id": logical["logical_run_id"],
            "benchmark": logical["benchmark"],
            "regime": logical["regime"],
            "stage_count": logical["stage_count"],
            "weighting": logical["weighting"],
            "K": logical["K"],
            "horizon_steps": logical["horizon_steps"],
            "seed": logical["seed"],
            "attempt_status": attempt_status,
            "scheduler_state": job_state,
            "exit_status": exit_status,
            "notes": notes,
        }
    return finalized


def summarize_logical_run(row: Dict[str, str], attempts_for_run: List[Dict[str, str]]) -> Dict[str, str]:
    status = "missing"
    canonical_run_dir = clean(row.get("planned_run_dir"))
    latest_ckpt = latest_checkpoint(canonical_run_dir)
    latest_eval = latest_eval_step(canonical_run_dir)
    notes = [clean(row.get("notes"))]
    if attempts_for_run:
        completed = [a for a in attempts_for_run if a["attempt_status"] == "completed"]
        running = [a for a in attempts_for_run if a["attempt_status"] == "running"]
        queued = [a for a in attempts_for_run if a["attempt_status"] == "queued"]
        failed = [a for a in attempts_for_run if a["attempt_status"] == "failed"]
        if completed:
            status = "completed"
            chosen = sorted(completed, key=lambda a: (a.get("finished_at", ""), a["run_dir"]))[-1]
        elif running:
            status = "running"
            chosen = sorted(running, key=lambda a: (a.get("started_at", ""), a["run_dir"]))[-1]
        elif queued:
            status = "queued"
            chosen = sorted(queued, key=lambda a: (a.get("submitted_at", ""), a["run_dir"]))[-1]
        elif failed:
            status = "failed"
            chosen = sorted(
                failed,
                key=lambda a: (
                    checkpoint_step_from_path(a.get("latest_checkpoint", "")),
                    a.get("finished_at", ""),
                    a["run_dir"],
                ),
            )[-1]
        else:
            chosen = None
        if chosen:
            canonical_run_dir = clean(chosen.get("run_dir")) or canonical_run_dir
            latest_ckpt = clean(chosen.get("latest_checkpoint")) or latest_checkpoint(canonical_run_dir)
            latest_eval = clean(chosen.get("latest_eval_step")) or latest_eval_step(canonical_run_dir)
        if len(attempts_for_run) > 1:
            notes.append(f"deduped {len(attempts_for_run)} execution attempts")
        if failed and status == "completed":
            notes.append(f"{len(failed)} earlier failed/resume fragments excluded from logical count")
        elif failed:
            notes.append(f"{len(failed)} failed attempt(s)")
    return {
        "ledger": row["ledger"],
        "phase_group": row["phase_group"],
        "family_id": row["family_id"],
        "logical_run_id": row["logical_run_id"],
        "scheduler_attempt_count": str(len(attempts_for_run)),
        "benchmark": row["benchmark"],
        "regime": row["regime"],
        "stage_count": row["stage_count"],
        "weighting": row["weighting"],
        "K": row["K"],
        "horizon_steps": row["horizon_steps"],
        "seed": row["seed"],
        "status": status,
        "paper_role": row["paper_role"],
        "canonical_run_dir": canonical_run_dir,
        "latest_checkpoint": latest_ckpt,
        "latest_eval_step": latest_eval,
        "notes": "; ".join(filter(None, notes)),
    }


def build_markdown_reports(
    master_rows: List[Dict[str, str]],
    attempts_rows: List[Dict[str, str]],
) -> Tuple[str, str]:
    lines: List[str] = []
    lines.append("# Paper Manifest Current")
    lines.append("")
    lines.append("Counts below always specify the unit explicitly.")
    lines.append("")

    ledger_counter = defaultdict(lambda: {"families": set(), "logical_runs": 0, "attempts": 0})
    for row in master_rows:
        bucket = ledger_counter[row["ledger"]]
        bucket["families"].add(row["family_id"])
        bucket["logical_runs"] += 1
    for row in attempts_rows:
        ledger_counter[row["ledger"]]["attempts"] += 1

    lines.append("## Counts By Ledger")
    lines.append("")
    lines.append("| ledger | family_count | logical_run_count | scheduler_attempt_count |")
    lines.append("| --- | ---: | ---: | ---: |")
    for ledger, data in sorted(ledger_counter.items()):
        lines.append(
            f"| {ledger} | {len(data['families'])} | {data['logical_runs']} | {data['attempts']} |"
        )
    lines.append("")

    current_rows = [row for row in master_rows if row["ledger"] == "current_cifar_paper"]
    phase_counts: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {"families": set(), "logical_runs": 0, "completed": 0, "running": 0, "queued": 0, "failed": 0, "missing": 0}
    )
    for row in current_rows:
        bucket = phase_counts[row["phase_group"]]
        bucket["families"].add(row["family_id"])
        bucket["logical_runs"] += 1
        bucket[row["status"]] += 1

    lines.append("## Counts By Current-CIFAR Phase")
    lines.append("")
    lines.append("| phase_group | family_count | logical_run_count | completed | running | queued | failed | missing |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for phase_group in sorted(phase_counts):
        data = phase_counts[phase_group]
        lines.append(
            f"| {phase_group} | {len(data['families'])} | {data['logical_runs']} | {data['completed']} | {data['running']} | {data['queued']} | {data['failed']} | {data['missing']} |"
        )
    lines.append("")

    family_phase_status: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for row in current_rows:
        family_phase_status[(row["phase_group"], row["family_id"])].append(row["status"])

    complete_now: List[str] = []
    blocked_now: List[str] = []
    for (phase_group, family_id), statuses in sorted(family_phase_status.items()):
        label = f"{phase_group}: {family_id}"
        if all(status == "completed" for status in statuses):
            complete_now.append(label)
        else:
            counts = Counter(statuses)
            blocked_now.append(
                f"{label} ({', '.join(f'{k}={v}' for k, v in sorted(counts.items()))})"
            )

    lines.append("## Current-CIFAR Families Complete Enough For Paper Writing Now")
    lines.append("")
    if complete_now:
        for item in complete_now:
            lines.append(f"- `{item}`")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Current-CIFAR Families Still Blocked")
    lines.append("")
    if blocked_now:
        for item in blocked_now:
            lines.append(f"- `{item}`")
    else:
        lines.append("- None")
    lines.append("")

    diff_lines: List[str] = []
    diff_lines.append("# Manifest Diff Vs Claimed Plan")
    diff_lines.append("")
    diff_lines.append("This report compares the claimed O1-O5 plan against actual filesystem / scheduler state.")
    diff_lines.append("")
    diff_lines.append("## Actual State For O1-O5 Logical Runs")
    diff_lines.append("")
    diff_lines.append("| phase_group | completed | running | queued | failed | missing |")
    diff_lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for phase_group in sorted(phase_counts):
        data = phase_counts[phase_group]
        diff_lines.append(
            f"| {phase_group} | {data['completed']} | {data['running']} | {data['queued']} | {data['failed']} | {data['missing']} |"
        )
    diff_lines.append("")

    duplicated = [row for row in master_rows if int(row["scheduler_attempt_count"] or "0") > 1]
    diff_lines.append("## Duplicated Logical Runs (multiple attempts / resume fragments)")
    diff_lines.append("")
    if duplicated:
        diff_lines.append("| logical_run_id | attempts | status | canonical_run_dir |")
        diff_lines.append("| --- | ---: | --- | --- |")
        for row in sorted(duplicated, key=lambda r: r["logical_run_id"]):
            diff_lines.append(
                f"| `{row['logical_run_id']}` | {row['scheduler_attempt_count']} | {row['status']} | `{row['canonical_run_dir']}` |"
            )
    else:
        diff_lines.append("- None")
    diff_lines.append("")

    missing = [row for row in current_rows if row["status"] == "missing"]
    diff_lines.append("## Missing Current-CIFAR Logical Runs")
    diff_lines.append("")
    if missing:
        for row in sorted(missing, key=lambda r: (r["phase_group"], r["family_id"], r["seed"])):
            diff_lines.append(
                f"- `{row['phase_group']}: {row['family_id']} seed={row['seed']} horizon={row['horizon_steps']}`"
            )
    else:
        diff_lines.append("- None")
    diff_lines.append("")

    historical_packages = sorted(
        {
            row["phase_group"]
            for row in master_rows
            if row["ledger"] == "historical_exploratory"
        }
    )
    diff_lines.append("## Historical Packages Excluded From Current Paper Ledger")
    diff_lines.append("")
    for item in historical_packages:
        diff_lines.append(f"- `{item}`")
    diff_lines.append("")

    cross_package_duplicates: Dict[Tuple[str, str, str, str], List[str]] = defaultdict(list)
    for row in master_rows:
        key = (row["benchmark"], row["family_id"], row["seed"], row["horizon_steps"])
        cross_package_duplicates[key].append(f"{row['ledger']}::{row['phase_group']}")
    duplicate_keys = {k: v for k, v in cross_package_duplicates.items() if len(set(v)) > 1}
    diff_lines.append("## Cross-Ledger / Cross-Package Duplicate Identities")
    diff_lines.append("")
    if duplicate_keys:
        for (benchmark, family_id, seed, horizon), locations in sorted(duplicate_keys.items()):
            diff_lines.append(
                f"- `{benchmark} / {family_id} / seed={seed} / horizon={horizon}` appears in {', '.join(sorted(set(locations)))}"
            )
    else:
        diff_lines.append("- None")
    diff_lines.append("")

    return "\n".join(lines) + "\n", "\n".join(diff_lines) + "\n"


def build_master_registry() -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    logical_rows = build_all_logical_rows()
    logical_index = build_logical_index(logical_rows)

    raw_attempts = scan_job_info_attempts() + scan_manifest_attempts()
    deduped_attempts = dedupe_attempts(raw_attempts)
    finalized_attempts = finalize_attempts(deduped_attempts, logical_index)

    attempts_by_logical: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for attempt in finalized_attempts.values():
        attempts_by_logical[attempt["logical_run_id"]].append(attempt)

    master_rows: List[Dict[str, str]] = []
    for row in logical_rows:
        master_rows.append(summarize_logical_run(row, attempts_by_logical.get(row["logical_run_id"], [])))

    attempt_rows = [
        {
            "ledger": attempt["ledger"],
            "phase_group": attempt["phase_group"],
            "family_id": attempt["family_id"],
            "logical_run_id": attempt["logical_run_id"],
            "scheduler_attempt_id": attempt["scheduler_attempt_id"],
            "exp_id": attempt["exp_id"],
            "benchmark": attempt["benchmark"],
            "regime": attempt["regime"],
            "stage_count": attempt["stage_count"],
            "weighting": attempt["weighting"],
            "K": attempt["K"],
            "horizon_steps": attempt["horizon_steps"],
            "seed": attempt["seed"],
            "attempt_kind": attempt["attempt_kind"],
            "attempt_status": attempt["attempt_status"],
            "scheduler_state": attempt["scheduler_state"],
            "exit_status": attempt["exit_status"],
            "train_job_id": attempt["train_job_id"],
            "submitted_at": attempt["submitted_at"],
            "started_at": attempt["started_at"],
            "finished_at": attempt["finished_at"],
            "run_dir": attempt["run_dir"],
            "latest_checkpoint": attempt["latest_checkpoint"],
            "latest_eval_step": attempt["latest_eval_step"],
            "notes": attempt["notes"],
        }
        for attempt in sorted(
            finalized_attempts.values(),
            key=lambda a: (a["ledger"], a["phase_group"], a["family_id"], a["seed"], a["scheduler_attempt_id"]),
        )
    ]

    return master_rows, attempt_rows


def main() -> None:
    master_rows, attempt_rows = build_master_registry()
    write_csv(MASTER_OUT, MASTER_FIELDS, master_rows)
    write_csv(ATTEMPTS_OUT, ATTEMPT_FIELDS, attempt_rows)
    paper_md, diff_md = build_markdown_reports(master_rows, attempt_rows)
    PAPER_MANIFEST_MD.write_text(paper_md, encoding="utf-8")
    MANIFEST_DIFF_MD.write_text(diff_md, encoding="utf-8")
    print(f"Wrote {MASTER_OUT}")
    print(f"Wrote {ATTEMPTS_OUT}")
    print(f"Wrote {PAPER_MANIFEST_MD}")
    print(f"Wrote {MANIFEST_DIFF_MD}")


if __name__ == "__main__":
    main()
