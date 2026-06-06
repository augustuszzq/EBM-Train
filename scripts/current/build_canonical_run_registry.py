#!/usr/bin/env python3
"""Build a canonical run registry for the current EBM experiment bundle."""

import csv
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


ROOT = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm")
BUNDLE = ROOT / "runs_final_bundle"
EXPERIMENTS = ROOT / "experiments"

PHASE1_SUMMARY = BUNDLE / "phase1_summary.csv"
PHASE2_SUMMARY = BUNDLE / "phase2_summary.csv"
ABLATION_500K_SUMMARY = BUNDLE / "ablation_500k_summary.csv"
PTW_CURRENT = BUNDLE / "pipeline_then_weighting_current.csv"
PTW_REPLAY = BUNDLE / "pipeline_then_weighting_replay_summary.csv"
PTW_MANIFEST = EXPERIMENTS / "pipeline_then_weighting_manifest.csv"
IMAGENET32_PHASEA = BUNDLE / "imagenet32_phasea_current.csv"
OUT_PATH = BUNDLE / "canonical_run_registry.csv"


OUT_FIELDS = [
    "registry_id",
    "source_table",
    "experiment_package",
    "benchmark",
    "conditioned",
    "story_bucket",
    "include_in_current_story",
    "exp_id",
    "group",
    "phase",
    "story",
    "train_mode",
    "execution_semantics",
    "system_role",
    "objective_family",
    "world_size",
    "pipe_stages",
    "seed",
    "K",
    "steps",
    "lr",
    "step_size",
    "weight_mode",
    "last2_beta",
    "num_nodes",
    "ppn",
    "data_dir",
    "run_dir",
    "eval_dir",
    "trajectory_dir",
    "source_status",
    "registry_status",
    "final_fid",
    "best_fid",
    "best_step",
    "final_step",
    "available_points",
    "top1_conditional_acc",
    "per_class_avg_acc",
    "pbs_job_id",
    "local_run",
    "config_signature",
    "config_hash",
    "git_commit_recorded",
    "identity_note",
]


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def clean(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def execution_semantics(train_mode: str) -> str:
    mapping = {
        "single_fullk": "single_strict",
        "ddp_fullk": "ddp_strict",
        "single_pipe_emul": "single_emulation",
        "pipe_strict": "strict_pipeline",
    }
    return mapping.get(train_mode, train_mode)


def system_role(train_mode: str) -> str:
    mapping = {
        "single_fullk": "baseline_control",
        "ddp_fullk": "baseline_control",
        "single_pipe_emul": "objective_emulation",
        "pipe_strict": "pipeline_realization",
    }
    return mapping.get(train_mode, train_mode)


def objective_family(train_mode: str, weight_mode: str) -> str:
    if train_mode in {"single_fullk", "ddp_fullk"}:
        return "strict_baseline"
    if weight_mode == "uniform":
        return "uniform_weighted"
    if weight_mode == "deep_only":
        return "deep_only_weighted"
    if weight_mode == "last2_beta":
        return "late_stage_weighted"
    return weight_mode or "unknown"


def normalize_status(source_status: str) -> str:
    status = clean(source_status)
    if status in {"reference", "done", "done_local"}:
        return "completed"
    if status == "F:0":
        return "completed"
    if status.startswith("F:"):
        return "historical_failed"
    if status in {"Q", "R", "H"}:
        return "incomplete"
    return status or "unknown"


def include_in_current_story(experiment_package: str, registry_status: str) -> str:
    if registry_status != "completed":
        return "no"
    if experiment_package in {"pipeline_then_weighting", "imagenet32_phasea"}:
        return "yes"
    return "context_only"


def config_signature(row: Mapping[str, str]) -> str:
    parts = [
        clean(row.get("benchmark")),
        clean(row.get("conditioned")),
        clean(row.get("train_mode")),
        clean(row.get("world_size")),
        clean(row.get("pipe_stages")),
        clean(row.get("seed")),
        clean(row.get("K")),
        clean(row.get("steps")),
        clean(row.get("lr")),
        clean(row.get("step_size")),
        clean(row.get("weight_mode")),
        clean(row.get("last2_beta")),
        clean(row.get("data_dir")),
        clean(row.get("story_bucket")),
    ]
    return "|".join(parts)


def config_hash(signature: str) -> str:
    return hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]


def normalize_ptw_story_bucket(exp_id: str) -> str:
    if exp_id.startswith("E1_single_fullk") or exp_id.startswith("E1_ddp_fullk"):
        return "objective_first_controls"
    if exp_id.startswith("E1_single_pipe_emul"):
        return "objective_first_p_sweep_single"
    if exp_id.startswith("E1_pipe_strict"):
        return "objective_first_p_sweep_pipeline"
    if exp_id.startswith("E2_single_pipe_emul"):
        return "objective_first_weighting_single"
    if exp_id.startswith("E2_pipe_strict"):
        return "objective_first_weighting_pipeline"
    return "pipeline_then_weighting"


def phase_story_bucket(group: str, source_name: str) -> str:
    if source_name == "phase1_summary":
        return {
            "A0": "legacy_anchor",
            "A1": "legacy_weighting_screen",
            "A2": "legacy_pipeline_factor_screen",
            "A3": "legacy_chain_depth_screen",
            "A4": "legacy_step_size_screen",
        }.get(group, "legacy_screening")
    if source_name == "phase2_summary":
        return {
            "P2A0": "legacy_anchor",
            "M1": "legacy_seed_repro",
            "M2": "legacy_high_k_control",
            "S1": "legacy_single_gpu_mechanism",
            "S2": "legacy_stage_count",
        }.get(group, "legacy_phase2")
    if source_name == "ablation_500k_summary":
        return "legacy_long_horizon"
    return "legacy_context"


def load_ptw_replay() -> Dict[str, Dict[str, str]]:
    rows = read_csv_rows(PTW_REPLAY)
    return {row["exp_id"]: row for row in rows}


def load_ptw_manifest() -> Dict[str, Dict[str, str]]:
    rows = read_csv_rows(PTW_MANIFEST)
    return {row["exp_id"]: row for row in rows}


def build_legacy_rows(summary_path: Path, source_name: str, experiment_package: str) -> Iterable[Dict[str, str]]:
    for row in read_csv_rows(summary_path):
        story_bucket = phase_story_bucket(row.get("group", ""), source_name)
        out = {
            "registry_id": f"{experiment_package}:{row['exp_id']}",
            "source_table": source_name,
            "experiment_package": experiment_package,
            "benchmark": "cifar10",
            "conditioned": "no",
            "story_bucket": story_bucket,
            "exp_id": row["exp_id"],
            "group": row.get("group", ""),
            "phase": row.get("group", ""),
            "story": row.get("story", ""),
            "train_mode": row.get("train_mode", ""),
            "execution_semantics": execution_semantics(row.get("train_mode", "")),
            "system_role": system_role(row.get("train_mode", "")),
            "objective_family": objective_family(row.get("train_mode", ""), row.get("weight_mode", "")),
            "world_size": row.get("world_size", ""),
            "pipe_stages": row.get("pipe_stages", ""),
            "seed": "1" if "seed" not in row else row.get("seed", ""),
            "K": row.get("K", ""),
            "steps": row.get("steps", ""),
            "lr": row.get("lr", ""),
            "step_size": row.get("step_size", ""),
            "weight_mode": row.get("weight_mode", ""),
            "last2_beta": row.get("last2_beta", ""),
            "num_nodes": "",
            "ppn": "",
            "data_dir": "/eagle/lc-mpi/Zhiqing/ebm/data/cifar10",
            "run_dir": row.get("run_dir", ""),
            "eval_dir": row.get("eval_dir", ""),
            "trajectory_dir": "",
            "source_status": row.get("status", ""),
            "registry_status": normalize_status(row.get("status", "")),
            "final_fid": row.get("fid_inception", ""),
            "best_fid": "",
            "best_step": "",
            "final_step": row.get("steps", ""),
            "available_points": "",
            "top1_conditional_acc": "",
            "per_class_avg_acc": "",
            "pbs_job_id": row.get("train_job_id", ""),
            "local_run": "no",
            "git_commit_recorded": "unavailable",
            "identity_note": "Registry row reconstructed from historical summary; per-run git commit was not recorded.",
        }
        out["include_in_current_story"] = include_in_current_story(experiment_package, out["registry_status"])
        out["config_signature"] = config_signature(out)
        out["config_hash"] = config_hash(out["config_signature"])
        yield {field: clean(out.get(field, "")) for field in OUT_FIELDS}


def build_ptw_rows() -> Iterable[Dict[str, str]]:
    replay_by_exp = load_ptw_replay()
    manifest_by_exp = load_ptw_manifest()
    for row in read_csv_rows(PTW_CURRENT):
        exp_id = row["exp_id"]
        manifest = manifest_by_exp.get(exp_id, {})
        replay = replay_by_exp.get(exp_id, {})
        story_bucket = normalize_ptw_story_bucket(exp_id)
        registry_status = normalize_status(row.get("status", ""))
        out = {
            "registry_id": f"pipeline_then_weighting:{exp_id}",
            "source_table": "pipeline_then_weighting_current",
            "experiment_package": "pipeline_then_weighting",
            "benchmark": "cifar10",
            "conditioned": "no",
            "story_bucket": story_bucket,
            "exp_id": exp_id,
            "group": row.get("group", ""),
            "phase": row.get("phase", ""),
            "story": manifest.get("story", ""),
            "train_mode": row.get("train_mode", ""),
            "execution_semantics": execution_semantics(row.get("train_mode", "")),
            "system_role": system_role(row.get("train_mode", "")),
            "objective_family": objective_family(row.get("train_mode", ""), row.get("weight_mode", "")),
            "world_size": manifest.get("world_size", ""),
            "pipe_stages": row.get("pipe_stages", ""),
            "seed": manifest.get("seed", ""),
            "K": manifest.get("K", ""),
            "steps": row.get("steps", ""),
            "lr": manifest.get("lr", ""),
            "step_size": manifest.get("step_size", ""),
            "weight_mode": row.get("weight_mode", ""),
            "last2_beta": row.get("last2_beta", ""),
            "num_nodes": manifest.get("num_nodes", ""),
            "ppn": manifest.get("ppn", ""),
            "data_dir": manifest.get("data_dir", ""),
            "run_dir": row.get("run_dir", ""),
            "eval_dir": str(Path(row.get("final_metrics_path", "")).parent) if row.get("final_metrics_path") else "",
            "trajectory_dir": str(Path(row.get("trajectory_csv", "")).parent) if row.get("trajectory_csv") else row.get("trajectory_csv", ""),
            "source_status": row.get("status", ""),
            "registry_status": registry_status,
            "final_fid": row.get("final_fid", ""),
            "best_fid": replay.get("best_fid", ""),
            "best_step": replay.get("best_step", ""),
            "final_step": replay.get("final_step", "") or row.get("last_trajectory_step", ""),
            "available_points": replay.get("available_points", ""),
            "top1_conditional_acc": "",
            "per_class_avg_acc": "",
            "pbs_job_id": row.get("pbs_job_id", ""),
            "local_run": row.get("local_run", ""),
            "git_commit_recorded": "unavailable",
            "identity_note": (
                "Canonical objective-first pilot row. "
                "Use these rows, not legacy beta001 summaries, for the current uniform/deep story."
            ),
        }
        out["include_in_current_story"] = include_in_current_story("pipeline_then_weighting", registry_status)
        out["config_signature"] = config_signature(out)
        out["config_hash"] = config_hash(out["config_signature"])
        yield {field: clean(out.get(field, "")) for field in OUT_FIELDS}


def build_imagenet_rows() -> Iterable[Dict[str, str]]:
    for row in read_csv_rows(IMAGENET32_PHASEA):
        registry_status = normalize_status(row.get("status", ""))
        out = {
            "registry_id": f"imagenet32_phasea:{row['exp_id']}",
            "source_table": "imagenet32_phasea_current",
            "experiment_package": "imagenet32_phasea",
            "benchmark": "imagenet32",
            "conditioned": "yes",
            "story_bucket": "conditional_phasea_smoke",
            "exp_id": row["exp_id"],
            "group": "PhaseA",
            "phase": "PhaseA",
            "story": "ImageNet-32 conditional smoke / stability validation",
            "train_mode": row.get("train_mode", ""),
            "execution_semantics": execution_semantics(row.get("train_mode", "")),
            "system_role": system_role(row.get("train_mode", "")),
            "objective_family": "conditional_smoke",
            "world_size": "",
            "pipe_stages": "4" if "pipeline" in row.get("exp_id", "") else "1",
            "seed": "1",
            "K": "100",
            "steps": row.get("steps", ""),
            "lr": "",
            "step_size": "",
            "weight_mode": "uniform" if "pipeline" not in row.get("exp_id", "") else "deep_only",
            "last2_beta": "",
            "num_nodes": "",
            "ppn": "",
            "data_dir": "/eagle/lc-mpi/Zhiqing/polaris_ebm/data/imagenet32",
            "run_dir": row.get("run_dir", ""),
            "eval_dir": "",
            "trajectory_dir": "",
            "source_status": row.get("status", ""),
            "registry_status": registry_status,
            "final_fid": row.get("final_fid", ""),
            "best_fid": "",
            "best_step": "",
            "final_step": row.get("steps", ""),
            "available_points": "",
            "top1_conditional_acc": row.get("top1_conditional_acc", ""),
            "per_class_avg_acc": row.get("per_class_avg_acc", ""),
            "pbs_job_id": row.get("train_job_id", ""),
            "local_run": "no",
            "git_commit_recorded": "unavailable",
            "identity_note": "Conditional ImageNet-32 smoke row; use for infrastructure validation, not for main quantitative claims.",
        }
        out["include_in_current_story"] = include_in_current_story("imagenet32_phasea", registry_status)
        out["config_signature"] = config_signature(out)
        out["config_hash"] = config_hash(out["config_signature"])
        yield {field: clean(out.get(field, "")) for field in OUT_FIELDS}


def build_rows() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    rows.extend(build_legacy_rows(PHASE1_SUMMARY, "phase1_summary", "phase1_screening"))
    rows.extend(build_legacy_rows(PHASE2_SUMMARY, "phase2_summary", "phase2_mainline"))
    rows.extend(build_legacy_rows(ABLATION_500K_SUMMARY, "ablation_500k_summary", "mainline_500k"))
    rows.extend(build_ptw_rows())
    rows.extend(build_imagenet_rows())
    return sorted(rows, key=lambda row: (row["experiment_package"], row["phase"], row["group"], row["exp_id"]))


def main() -> None:
    rows = build_rows()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[DONE] rows={len(rows)} out={OUT_PATH}")


if __name__ == "__main__":
    main()
