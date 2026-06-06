#!/usr/bin/env python3
"""Shared YAML config loading and semantic validation for large benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml
except Exception:  # pragma: no cover - optional import guard
    yaml = None

try:
    from ablation_common import compute_k_slices
except Exception:
    from polaris_ebm.scripts.current.ablation_common import compute_k_slices


@dataclass
class BenchmarkSection:
    name: str
    conditional: bool = False
    num_classes: int = 0
    image_size: int = 32
    data_root: str = ""
    split_train: str = "train"
    split_val: str = "val"


@dataclass
class TrainSection:
    steps: int
    batch_size: int
    K: int
    lr: float


@dataclass
class PipelineSection:
    pipe_stages: int = 1
    weight_mode: str = "uniform"
    last2_beta: float = 0.01
    stage_weights: List[float] = field(default_factory=list)


@dataclass
class BenchmarkConfig:
    benchmark: BenchmarkSection
    train: TrainSection
    pipeline: PipelineSection
    raw: Dict


def _require_yaml():
    if yaml is None:
        raise RuntimeError("PyYAML is required to load benchmark configs")


def _get_section(raw: Dict, name: str) -> Dict:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"config section '{name}' must be a mapping")
    return value


def validate_pipeline_contract(total_k: int, pipe_stages: int, stage_weights: Optional[List[float]] = None) -> Dict:
    if int(total_k) <= 0:
        raise ValueError("total_k must be > 0")
    if int(pipe_stages) <= 0:
        raise ValueError("pipe_stages must be > 0")
    weights = list(stage_weights or [1.0 for _ in range(int(pipe_stages))])
    if len(weights) != int(pipe_stages):
        raise ValueError("stage_weights length must match pipe_stages")
    total_w = float(sum(float(x) for x in weights))
    if total_w <= 0.0:
        raise ValueError("stage_weights must sum to a positive value")
    normalized = [float(x) / total_w for x in weights]
    if abs(sum(normalized) - 1.0) > 1e-9:
        raise ValueError("normalized stage_weights must sum to 1")
    k_slices = compute_k_slices(int(total_k), int(pipe_stages))
    if int(sum(k_slices)) != int(total_k):
        raise ValueError("K slice allocation must sum to total_k")
    return {"k_slices": k_slices, "stage_weights": normalized}


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    _require_yaml()
    cfg_path = Path(path)
    raw = yaml.safe_load(cfg_path.read_text()) or {}
    benchmark_raw = _get_section(raw, "benchmark")
    train_raw = _get_section(raw, "train")
    pipe_raw = _get_section(raw, "pipeline")

    benchmark = BenchmarkSection(
        name=str(benchmark_raw["name"]),
        conditional=bool(benchmark_raw.get("conditional", False)),
        num_classes=int(benchmark_raw.get("num_classes", 0)),
        image_size=int(benchmark_raw.get("image_size", 32)),
        data_root=str(benchmark_raw.get("data_root", "")),
        split_train=str(benchmark_raw.get("split_train", "train")),
        split_val=str(benchmark_raw.get("split_val", "val")),
    )
    if benchmark.conditional and benchmark.num_classes <= 0:
        raise ValueError("conditional benchmark configs must set num_classes > 0")

    train = TrainSection(
        steps=int(train_raw["steps"]),
        batch_size=int(train_raw["batch_size"]),
        K=int(train_raw["K"]),
        lr=float(train_raw["lr"]),
    )
    pipeline = PipelineSection(
        pipe_stages=int(pipe_raw.get("pipe_stages", 1)),
        weight_mode=str(pipe_raw.get("weight_mode", "uniform")),
        last2_beta=float(pipe_raw.get("last2_beta", 0.01)),
        stage_weights=[float(x) for x in pipe_raw.get("stage_weights", [])],
    )

    validate_pipeline_contract(
        total_k=train.K,
        pipe_stages=pipeline.pipe_stages,
        stage_weights=pipeline.stage_weights or [1.0 for _ in range(pipeline.pipe_stages)],
    )

    return BenchmarkConfig(benchmark=benchmark, train=train, pipeline=pipeline, raw=raw)
