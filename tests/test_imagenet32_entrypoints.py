import json
import sys
from pathlib import Path

import torch

from polaris_ebm.scripts.current.benchmark_runtime import (
    build_energy_model,
    make_eval_label_schedule,
    resolve_benchmark_runtime,
)
from polaris_ebm.scripts.current.eval_generate import parse_args as parse_eval_generate_args
from polaris_ebm.scripts.current.ebm_train_sync_mode_a import parse_args as parse_sync_args


PROJECT_DIR = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")


def test_resolve_benchmark_runtime_defaults_to_legacy_cifar():
    runtime = resolve_benchmark_runtime(config_path="")
    assert runtime.name == "cifar10"
    assert runtime.conditional is False
    assert runtime.image_shape == (3, 32, 32)


def test_resolve_benchmark_runtime_uses_imagenet32_config():
    cfg = PROJECT_DIR / "configs" / "imagenet32_pipeline_strict.yaml"
    runtime = resolve_benchmark_runtime(config_path=str(cfg))
    assert runtime.name == "imagenet32"
    assert runtime.conditional is True
    assert runtime.num_classes == 1000
    assert runtime.image_shape == (3, 32, 32)


def test_build_energy_model_switches_to_conditional_variant():
    runtime = resolve_benchmark_runtime(config_path=str(PROJECT_DIR / "configs" / "imagenet32_single_strict.yaml"))
    model = build_energy_model(runtime=runtime, n_f=16)
    x = torch.randn((2, 3, 32, 32))
    y = torch.tensor([0, 1], dtype=torch.long)
    out = model(x, y)
    assert out.shape == (2,)


def test_make_eval_label_schedule_is_class_balanced():
    labels = make_eval_label_schedule(num_classes=4, images_per_class=3)
    assert labels.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_sync_entrypoint_accepts_config_arg(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ebm_train_sync_mode_a.py",
            "--config",
            str(PROJECT_DIR / "configs" / "imagenet32_ddp_strict.yaml"),
        ],
    )
    args = parse_sync_args()
    assert str(args.config).endswith("imagenet32_ddp_strict.yaml")


def test_eval_generate_accepts_config_and_images_per_class(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_generate.py",
            "--ckpt",
            "/tmp/mock.pt",
            "--out_dir",
            "/tmp/out",
            "--config",
            str(PROJECT_DIR / "configs" / "imagenet32_pipeline_strict.yaml"),
            "--images_per_class",
            "50",
        ],
    )
    args = parse_eval_generate_args()
    assert args.images_per_class == 50
    assert str(args.config).endswith("imagenet32_pipeline_strict.yaml")
