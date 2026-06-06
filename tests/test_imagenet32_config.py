from pathlib import Path

import yaml

from polaris_ebm.scripts.current.benchmark_config import (
    load_benchmark_config,
    validate_pipeline_contract,
)


def test_load_benchmark_config_reads_conditional_imagenet32_yaml(tmp_path: Path):
    cfg_path = tmp_path / "imagenet32.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "name": "imagenet32",
                    "conditional": True,
                    "num_classes": 1000,
                    "image_size": 32,
                    "data_root": "/tmp/imagenet32",
                },
                "train": {"steps": 20000, "batch_size": 128, "K": 100, "lr": 1e-4},
                "pipeline": {"pipe_stages": 4, "weight_mode": "last2_beta", "last2_beta": 0.01},
            }
        )
    )

    cfg = load_benchmark_config(cfg_path)
    assert cfg.benchmark.name == "imagenet32"
    assert cfg.benchmark.conditional is True
    assert cfg.benchmark.num_classes == 1000
    assert cfg.train.K == 100
    assert cfg.pipeline.pipe_stages == 4


def test_validate_pipeline_contract_computes_nonuniform_k_slices():
    info = validate_pipeline_contract(total_k=100, pipe_stages=8, stage_weights=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.99])
    assert info["k_slices"] == [13, 13, 13, 13, 12, 12, 12, 12]
    assert abs(sum(info["stage_weights"]) - 1.0) < 1e-12


def test_load_benchmark_config_defaults_imagenet32_val_split(tmp_path: Path):
    cfg_path = tmp_path / "imagenet32_defaults.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "name": "imagenet32",
                    "conditional": True,
                    "num_classes": 1000,
                    "data_root": "/tmp/imagenet32",
                },
                "train": {"steps": 1000, "batch_size": 32, "K": 100, "lr": 1e-4},
                "pipeline": {"pipe_stages": 1, "stage_weights": [1.0]},
            }
        )
    )

    cfg = load_benchmark_config(cfg_path)

    assert cfg.benchmark.split_train == "train"
    assert cfg.benchmark.split_val == "val"
