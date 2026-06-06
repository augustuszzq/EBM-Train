from argparse import Namespace
from pathlib import Path

from PIL import Image
import yaml

from polaris_ebm.scripts.current.eval_metrics import load_real_samples


def test_load_real_samples_uses_materialized_imagenet32_split(tmp_path: Path):
    data_root = tmp_path / "imagenet32"
    split_dir = data_root / "validation"
    (split_dir / "images" / "1").mkdir(parents=True)
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(split_dir / "images" / "1" / "00000000.png")
    (split_dir / "labels.csv").write_text("index,label,relpath\n0,1,1/00000000.png\n")

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "name": "imagenet32",
                    "conditional": True,
                    "num_classes": 1000,
                    "image_size": 32,
                    "data_root": str(data_root),
                    "split_train": "train",
                    "split_val": "validation",
                },
                "train": {"steps": 1000, "batch_size": 4, "K": 100, "lr": 1e-4},
                "pipeline": {"pipe_stages": 4, "weight_mode": "last2_beta", "last2_beta": 0.01},
            }
        )
    )

    args = Namespace(
        real_samples="",
        config=str(cfg_path),
        data_dir="",
        num_real=1,
    )
    real = load_real_samples(args)
    assert tuple(real.shape) == (1, 3, 32, 32)
