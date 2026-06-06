from pathlib import Path

import pytest
from PIL import Image

from polaris_ebm.scripts.current import imagenet32_data


def test_benchmark_data_spec_for_imagenet32():
    spec = imagenet32_data.get_benchmark_data_spec("imagenet32")
    assert spec.name == "imagenet32"
    assert spec.conditional is True
    assert spec.num_classes == 1000
    assert spec.image_size == 32


def test_imagenet32_loader_requires_datasets_dependency(monkeypatch):
    monkeypatch.setattr(imagenet32_data, "hf_datasets", None)
    with pytest.raises(RuntimeError, match="ChocolateDave/imagenet-32"):
        imagenet32_data.load_imagenet32_hf(split="train")


def test_materialize_imagenet32_is_restart_safe(tmp_path: Path, monkeypatch):
    rows = [
        {"image": Image.new("RGB", (32, 32), color=(255, 0, 0)), "label": 3},
        {"image": Image.new("RGB", (32, 32), color=(0, 255, 0)), "label": 7},
    ]

    monkeypatch.setattr(imagenet32_data, "load_imagenet32_hf", lambda split, **kwargs: rows)

    out = imagenet32_data.materialize_imagenet32(out_root=tmp_path, split="train")
    assert out["count"] == 2
    assert (tmp_path / "train" / "images" / "3" / "00000000.png").exists()
    assert (tmp_path / "train" / "_SUCCESS.json").exists()

    out2 = imagenet32_data.materialize_imagenet32(out_root=tmp_path, split="train")
    assert out2["count"] == 2
    assert out2["status"] == "cached"


def test_materialized_imagenet32_dataset_reads_labels_csv(tmp_path: Path):
    split_dir = tmp_path / "train"
    (split_dir / "images" / "3").mkdir(parents=True)
    (split_dir / "images" / "7").mkdir(parents=True)
    Image.new("RGB", (32, 32), color=(255, 0, 0)).save(split_dir / "images" / "3" / "00000000.png")
    Image.new("RGB", (32, 32), color=(0, 255, 0)).save(split_dir / "images" / "7" / "00000001.png")
    (split_dir / "labels.csv").write_text("index,label,relpath\n0,3,3/00000000.png\n1,7,7/00000001.png\n")

    ds = imagenet32_data.MaterializedImageNet32Dataset(split_dir=split_dir)
    x0, y0 = ds[0]
    x1, y1 = ds[1]
    assert tuple(x0.shape) == (3, 32, 32)
    assert y0 == 3
    assert y1 == 7


def test_load_imagenet32_hf_normalizes_validation_alias(monkeypatch):
    calls = []

    class _FakeDatasets:
        @staticmethod
        def load_dataset(repo, **kwargs):
            calls.append((repo, kwargs))
            return ["ok"]

    monkeypatch.setattr(imagenet32_data, "hf_datasets", _FakeDatasets())

    out = imagenet32_data.load_imagenet32_hf(split="validation")

    assert out == ["ok"]
    assert calls == [("ChocolateDave/imagenet-32", {"split": "val", "streaming": False})]


def test_materialize_imagenet32_honors_max_examples(tmp_path: Path, monkeypatch):
    rows = [
        {"image": Image.new("RGB", (32, 32), color=(255, 0, 0)), "label": 3},
        {"image": Image.new("RGB", (32, 32), color=(0, 255, 0)), "label": 7},
        {"image": Image.new("RGB", (32, 32), color=(0, 0, 255)), "label": 9},
    ]

    monkeypatch.setattr(imagenet32_data, "load_imagenet32_hf", lambda split, **kwargs: rows)

    out = imagenet32_data.materialize_imagenet32(out_root=tmp_path, split="train", max_examples=2)

    assert out["count"] == 2
    assert (tmp_path / "train" / "images" / "3" / "00000000.png").exists()
    assert (tmp_path / "train" / "images" / "7" / "00000001.png").exists()
    assert not (tmp_path / "train" / "images" / "9" / "00000002.png").exists()
