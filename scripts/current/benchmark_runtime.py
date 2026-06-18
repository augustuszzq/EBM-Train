#!/usr/bin/env python3
"""Shared runtime helpers for benchmark-aware training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Optional, Sequence

import torch as t

try:
    from torchvision import datasets, transforms
except Exception:  # pragma: no cover
    datasets = None
    transforms = None

try:
    from benchmark_config import BenchmarkConfig, load_benchmark_config
    from celebahq256_data import build_celebahq256_dataset
    from celebahq256_latent_data import build_celebahq256_latent_dataset
    from conditional_model import ConditionalEnergyModel
    from imagenet32_data import MaterializedImageNet32Dataset, get_benchmark_data_spec
except Exception:
    from polaris_ebm.scripts.current.benchmark_config import BenchmarkConfig, load_benchmark_config
    from polaris_ebm.scripts.current.celebahq256_data import build_celebahq256_dataset
    from polaris_ebm.scripts.current.celebahq256_latent_data import build_celebahq256_latent_dataset
    from polaris_ebm.scripts.current.conditional_model import ConditionalEnergyModel
    from polaris_ebm.scripts.current.imagenet32_data import (
        MaterializedImageNet32Dataset,
        get_benchmark_data_spec,
    )


@dataclass
class BenchmarkRuntime:
    name: str
    conditional: bool
    num_classes: int
    image_shape: tuple[int, int, int]
    data_root: str
    split_train: str
    split_val: str
    config: Optional[BenchmarkConfig] = None


def resolve_benchmark_runtime(config_path: str = "", data_dir: str = "./data/cifar10") -> BenchmarkRuntime:
    text = str(config_path).strip()
    if not text:
        return BenchmarkRuntime(
            name="cifar10",
            conditional=False,
            num_classes=0,
            image_shape=(3, 32, 32),
            data_root=str(data_dir),
            split_train="train",
            split_val="train",
            config=None,
        )
    cfg = load_benchmark_config(text)
    spec = get_benchmark_data_spec(cfg.benchmark.name)
    channels = 4 if spec.name == "celebahq256_latent" else 3
    return BenchmarkRuntime(
        name=spec.name,
        conditional=spec.conditional,
        num_classes=spec.num_classes,
        image_shape=(channels, spec.image_size, spec.image_size),
        data_root=cfg.benchmark.data_root,
        split_train=cfg.benchmark.split_train,
        split_val=cfg.benchmark.split_val,
        config=cfg,
    )


def build_energy_model(runtime: BenchmarkRuntime, n_f: int, unconditional_cls=None):
    if runtime.conditional:
        return ConditionalEnergyModel(n_f=n_f, num_classes=runtime.num_classes, image_size=runtime.image_shape[-1])
    if unconditional_cls is None:
        raise ValueError("unconditional_cls is required for unconditional benchmark runtime")
    signature = inspect.signature(unconditional_cls)
    kwargs = {"n_f": n_f}
    if "n_c" in signature.parameters:
        kwargs["n_c"] = int(runtime.image_shape[0])
    if "image_size" in signature.parameters:
        kwargs["image_size"] = runtime.image_shape[-1]
    return unconditional_cls(**kwargs)


def energy_call(model, x: t.Tensor, labels: Optional[t.Tensor] = None) -> t.Tensor:
    if labels is None:
        return model(x)
    return model(x, labels)


def make_eval_label_schedule(num_classes: int, images_per_class: int) -> t.Tensor:
    if int(num_classes) <= 0:
        raise ValueError("num_classes must be > 0")
    if int(images_per_class) <= 0:
        raise ValueError("images_per_class must be > 0")
    return t.arange(int(num_classes), dtype=t.long).repeat_interleave(int(images_per_class))


def build_dataset(runtime: BenchmarkRuntime, cifar_builder=None, train: bool = True):
    if runtime.name == "cifar10":
        if cifar_builder is None:
            if datasets is None or transforms is None:
                raise RuntimeError("torchvision is required for CIFAR10")
            tfm = transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            return datasets.CIFAR10(root=runtime.data_root, train=bool(train), download=False, transform=tfm)
        return cifar_builder(runtime.data_root)
    if runtime.name == "imagenet32":
        split = runtime.split_train if train else runtime.split_val
        return MaterializedImageNet32Dataset(Path(runtime.data_root) / split)
    if runtime.name == "celeba64":
        if datasets is None or transforms is None:
            raise RuntimeError("torchvision is required for CelebA-64 fallback")
        tfm = transforms.Compose(
            [
                transforms.CenterCrop(178),
                transforms.Resize(runtime.image_shape[-1]),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        split = "train" if train else "valid"
        return datasets.CelebA(root=runtime.data_root, split=split, target_type="identity", download=False, transform=tfm)
    if runtime.name == "celebahq256":
        split = runtime.split_train if train else runtime.split_val
        return build_celebahq256_dataset(
            root=runtime.data_root,
            split=split,
            image_size=runtime.image_shape[-1],
        )
    if runtime.name == "celebahq256_latent":
        split = runtime.split_train if train else runtime.split_val
        return build_celebahq256_latent_dataset(root=runtime.data_root, split=split)
    raise ValueError(f"unsupported benchmark runtime: {runtime.name}")


def unpack_batch(batch, device: t.device, conditional: bool):
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        x = batch[0].to(device, non_blocking=True)
        y = batch[1]
        if not t.is_tensor(y):
            y = t.as_tensor(y, dtype=t.long)
        y = y.to(device, non_blocking=True).long()
        return x, (y if conditional else None)
    x = batch.to(device, non_blocking=True)
    return x, None
