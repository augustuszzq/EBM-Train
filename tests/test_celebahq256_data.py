from pathlib import Path

import torch
from PIL import Image

import lmdb

from polaris_ebm.scripts.current.benchmark_config import load_benchmark_config
from polaris_ebm.scripts.current.benchmark_runtime import build_dataset, build_energy_model, resolve_benchmark_runtime
from polaris_ebm.scripts.current.celebahq256_data import build_celebahq256_dataset, get_celebahq256_data_spec
from polaris_ebm.scripts.current.ebm_train_sync_mode_a import EnergyModel

PROJECT_DIR = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm")


def _write_config(path: Path, data_root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "benchmark:",
                "  name: celebahq256",
                "  conditional: false",
                "  num_classes: 0",
                "  image_size: 256",
                f"  data_root: {data_root}",
                "  split_train: train",
                "  split_val: validation",
                "train:",
                "  steps: 10",
                "  batch_size: 2",
                "  K: 4",
                "  lr: 0.0001",
                "pipeline:",
                "  pipe_stages: 1",
                "  weight_mode: uniform",
                "",
            ]
        )
    )


def test_celebahq256_data_spec_matches_unconditional_256_benchmark():
    spec = get_celebahq256_data_spec()
    assert spec.name == "celebahq256"
    assert spec.conditional is False
    assert spec.num_classes == 0
    assert spec.image_size == 256


def test_celebahq256_image_folder_dataset_returns_normalized_tensor_and_dummy_label(tmp_path: Path):
    image_root = tmp_path / "celeba_hq_256"
    (image_root / "train").mkdir(parents=True)
    Image.new("RGB", (300, 280), color=(255, 0, 128)).save(image_root / "train" / "000001.png")
    cfg_path = tmp_path / "celebahq256.yaml"
    _write_config(cfg_path, image_root)

    runtime = resolve_benchmark_runtime(config_path=str(cfg_path))
    dataset = build_dataset(runtime=runtime, train=True)
    x, y = dataset[0]

    assert runtime.name == "celebahq256"
    assert runtime.image_shape == (3, 256, 256)
    assert tuple(x.shape) == (3, 256, 256)
    assert x.dtype == torch.float32
    assert float(x.min()) >= -1.0
    assert float(x.max()) <= 1.0
    assert int(y) == 0


def test_celebahq256_repository_configs_load():
    for name in [
        "celebahq256_single_strict.yaml",
        "celebahq256_ddp_strict.yaml",
        "celebahq256_pipeline_strict.yaml",
    ]:
        cfg = load_benchmark_config(PROJECT_DIR / "configs" / name)
        assert cfg.benchmark.name == "celebahq256"
        assert cfg.benchmark.conditional is False
        assert cfg.benchmark.image_size == 256


def test_celebahq256_lmdb_dataset_can_be_opened_twice_in_one_process(tmp_path: Path):
    root = tmp_path / "celeba-lmdb"
    root.mkdir(parents=True)
    env = lmdb.open(str(root / "train.lmdb"), map_size=int(1e8))
    image = Image.new("RGB", (256, 256), color=(255, 0, 128))
    with env.begin(write=True) as txn:
        txn.put(b"0", image.tobytes())
    env.close()

    ds0 = build_celebahq256_dataset(root, split="train", image_size=256)
    ds1 = build_celebahq256_dataset(root, split="train", image_size=256)

    x0, y0 = ds0[0]
    x1, y1 = ds1[0]
    assert tuple(x0.shape) == (3, 256, 256)
    assert tuple(x1.shape) == (3, 256, 256)
    assert int(y0) == 0
    assert int(y1) == 0


def test_original_unconditional_energy_model_returns_scalar_for_celebahq256_config():
    runtime = resolve_benchmark_runtime(PROJECT_DIR / "configs" / "celebahq256_ddp_strict.yaml")
    model = build_energy_model(runtime=runtime, n_f=8, unconditional_cls=EnergyModel)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert tuple(out.shape) == (2,)


def test_celebahq256_latent_dataset_returns_four_channel_32x32_latent(tmp_path: Path):
    latent_root = tmp_path / "celebahq256_latents"
    split_root = latent_root / "train"
    split_root.mkdir(parents=True)
    torch.save(
        {
            "latents": torch.randn(3, 4, 32, 32, dtype=torch.float16),
            "labels": torch.zeros(3, dtype=torch.long),
            "meta": {"source_image_size": 256, "latent_scaling_factor": 0.18215},
        },
        split_root / "latents.pt",
    )
    cfg_path = tmp_path / "celebahq256_latent.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "benchmark:",
                "  name: celebahq256_latent",
                "  conditional: false",
                "  num_classes: 0",
                "  image_size: 32",
                f"  data_root: {latent_root}",
                "  split_train: train",
                "  split_val: validation",
                "train:",
                "  steps: 10",
                "  batch_size: 2",
                "  K: 4",
                "  lr: 0.0001",
                "pipeline:",
                "  pipe_stages: 1",
                "  weight_mode: uniform",
                "",
            ]
        )
    )

    runtime = resolve_benchmark_runtime(config_path=str(cfg_path))
    dataset = build_dataset(runtime=runtime, train=True)
    x, y = dataset[0]

    assert runtime.name == "celebahq256_latent"
    assert runtime.image_shape == (4, 32, 32)
    assert tuple(x.shape) == (4, 32, 32)
    assert x.dtype == torch.float32
    assert int(y) == 0


def test_celebahq256_latent_energy_model_accepts_four_channel_latents():
    runtime = resolve_benchmark_runtime(PROJECT_DIR / "configs" / "celebahq256_latent_ddp_strict.yaml")
    model = build_energy_model(runtime=runtime, n_f=8, unconditional_cls=EnergyModel)
    first_conv = model.net[0]

    assert first_conv.in_channels == 4
    out = model(torch.randn(2, 4, 32, 32))
    assert tuple(out.shape) == (2,)
