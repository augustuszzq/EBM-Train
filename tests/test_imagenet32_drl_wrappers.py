from types import SimpleNamespace

import torch

from polaris_ebm.scripts.current import ebm_train_sync_mode_a as sync_base
from polaris_ebm.scripts.current.drl_resnet_energy import DRLResNetEnergy
from polaris_ebm.scripts.current.ebm_train_imagenet32_drl_single import (
    build_drl_energy_model as build_single_drl_energy_model,
)
from polaris_ebm.scripts.current.ebm_train_imagenet32_drl_sync import (
    build_drl_energy_model as build_sync_drl_energy_model,
)


def make_runtime():
    return SimpleNamespace(
        image_shape=(3, 32, 32),
        config=SimpleNamespace(
            raw={
                "model": {
                    "family": "drl_resnet_energy",
                    "ch": 8,
                    "ch_mult": [1, 2],
                    "num_res_blocks": 1,
                    "attention_resolutions": [],
                    "spectral_norm": False,
                }
            }
        ),
    )


def test_drl_sync_wrapper_builds_drl_model_without_labels():
    model = build_sync_drl_energy_model(runtime=make_runtime(), n_f=96, unconditional_cls=None)
    x = torch.randn((2, 3, 32, 32))
    y = torch.tensor([4, 5], dtype=torch.long)

    assert isinstance(model, DRLResNetEnergy)
    assert model(x, y).shape == (2,)


def test_drl_single_wrapper_builds_same_model_family():
    model = build_single_drl_energy_model(runtime=make_runtime(), n_f=96, unconditional_cls=None)

    assert isinstance(model, DRLResNetEnergy)


def test_importing_drl_wrapper_does_not_mutate_canonical_trainer_binding():
    before = sync_base.build_energy_model

    import polaris_ebm.scripts.current.ebm_train_imagenet32_drl_sync  # noqa: F401

    assert sync_base.build_energy_model is before
