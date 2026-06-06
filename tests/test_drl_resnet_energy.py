import torch

from polaris_ebm.scripts.current.drl_resnet_energy import (
    DRLResNetEnergy,
    SelfAttention2d,
    build_drl_resnet_energy_from_config,
)


def test_drl_resnet_energy_returns_per_image_scalar_and_ignores_labels():
    model = DRLResNetEnergy(
        image_size=32,
        ch=16,
        ch_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(),
        spectral_norm=False,
    )
    x = torch.randn((3, 3, 32, 32))
    y = torch.tensor([0, 7, 999], dtype=torch.long)

    energy_without_labels = model(x)
    energy_with_labels = model(x, y)

    assert energy_without_labels.shape == (3,)
    assert torch.allclose(energy_without_labels, energy_with_labels)


def test_drl_resnet_energy_can_enable_attention_at_16x16():
    model = DRLResNetEnergy(
        image_size=32,
        ch=8,
        ch_mult=(1, 2, 2),
        num_res_blocks=1,
        attention_resolutions=(16,),
        spectral_norm=False,
    )
    x = torch.randn((2, 3, 32, 32))

    out = model(x)

    assert out.shape == (2,)
    assert any(isinstance(module, SelfAttention2d) for module in model.modules())


def test_drl_resnet_energy_applies_spectral_norm_to_convs_when_requested():
    model = DRLResNetEnergy(
        image_size=32,
        ch=8,
        ch_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(),
        spectral_norm=True,
    )

    assert any(
        isinstance(module, torch.nn.Conv2d) and hasattr(module, "weight_orig")
        for module in model.modules()
    )


def test_build_drl_resnet_energy_from_config_uses_model_section_defaults():
    model = build_drl_resnet_energy_from_config(
        {
            "family": "drl_resnet_energy",
            "ch": 8,
            "ch_mult": [1, 2],
            "num_res_blocks": 1,
            "attention_resolutions": [16],
            "spectral_norm": False,
        },
        image_size=32,
    )

    x = torch.randn((1, 3, 32, 32))
    assert model(x).shape == (1,)


def test_drl_energy_sign_and_scale_wrap_raw_energy():
    model = DRLResNetEnergy(
        image_size=32,
        ch=8,
        ch_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(),
        spectral_norm=False,
        energy_sign=-1,
        energy_scale=2.5,
    )
    x = torch.randn((4, 3, 32, 32))

    raw = model.raw_energy(x)
    energy = model(x)

    assert torch.allclose(energy, -2.5 * raw)


def test_new_linear_energy_head_zero_init_starts_at_zero_energy():
    model = DRLResNetEnergy(
        image_size=32,
        ch=8,
        ch_mult=(1, 2),
        num_res_blocks=1,
        attention_resolutions=(),
        spectral_norm=True,
        use_new_energy_head=True,
        zero_init_energy_head=True,
    )
    x = torch.randn((3, 3, 32, 32))

    out = model(x)

    assert isinstance(model.head, torch.nn.Linear)
    assert not hasattr(model.head, "weight_orig")
    assert torch.allclose(model.head.weight, torch.zeros_like(model.head.weight))
    assert torch.allclose(model.head.bias, torch.zeros_like(model.head.bias))
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_build_drl_resnet_energy_from_config_reads_port_sanity_knobs():
    model = build_drl_resnet_energy_from_config(
        {
            "family": "drl_resnet_energy",
            "ch": 8,
            "ch_mult": [1, 2],
            "num_res_blocks": 1,
            "spectral_norm": False,
            "energy_sign": -1,
            "energy_scale": 0.25,
            "use_new_energy_head": True,
            "zero_init_energy_head": True,
        },
        image_size=32,
    )

    assert model.energy_sign == -1.0
    assert model.energy_scale == 0.25
    assert model.use_new_energy_head is True
    assert model.zero_init_energy_head is True
