import torch

from polaris_ebm.scripts.current.conditional_model import ConditionalEnergyModel
from polaris_ebm.scripts.current.conditional_sampler import langevin_sample_conditional


def test_conditional_energy_model_accepts_x_and_y():
    model = ConditionalEnergyModel(n_f=16, num_classes=10, image_size=32)
    x = torch.randn((2, 3, 32, 32))
    y = torch.tensor([1, 7], dtype=torch.long)

    out = model(x, y)
    assert out.shape == (2,)


class TinyConditionalEnergy(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        flat = x.view(x.size(0), -1).sum(dim=1)
        return flat + y.float()


def test_langevin_sample_conditional_keeps_labels_fixed():
    model = TinyConditionalEnergy()
    chain = torch.zeros((2, 1, 2, 2), dtype=torch.float32)
    y = torch.tensor([3, 5], dtype=torch.long)
    y_before = y.clone()

    out = langevin_sample_conditional(
        model=model,
        chain=chain,
        labels=y,
        k_steps=2,
        langevin_sign=1.0,
        step_size=0.1,
        noise_std=0.0,
        clamp_x=False,
    )

    assert out.shape == chain.shape
    assert torch.equal(y, y_before)
