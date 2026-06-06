import torch
from torch import nn

from polaris_ebm.scripts.current.ebm_train_sync_mode_a import langevin_sample


class TinyEnergy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(4, 1, bias=False))
        with torch.no_grad():
            self.net[1].weight.fill_(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(x.size(0))


def test_langevin_works_under_outer_no_grad():
    model = TinyEnergy()
    chain = torch.zeros((2, 1, 2, 2), dtype=torch.float32)

    with torch.no_grad():
        out = langevin_sample(
            model=model,
            chain=chain,
            k_steps=1,
            langevin_sign=1.0,
            step_size=0.1,
            noise_std=0.0,
            clamp_x=False,
        )

    assert out.shape == chain.shape
    assert torch.any(out != 0.0)
