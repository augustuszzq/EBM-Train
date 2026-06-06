import torch

from polaris_ebm.scripts.current.eval_generate import summarize_tensor


def test_summarize_tensor_binary_values():
    x = torch.tensor([[0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    s = summarize_tensor(x, unique_limit=100)
    assert s["shape"] == [2, 2]
    assert s["dtype"] == "torch.float32"
    assert s["unique_count"] == 2
    assert s["unique_mode"] == "exact"
    assert abs(s["frac_zero"] - 0.25) < 1e-9
    assert abs(s["frac_one"] - 0.75) < 1e-9


def test_summarize_tensor_uses_sampled_mode_when_large():
    x = torch.linspace(0.0, 1.0, steps=2000, dtype=torch.float32).view(1000, 2)
    s = summarize_tensor(x, unique_limit=100)
    assert s["numel"] == 2000
    assert s["unique_mode"] == "sampled"
    assert s["unique_count"] > 1
