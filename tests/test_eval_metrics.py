import json
import sys
from pathlib import Path

import torch

from polaris_ebm.scripts.current import eval_metrics


def test_frechet_distance_zero_for_identical_distributions():
    x = torch.randn(128, 16, dtype=torch.float64)
    mu, cov = eval_metrics.gaussian_stats(x)
    fid = eval_metrics.frechet_distance(mu, cov, mu, cov)
    assert abs(fid) < 1e-9


def test_load_samples_tensor_from_saved_dict(tmp_path: Path):
    p = tmp_path / "samples.pt"
    data = torch.rand(8, 3, 32, 32)
    torch.save({"samples_01": data}, p)
    loaded = eval_metrics.load_samples_tensor(str(p))
    assert loaded.shape == (8, 3, 32, 32)
    assert loaded.dtype == torch.float32
    assert torch.allclose(loaded, data.float())


def test_eval_metrics_main_with_real_samples_file(tmp_path: Path, monkeypatch):
    real = torch.rand(64, 3, 32, 32)
    baseline = torch.clamp(real + 0.05 * torch.randn_like(real), 0.0, 1.0)
    pipeline = torch.clamp(real + 0.15 * torch.randn_like(real), 0.0, 1.0)

    real_pt = tmp_path / "real.pt"
    base_pt = tmp_path / "base.pt"
    pipe_pt = tmp_path / "pipe.pt"
    out_json = tmp_path / "metrics.json"

    torch.save({"samples_01": real}, real_pt)
    torch.save({"samples_01": baseline}, base_pt)
    torch.save({"samples_01": pipeline}, pipe_pt)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_metrics.py",
            "--baseline_samples",
            str(base_pt),
            "--pipeline_samples",
            str(pipe_pt),
            "--real_samples",
            str(real_pt),
            "--num_real",
            "64",
            "--batch",
            "16",
            "--feature_pool",
            "8",
            "--skip_standard_fid",
            "--out_json",
            str(out_json),
        ],
    )
    eval_metrics.main()

    metrics = json.loads(out_json.read_text())
    assert "fid_feature_baseline_vs_real" in metrics
    assert "fid_feature_pipeline_vs_real" in metrics
    assert "fid_inception_baseline_vs_real" in metrics
    assert "fid_inception_pipeline_vs_real" in metrics
    assert metrics["fid_inception_baseline_vs_real"] is None
    assert metrics["fid_inception_pipeline_vs_real"] is None
    assert "baseline_distinct_images" in metrics
    assert "pipeline_distinct_images" in metrics
    assert metrics["num_real"] == 64
    assert metrics["num_gen_baseline"] == 64
    assert metrics["num_gen_pipeline"] == 64
