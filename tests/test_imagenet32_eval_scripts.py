import json
import sys
from pathlib import Path

import torch

from polaris_ebm.scripts.current import eval_imagenet32_conditional_acc, eval_imagenet32_fid


def test_compute_single_sample_metrics_zero_feature_fid_for_identical_inputs():
    samples = torch.rand(8, 3, 32, 32)
    metrics = eval_imagenet32_fid.compute_single_sample_metrics(
        samples,
        samples.clone(),
        batch=4,
        feature_pool=4,
        device=torch.device("cpu"),
        skip_standard_fid=True,
        standard_fid_strict=False,
        standard_fid_resize=299,
    )
    assert abs(metrics["fid_feature"]) < 1e-9
    assert metrics["num_real"] == 8
    assert metrics["num_gen"] == 8


def test_summarize_predictions_reports_accuracy_and_confusions():
    labels = torch.tensor([0, 0, 1, 1])
    preds = torch.tensor([0, 1, 1, 1])
    metrics = eval_imagenet32_conditional_acc.summarize_predictions(labels, preds)
    assert metrics["top1_conditional_acc"] == 0.75
    assert metrics["per_class_avg_acc"] == 0.75
    assert metrics["num_classes_present"] == 2
    assert metrics["top_confusions"][0]["requested_label"] == 0
    assert metrics["top_confusions"][0]["predicted_label"] == 1


def test_eval_imagenet32_conditional_acc_main_with_checkpoint_override(tmp_path: Path, monkeypatch):
    samples = torch.rand(4, 3, 32, 32)
    labels = torch.tensor([0, 1, 0, 1])
    sample_path = tmp_path / "samples.pt"
    out_json = tmp_path / "conditional_acc.json"
    torch.save({"samples_01": samples, "labels": labels}, sample_path)

    class DummyClassifier(torch.nn.Module):
        def forward(self, xb):
            logits = torch.zeros((xb.size(0), 2), dtype=xb.dtype, device=xb.device)
            logits[:, 1] = 1.0
            return logits

    monkeypatch.setattr(
        eval_imagenet32_conditional_acc,
        "build_classifier",
        lambda *args, **kwargs: DummyClassifier(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_imagenet32_conditional_acc.py",
            "--samples",
            str(sample_path),
            "--out_json",
            str(out_json),
            "--device",
            "cpu",
        ],
    )
    eval_imagenet32_conditional_acc.main()
    payload = json.loads(out_json.read_text())
    assert payload["top1_conditional_acc"] == 0.5
    assert payload["num_samples"] == 4

