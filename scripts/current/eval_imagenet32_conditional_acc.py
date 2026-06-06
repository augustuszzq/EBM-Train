#!/usr/bin/env python3
"""Classifier-based conditional-faithfulness metric for ImageNet-32 samples."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch as t
import torch.nn.functional as F

try:
    from torchvision import models
    from torchvision.models import ResNet18_Weights, ResNet50_Weights
except Exception:
    models = None
    ResNet18_Weights = None
    ResNet50_Weights = None

try:
    from eval_metrics import _first_tensor_from_obj
except Exception:
    from polaris_ebm.scripts.current.eval_metrics import _first_tensor_from_obj


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Evaluate conditional accuracy on generated ImageNet-32 samples")
    ap.add_argument("--samples", type=str, required=True)
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--classifier_arch", type=str, default="resnet50", choices=["resnet18", "resnet50"])
    ap.add_argument("--classifier_checkpoint", type=str, default="")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--strict_pretrained", action="store_true")
    return ap.parse_args()


def select_device(mode: str) -> t.device:
    if mode == "cpu":
        return t.device("cpu")
    if mode == "cuda":
        if not t.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return t.device("cuda")
    return t.device("cuda" if t.cuda.is_available() else "cpu")


def load_samples_and_labels(path: str) -> Tuple[t.Tensor, t.Tensor]:
    obj = t.load(path, map_location="cpu")
    samples = _first_tensor_from_obj(obj).float().clamp(0.0, 1.0)
    labels = obj.get("labels") if isinstance(obj, dict) else None
    if labels is None or not t.is_tensor(labels):
        raise RuntimeError("samples payload is missing 'labels'; conditional accuracy requires requested labels")
    labels = labels.long().cpu()
    if samples.size(0) != labels.numel():
        raise RuntimeError(
            "samples/labels mismatch: num_samples=%d num_labels=%d" % (samples.size(0), labels.numel())
        )
    return samples, labels


def _build_resnet(arch: str, num_classes: int, checkpoint: str):
    if models is None:
        raise RuntimeError("torchvision.models is required for conditional accuracy")
    if arch == "resnet18":
        if checkpoint:
            model = models.resnet18(weights=None)
        else:
            weights = ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights)
    else:
        if checkpoint:
            model = models.resnet50(weights=None)
        else:
            weights = getattr(ResNet50_Weights, "IMAGENET1K_V2", ResNet50_Weights.IMAGENET1K_V1)
            model = models.resnet50(weights=weights)
    if checkpoint:
        if model.fc.out_features != num_classes:
            model.fc = t.nn.Linear(model.fc.in_features, num_classes)
        obj = t.load(checkpoint, map_location="cpu")
        state = obj.get("state_dict", obj)
        model.load_state_dict(state, strict=True)
    return model


def build_classifier(
    arch: str,
    *,
    checkpoint: str,
    num_classes: int,
    strict_pretrained: bool,
) -> t.nn.Module:
    try:
        return _build_resnet(arch=arch, num_classes=num_classes, checkpoint=checkpoint)
    except Exception:
        if strict_pretrained or checkpoint:
            raise
        raise RuntimeError(
            "failed to load torchvision pretrained classifier; pass --classifier_checkpoint or retry with network access"
        )


def preprocess_for_classifier(samples: t.Tensor, resize: int) -> t.Tensor:
    if resize <= 0:
        raise RuntimeError("resize must be > 0")
    x = samples.float().clamp(0.0, 1.0)
    if x.size(-1) != resize or x.size(-2) != resize:
        x = F.interpolate(x, size=(resize, resize), mode="bilinear", align_corners=False, antialias=True)
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def summarize_predictions(labels: t.Tensor, preds: t.Tensor) -> Dict[str, object]:
    labels = labels.long().cpu()
    preds = preds.long().cpu()
    if labels.numel() != preds.numel():
        raise RuntimeError("labels/preds length mismatch")
    total = int(labels.numel())
    if total <= 0:
        raise RuntimeError("cannot summarize empty predictions")

    num_correct = int((labels == preds).sum().item())
    label_counts: Dict[int, int] = defaultdict(int)
    label_correct: Dict[int, int] = defaultdict(int)
    confusion: Dict[Tuple[int, int], int] = defaultdict(int)
    for y, p in zip(labels.tolist(), preds.tolist()):
        label_counts[int(y)] += 1
        if int(y) == int(p):
            label_correct[int(y)] += 1
        else:
            confusion[(int(y), int(p))] += 1

    per_class = []
    for label in sorted(label_counts):
        count = int(label_counts[label])
        acc = float(label_correct[label] / count) if count > 0 else 0.0
        per_class.append({"label": int(label), "count": count, "top1_acc": acc})
    top_confusions = [
        {"requested_label": int(y), "predicted_label": int(p), "count": int(c)}
        for (y, p), c in sorted(confusion.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))[:20]
    ]
    return {
        "top1_conditional_acc": float(num_correct / total),
        "per_class_avg_acc": float(sum(row["top1_acc"] for row in per_class) / len(per_class)),
        "num_samples": total,
        "num_classes_present": int(len(per_class)),
        "class_min_count": int(min(row["count"] for row in per_class)),
        "class_max_count": int(max(row["count"] for row in per_class)),
        "per_class": per_class,
        "top_confusions": top_confusions,
    }


def compute_conditional_accuracy(
    model: t.nn.Module,
    samples: t.Tensor,
    labels: t.Tensor,
    *,
    batch: int,
    device: t.device,
    resize: int,
) -> Dict[str, object]:
    preds: List[t.Tensor] = []
    model = model.to(device)
    model.eval()
    with t.no_grad():
        for s in range(0, samples.size(0), batch):
            xb = preprocess_for_classifier(samples[s : s + batch], resize=resize).to(device, non_blocking=True)
            logits = model(xb)
            preds.append(logits.argmax(dim=1).cpu())
    pred = t.cat(preds, dim=0)
    return summarize_predictions(labels=labels, preds=pred)


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    samples, labels = load_samples_and_labels(args.samples)
    num_classes = int(labels.max().item()) + 1
    model = build_classifier(
        args.classifier_arch,
        checkpoint=args.classifier_checkpoint,
        num_classes=num_classes,
        strict_pretrained=bool(args.strict_pretrained),
    )
    metrics = compute_conditional_accuracy(
        model,
        samples,
        labels,
        batch=int(args.batch),
        device=device,
        resize=int(args.resize),
    )
    metrics["samples_path"] = args.samples
    metrics["classifier_arch"] = args.classifier_arch
    metrics["classifier_checkpoint"] = args.classifier_checkpoint
    with open(args.out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print("[DONE] wrote %s" % args.out_json, flush=True)


if __name__ == "__main__":
    main()
