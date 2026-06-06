#!/usr/bin/env python3
"""Compare baseline vs pipeline sample quality with unified metrics."""
import argparse
import hashlib
import json
import os
from typing import Dict, Optional, Tuple

import torch as t
import torch.nn.functional as F

try:
    from torchvision import datasets, models, transforms
    from torchvision.models import Inception_V3_Weights
except Exception:
    datasets = None
    models = None
    transforms = None
    Inception_V3_Weights = None

try:
    from ebm_train_sync_mode_a import EnergyModel
except Exception:
    from polaris_ebm.scripts.current.ebm_train_sync_mode_a import EnergyModel

try:
    from benchmark_runtime import build_dataset, build_energy_model, energy_call, resolve_benchmark_runtime
except Exception:
    from polaris_ebm.scripts.current.benchmark_runtime import (
        build_dataset,
        build_energy_model,
        energy_call,
        resolve_benchmark_runtime,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Compare baseline/pipeline samples with unified metrics")
    ap.add_argument("--baseline_samples", type=str, required=True)
    ap.add_argument("--pipeline_samples", type=str, required=True)
    ap.add_argument("--config", type=str, default="")
    ap.add_argument("--real_samples", type=str, default="")
    ap.add_argument("--data_dir", type=str, default="")
    ap.add_argument("--num_real", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--feature_pool", type=int, default=8)
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--baseline_ckpt", type=str, default="")
    ap.add_argument("--pipeline_ckpt", type=str, default="")
    ap.add_argument("--n_f", type=int, default=-1)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument(
        "--skip_standard_fid",
        action="store_true",
        help="skip standard InceptionV3 FID computation (keeps fid_feature_* only)",
    )
    ap.add_argument(
        "--standard_fid_strict",
        action="store_true",
        help="fail if standard InceptionV3 FID cannot be computed",
    )
    ap.add_argument("--standard_fid_resize", type=int, default=299)
    return ap.parse_args()


def select_device(mode: str) -> t.device:
    if mode == "cpu":
        return t.device("cpu")
    if mode == "cuda":
        if not t.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return t.device("cuda")
    return t.device("cuda" if t.cuda.is_available() else "cpu")


def load_checkpoint(path: str) -> Tuple[Dict, Dict]:
    obj = t.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"], obj
    return obj, {}


def infer_n_f(meta: Dict, override_n_f: int) -> int:
    if int(override_n_f) > 0:
        return int(override_n_f)
    args_meta = meta.get("args", {})
    if isinstance(args_meta, dict) and "n_f" in args_meta:
        return int(args_meta["n_f"])
    return 64


def _first_tensor_from_obj(obj):
    if t.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        if "samples_01" in obj and t.is_tensor(obj["samples_01"]):
            return obj["samples_01"]
        if "samples_m11" in obj and t.is_tensor(obj["samples_m11"]):
            return ((obj["samples_m11"].float().clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)
        for v in obj.values():
            if t.is_tensor(v):
                return v
    raise RuntimeError("could not find tensor payload in input object")


def load_samples_tensor(path: str) -> t.Tensor:
    obj = t.load(path, map_location="cpu")
    x = _first_tensor_from_obj(obj).float()
    if x.dim() != 4 or x.size(1) != 3:
        raise RuntimeError(f"{path} must contain images shaped [N,3,H,W], got {tuple(x.shape)}")
    return x.clamp(0.0, 1.0)


def load_real_samples(args: argparse.Namespace) -> t.Tensor:
    if args.real_samples:
        return load_samples_tensor(args.real_samples)[: args.num_real]

    runtime = resolve_benchmark_runtime(config_path=str(getattr(args, "config", "")), data_dir=args.data_dir or "./data/cifar10")
    if runtime.name != "cifar10":
        ds = build_dataset(runtime=runtime, cifar_builder=None, train=False)
        imgs = []
        for i in range(min(args.num_real, len(ds))):
            img, _ = ds[i]
            imgs.append(((img.float().clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0))
        return t.stack(imgs, dim=0)

    if not args.data_dir:
        raise RuntimeError("either --real_samples or --data_dir must be provided")
    if datasets is None or transforms is None:
        raise RuntimeError("torchvision is required when using --data_dir")

    tfm = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    ds = datasets.CIFAR10(root=args.data_dir, train=True, download=False, transform=tfm)
    if len(ds) <= 0:
        raise RuntimeError("empty CIFAR10 dataset")

    imgs = []
    for i in range(min(args.num_real, len(ds))):
        img, _ = ds[i]
        imgs.append(img)
    return t.stack(imgs, dim=0).float().clamp(0.0, 1.0)


def to_m11(x01: t.Tensor) -> t.Tensor:
    return (x01.clamp(0.0, 1.0) * 2.0 - 1.0).clamp(-1.0, 1.0)


def extract_features(x01: t.Tensor, feature_pool: int) -> t.Tensor:
    if feature_pool <= 0:
        raise RuntimeError(f"feature_pool must be > 0, got {feature_pool}")
    pooled = F.adaptive_avg_pool2d(x01.float(), output_size=(feature_pool, feature_pool))
    return pooled.flatten(start_dim=1).double()


def gaussian_stats(features: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
    x = features.double()
    if x.dim() != 2:
        raise RuntimeError(f"features must be [N,D], got shape={tuple(x.shape)}")
    n = x.size(0)
    if n <= 0:
        raise RuntimeError("features must have positive batch size")
    mu = x.mean(dim=0)
    xc = x - mu
    if n == 1:
        cov = t.zeros((x.size(1), x.size(1)), dtype=x.dtype, device=x.device)
    else:
        cov = (xc.t() @ xc) / float(n - 1)
    return mu, cov


def frechet_distance(mu1: t.Tensor, cov1: t.Tensor, mu2: t.Tensor, cov2: t.Tensor) -> float:
    diff = (mu1 - mu2).double()
    cov_prod = cov1.double() @ cov2.double()
    eigvals = t.linalg.eigvals(cov_prod).real
    eigvals = t.clamp(eigvals, min=0.0)
    tr_covmean = t.sqrt(eigvals).sum()
    fid = diff.dot(diff) + t.trace(cov1) + t.trace(cov2) - 2.0 * tr_covmean
    fid_val = float(fid.item())
    if fid_val < 0.0 and abs(fid_val) < 1e-8:
        fid_val = 0.0
    return fid_val


def build_inception_v3_for_fid(device: t.device) -> t.nn.Module:
    if models is None or Inception_V3_Weights is None:
        raise RuntimeError("torchvision.models InceptionV3 is unavailable")
    weights = Inception_V3_Weights.IMAGENET1K_V1
    # torchvision>=0.21 requires aux_logits=True when loading pretrained weights.
    # In eval mode, forward still returns logits only, so this remains suitable for FID features.
    model = models.inception_v3(weights=weights, aux_logits=True, transform_input=False)
    model.fc = t.nn.Identity()
    model.eval()
    model.to(device)
    return model


def _preprocess_for_inception(x01: t.Tensor, out_size: int) -> t.Tensor:
    if out_size <= 0:
        raise RuntimeError(f"standard_fid_resize must be > 0, got {out_size}")
    x = x01.float().clamp(0.0, 1.0)
    if x.size(-1) != out_size or x.size(-2) != out_size:
        x = F.interpolate(x, size=(out_size, out_size), mode="bilinear", align_corners=False, antialias=True)
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


def extract_inception_features(
    model: t.nn.Module,
    x01: t.Tensor,
    batch: int,
    device: t.device,
    out_size: int,
) -> t.Tensor:
    if batch <= 0:
        raise RuntimeError(f"batch must be > 0, got {batch}")
    feats = []
    with t.no_grad():
        for s in range(0, x01.size(0), batch):
            xb = x01[s : s + batch].to(device, non_blocking=True)
            xb = _preprocess_for_inception(xb, out_size=out_size)
            fb = model(xb)
            if isinstance(fb, tuple):
                fb = fb[0]
            feats.append(fb.detach().cpu().double())
    return t.cat(feats, dim=0)


def diversity_trace(features: t.Tensor) -> float:
    _, cov = gaussian_stats(features)
    return float(t.trace(cov).item())


def saturation_stats(x01: t.Tensor, eps: float = 1e-6) -> Dict[str, float]:
    x = x01.float().clamp(0.0, 1.0)
    sat0 = float((x <= eps).float().mean().item())
    sat1 = float((x >= (1.0 - eps)).float().mean().item())
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "frac_zero": sat0,
        "frac_one": sat1,
    }


def distinct_image_stats(x01: t.Tensor) -> Dict[str, float]:
    x = (x01.float().clamp(0.0, 1.0) * 255.0 + 0.5).to(t.uint8).cpu().contiguous()
    n = int(x.size(0))
    if n <= 0:
        return {"unique_images": 0, "unique_ratio": 0.0}
    seen = set()
    for i in range(n):
        h = hashlib.sha1(x[i].numpy().tobytes()).digest()
        seen.add(h)
    return {"unique_images": int(len(seen)), "unique_ratio": float(len(seen) / n)}


def _batched_energy(model: t.nn.Module, x_m11: t.Tensor, batch: int, device: t.device) -> t.Tensor:
    vals = []
    with t.no_grad():
        for s in range(0, x_m11.size(0), batch):
            xb = x_m11[s : s + batch].to(device, non_blocking=True)
            eb = model(xb).view(-1).detach().cpu()
            vals.append(eb)
    return t.cat(vals, dim=0)


def energy_gap_metrics(
    ckpt_path: str,
    real_01: t.Tensor,
    gen_01: t.Tensor,
    batch: int,
    device: t.device,
    override_n_f: int,
) -> Dict[str, float]:
    state, meta = load_checkpoint(ckpt_path)
    n_f = infer_n_f(meta, override_n_f)

    runtime = resolve_benchmark_runtime(
        config_path=(meta.get("args", {}) or {}).get("config", ""),
        data_dir=(meta.get("args", {}) or {}).get("data_dir", "./data/cifar10"),
    )
    if runtime.conditional:
        return {
            "n_f": int(n_f),
            "conditional_energy_gap_skipped": True,
        }
    model = build_energy_model(runtime=runtime, n_f=n_f, unconditional_cls=EnergyModel).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint/model mismatch for %s: missing=%s unexpected=%s"
            % (ckpt_path, str(missing), str(unexpected))
        )
    model.eval()

    real_e = _batched_energy(model, to_m11(real_01), batch=batch, device=device)
    gen_e = _batched_energy(model, to_m11(gen_01), batch=batch, device=device)
    return {
        "n_f": int(n_f),
        "real_energy_mean": float(real_e.mean().item()),
        "real_energy_std": float(real_e.std(unbiased=False).item()),
        "gen_energy_mean": float(gen_e.mean().item()),
        "gen_energy_std": float(gen_e.std(unbiased=False).item()),
        "energy_gap_real_minus_gen": float((real_e.mean() - gen_e.mean()).item()),
    }


def main() -> None:
    args = parse_args()
    if args.num_real <= 0:
        raise RuntimeError(f"num_real must be > 0, got {args.num_real}")
    if args.batch <= 0:
        raise RuntimeError(f"batch must be > 0, got {args.batch}")

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)

    real_01 = load_real_samples(args)
    base_01 = load_samples_tensor(args.baseline_samples)
    pipe_01 = load_samples_tensor(args.pipeline_samples)

    n_real = min(args.num_real, real_01.size(0))
    n_base = min(n_real, base_01.size(0))
    n_pipe = min(n_real, pipe_01.size(0))
    n = min(n_real, n_base, n_pipe)
    if n <= 0:
        raise RuntimeError("no samples available for comparison")

    real_01 = real_01[:n]
    base_01 = base_01[:n]
    pipe_01 = pipe_01[:n]

    real_f = extract_features(real_01, feature_pool=args.feature_pool)
    base_f = extract_features(base_01, feature_pool=args.feature_pool)
    pipe_f = extract_features(pipe_01, feature_pool=args.feature_pool)

    mu_r, cov_r = gaussian_stats(real_f)
    mu_b, cov_b = gaussian_stats(base_f)
    mu_p, cov_p = gaussian_stats(pipe_f)

    fid_b = frechet_distance(mu_r, cov_r, mu_b, cov_b)
    fid_p = frechet_distance(mu_r, cov_r, mu_p, cov_p)

    metrics = {
        "num_real": int(n),
        "num_gen_baseline": int(n),
        "num_gen_pipeline": int(n),
        "feature_pool": int(args.feature_pool),
        "fid_feature_baseline_vs_real": float(fid_b),
        "fid_feature_pipeline_vs_real": float(fid_p),
        "fid_feature_pipeline_minus_baseline": float(fid_p - fid_b),
        "diversity_trace_baseline": float(diversity_trace(base_f)),
        "diversity_trace_pipeline": float(diversity_trace(pipe_f)),
        "real_stats": saturation_stats(real_01),
        "baseline_stats": saturation_stats(base_01),
        "pipeline_stats": saturation_stats(pipe_01),
        "real_distinct_images": distinct_image_stats(real_01),
        "baseline_distinct_images": distinct_image_stats(base_01),
        "pipeline_distinct_images": distinct_image_stats(pipe_01),
        "baseline_samples_path": args.baseline_samples,
        "pipeline_samples_path": args.pipeline_samples,
        "real_source": args.real_samples if args.real_samples else args.data_dir,
        "fid_inception_baseline_vs_real": None,
        "fid_inception_pipeline_vs_real": None,
        "fid_inception_pipeline_minus_baseline": None,
        "fid_inception_model": "torchvision_inception_v3_imagenet1k_v1",
        "fid_inception_error": "",
    }

    if args.skip_standard_fid:
        metrics["fid_inception_error"] = "skipped_by_flag"
    else:
        try:
            fid_device = select_device(args.device)
            inception_model = build_inception_v3_for_fid(device=fid_device)
            real_i = extract_inception_features(
                inception_model,
                real_01,
                batch=args.batch,
                device=fid_device,
                out_size=args.standard_fid_resize,
            )
            base_i = extract_inception_features(
                inception_model,
                base_01,
                batch=args.batch,
                device=fid_device,
                out_size=args.standard_fid_resize,
            )
            pipe_i = extract_inception_features(
                inception_model,
                pipe_01,
                batch=args.batch,
                device=fid_device,
                out_size=args.standard_fid_resize,
            )
            mu_ri, cov_ri = gaussian_stats(real_i)
            mu_bi, cov_bi = gaussian_stats(base_i)
            mu_pi, cov_pi = gaussian_stats(pipe_i)
            fid_bi = frechet_distance(mu_ri, cov_ri, mu_bi, cov_bi)
            fid_pi = frechet_distance(mu_ri, cov_ri, mu_pi, cov_pi)
            metrics["fid_inception_baseline_vs_real"] = float(fid_bi)
            metrics["fid_inception_pipeline_vs_real"] = float(fid_pi)
            metrics["fid_inception_pipeline_minus_baseline"] = float(fid_pi - fid_bi)
        except Exception as e:
            metrics["fid_inception_error"] = str(e)
            if args.standard_fid_strict:
                raise

    device = select_device(args.device)
    if args.baseline_ckpt:
        metrics["energy_baseline_model"] = energy_gap_metrics(
            ckpt_path=args.baseline_ckpt,
            real_01=real_01,
            gen_01=base_01,
            batch=args.batch,
            device=device,
            override_n_f=args.n_f,
        )
    if args.pipeline_ckpt:
        metrics["energy_pipeline_model"] = energy_gap_metrics(
            ckpt_path=args.pipeline_ckpt,
            real_01=real_01,
            gen_01=pipe_01,
            batch=args.batch,
            device=device,
            override_n_f=args.n_f,
        )

    with open(args.out_json, "w") as f:
        json.dump(metrics, f, indent=2)

    print("[METRICS] wrote %s" % args.out_json, flush=True)
    print(
        "[METRICS] feature-FID baseline=%.6f pipeline=%.6f delta=%.6f"
        % (fid_b, fid_p, fid_p - fid_b),
        flush=True,
    )
    if metrics["fid_inception_baseline_vs_real"] is not None:
        print(
            "[METRICS] inception-FID baseline=%.6f pipeline=%.6f delta=%.6f"
            % (
                metrics["fid_inception_baseline_vs_real"],
                metrics["fid_inception_pipeline_vs_real"],
                metrics["fid_inception_pipeline_minus_baseline"],
            ),
            flush=True,
        )
    elif metrics["fid_inception_error"]:
        print("[METRICS] inception-FID unavailable: %s" % metrics["fid_inception_error"], flush=True)


if __name__ == "__main__":
    main()
