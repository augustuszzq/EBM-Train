#!/usr/bin/env python3
"""Evaluate a single generated sample set against ImageNet-32 real samples."""

from __future__ import annotations

import argparse
import json
from typing import Dict

import torch as t

try:
    from eval_metrics import (
        build_inception_v3_for_fid,
        distinct_image_stats,
        diversity_trace,
        extract_features,
        extract_inception_features,
        frechet_distance,
        gaussian_stats,
        load_real_samples,
        load_samples_tensor,
        saturation_stats,
        select_device,
    )
except Exception:
    from polaris_ebm.scripts.current.eval_metrics import (
        build_inception_v3_for_fid,
        distinct_image_stats,
        diversity_trace,
        extract_features,
        extract_inception_features,
        frechet_distance,
        gaussian_stats,
        load_real_samples,
        load_samples_tensor,
        saturation_stats,
        select_device,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Evaluate ImageNet-32 FID for one sample file")
    ap.add_argument("--samples", type=str, required=True)
    ap.add_argument("--config", type=str, default="")
    ap.add_argument("--real_samples", type=str, default="")
    ap.add_argument("--data_dir", type=str, default="")
    ap.add_argument("--num_real", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--feature_pool", type=int, default=8)
    ap.add_argument("--out_json", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--skip_standard_fid", action="store_true")
    ap.add_argument("--standard_fid_strict", action="store_true")
    ap.add_argument("--standard_fid_resize", type=int, default=299)
    return ap.parse_args()


def compute_single_sample_metrics(
    samples: t.Tensor,
    real: t.Tensor,
    *,
    batch: int,
    feature_pool: int,
    device: t.device,
    skip_standard_fid: bool,
    standard_fid_strict: bool,
    standard_fid_resize: int,
) -> Dict[str, object]:
    if samples.size(0) <= 0:
        raise RuntimeError("samples must contain at least one image")
    if real.size(0) <= 0:
        raise RuntimeError("real samples must contain at least one image")

    gen_feats = extract_features(samples, feature_pool=feature_pool)
    real_feats = extract_features(real, feature_pool=feature_pool)
    mu_g, cov_g = gaussian_stats(gen_feats)
    mu_r, cov_r = gaussian_stats(real_feats)
    fid_feature = frechet_distance(mu_g, cov_g, mu_r, cov_r)

    fid_standard = None
    standard_fid_error = None
    if not skip_standard_fid:
        try:
            inception = build_inception_v3_for_fid(device=device)
            gen_inc = extract_inception_features(
                inception,
                samples,
                batch=batch,
                device=device,
                out_size=standard_fid_resize,
            )
            real_inc = extract_inception_features(
                inception,
                real,
                batch=batch,
                device=device,
                out_size=standard_fid_resize,
            )
            mu_gi, cov_gi = gaussian_stats(gen_inc)
            mu_ri, cov_ri = gaussian_stats(real_inc)
            fid_standard = frechet_distance(mu_gi, cov_gi, mu_ri, cov_ri)
        except Exception as exc:
            standard_fid_error = str(exc)
            if standard_fid_strict:
                raise

    return {
        "final_fid": float(fid_standard if fid_standard is not None else fid_feature),
        "fid_inception": (None if fid_standard is None else float(fid_standard)),
        "fid_feature": float(fid_feature),
        "standard_fid_error": standard_fid_error,
        "num_real": int(real.size(0)),
        "num_gen": int(samples.size(0)),
        "distinct_images": int(distinct_image_stats(samples)["unique_images"]),
        "unique_ratio": float(distinct_image_stats(samples)["unique_ratio"]),
        "diversity_trace": float(diversity_trace(gen_feats)),
        "sample_stats": saturation_stats(samples),
        "real_stats": saturation_stats(real),
    }


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    samples = load_samples_tensor(args.samples)
    real = load_real_samples(args)
    metrics = compute_single_sample_metrics(
        samples,
        real,
        batch=int(args.batch),
        feature_pool=int(args.feature_pool),
        device=device,
        skip_standard_fid=bool(args.skip_standard_fid),
        standard_fid_strict=bool(args.standard_fid_strict),
        standard_fid_resize=int(args.standard_fid_resize),
    )
    metrics["samples_path"] = args.samples
    metrics["config"] = args.config
    with open(args.out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print("[DONE] wrote %s" % args.out_json, flush=True)


if __name__ == "__main__":
    main()
