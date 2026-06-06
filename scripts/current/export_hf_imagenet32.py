#!/usr/bin/env python3
"""Materialize HF ImageNet-32 into a restart-safe local folder layout."""

from __future__ import annotations

import argparse
import json

try:
    from imagenet32_data import materialize_imagenet32
except Exception:
    from polaris_ebm.scripts.current.imagenet32_data import materialize_imagenet32


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Export ChocolateDave/imagenet-32 to a local folder cache")
    ap.add_argument("--out-root", type=str, required=True)
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "valid", "validation"])
    ap.add_argument("--cache-dir", type=str, default="")
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--max-examples", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    payload = materialize_imagenet32(
        out_root=args.out_root,
        split=args.split,
        cache_dir=args.cache_dir,
        streaming=bool(args.streaming),
        max_examples=int(args.max_examples),
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
