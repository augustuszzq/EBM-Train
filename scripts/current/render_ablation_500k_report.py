#!/usr/bin/env python3
"""Render a compact 500k summary markdown table."""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List

try:
    from ablation_common import pretty_walltime
except Exception:
    from polaris_ebm.scripts.current.ablation_common import pretty_walltime


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def fmt_num(text: str, digits: int = 3) -> str:
    if text in ("", "None", "nan", "NaN", None):
        return ""
    return f"{float(text):.{digits}f}"


def infer_seed(row: Dict[str, str]) -> str:
    seed = row.get("seed", "")
    if seed:
        return seed
    match = re.search(r"_seed(\d+)_", row.get("exp_id", ""))
    return match.group(1) if match else ""


def main() -> None:
    ap = argparse.ArgumentParser("Render 500k summary markdown report")
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    rows = load_rows(Path(args.summary_csv))
    lines = [
        "# 500k Summary",
        "",
        "| exp_id | train_mode | seed | K | FID | feature-FID | unique | walltime | status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {exp_id} | {train_mode} | {seed} | {K} | {fid} | {fidf} | {uniq} | {wall} | {status} |".format(
                exp_id=row["exp_id"],
                train_mode=row["train_mode"],
                seed=infer_seed(row),
                K=row["K"],
                fid=fmt_num(row.get("fid_inception", "")),
                fidf=fmt_num(row.get("fid_feature", "")),
                uniq=fmt_num(row.get("unique_ratio", "")),
                wall=pretty_walltime(row.get("walltime", "")),
                status=row.get("status", ""),
            )
        )
    Path(args.out_md).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
