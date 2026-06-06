#!/usr/bin/env python3
"""Render Phase 2 300k summary tables as Markdown."""

import argparse
import csv
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


def render_table(rows: List[Dict[str, str]]) -> List[str]:
    lines = [
        "| exp_id | train_mode | ws | P | K | eps | weight | FID | feature-FID | unique | walltime | status |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {exp_id} | {train_mode} | {world_size} | {pipe_stages} | {K} | {step_size} | {weight_mode} | {fid} | {fidf} | {uniq} | {wall} | {status} |".format(
                exp_id=row["exp_id"],
                train_mode=row["train_mode"],
                world_size=row["world_size"],
                pipe_stages=row["pipe_stages"],
                K=row["K"],
                step_size=row["step_size"],
                weight_mode=row["weight_mode"],
                fid=fmt_num(row.get("fid_inception", "")),
                fidf=fmt_num(row.get("fid_feature", "")),
                uniq=fmt_num(row.get("unique_ratio", "")),
                wall=pretty_walltime(row.get("walltime", "")),
                status=row.get("status", ""),
            )
        )
    if len(rows) == 0:
        lines.append("| _none_ |  |  |  |  |  |  |  |  |  |  |  |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser("Render Phase 2 ablation markdown report")
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    rows = load_rows(Path(args.summary_csv))

    sections = [
        ("Anchors", [row for row in rows if row["group"] == "P2A0"]),
        ("Reproducibility Seed Sweep", [row for row in rows if row["group"] == "M1"]),
        ("High-K Shortcut Control", [row for row in rows if row["group"] == "M2"]),
        ("Single-GPU Mechanism", [row for row in rows if row["group"] == "S1"]),
        ("Pipeline Stage-Count Extension", [row for row in rows if row["group"] == "S2"]),
    ]

    out_lines = ["# Phase 2 300k Summary", ""]
    for title, sec_rows in sections:
        out_lines.append(f"## {title}")
        out_lines.extend(render_table(sec_rows))
        out_lines.append("")

    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines))


if __name__ == "__main__":
    main()
