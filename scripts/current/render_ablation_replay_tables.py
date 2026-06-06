#!/usr/bin/env python3
"""Render replay-based ablation tables using final replay FID values."""

import argparse
import csv
from pathlib import Path
from typing import Dict, List


GAP_THRESHOLD = 5.0


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def fmt_num(text) -> str:
    if text in ("", None, "None"):
        return ""
    return f"{float(text):.3f}"


def render_table(rows: List[Dict[str, str]]) -> List[str]:
    lines = [
        "| exp_id | train_mode | seed | steps | final FID | feature-FID | unique | status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {exp_id} | {train_mode} | {seed} | {steps} | {final_fid} | {fid_feature} | {unique_ratio} | {status} |".format(
                exp_id=row["exp_id"],
                train_mode=row.get("train_mode", ""),
                seed=row.get("seed", ""),
                steps=row.get("steps", ""),
                final_fid=fmt_num(row.get("final_fid", "")),
                fid_feature=fmt_num(row.get("fid_feature", "")),
                unique_ratio=fmt_num(row.get("unique_ratio", "")),
                status=row.get("status", ""),
            )
        )
    if not rows:
        lines.append("| _none_ |  |  |  |  |  |  |  |")
    return lines


def render_notes(rows: List[Dict[str, str]]) -> List[str]:
    notes = []
    for row in rows:
        try:
            gap = abs(float(row.get("final_minus_best", "")))
        except Exception:
            gap = 0.0
        if gap >= GAP_THRESHOLD:
            notes.append(
                "- `{exp_id}` best checkpoint: step `{best_step}` -> `{best_fid}` vs final `{final_fid}`.".format(
                    exp_id=row["exp_id"],
                    best_step=row.get("best_step", ""),
                    best_fid=fmt_num(row.get("best_fid", "")),
                    final_fid=fmt_num(row.get("final_fid", "")),
                )
            )
    return notes


def main() -> None:
    ap = argparse.ArgumentParser("Render replay-based ablation tables")
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--out_md", required=True)
    args = ap.parse_args()

    rows = load_rows(Path(args.summary_csv))

    def by_exp(exp_ids: List[str]) -> List[Dict[str, str]]:
        wanted = set(exp_ids)
        return [row for row in rows if row["exp_id"] in wanted]

    sections = [
        (
            "Baseline Alignment",
            by_exp(
                [
                    "A0_single_strict_300k_ref",
                    "A0_ddp_strict_300k_ref",
                    "A0_pipeline_strict_300k_ref",
                    "A1_single_fullk_k100_s20k",
                    "A3_ddp_fullk_k100_s20k",
                    "A3_pipe_last2b001_p4_k100_s20k",
                ]
            ),
        ),
        ("Weighted-Sum", [row for row in rows if row.get("group") == "A1"]),
        ("Pipeline Factor", [row for row in rows if row.get("group") == "A2"]),
        ("K Sweep", [row for row in rows if row.get("group") == "A3"]),
        ("Step-Size", [row for row in rows if row.get("group") == "A4"]),
    ]

    out_lines = ["# Ablation Replay Tables", ""]
    for title, sec_rows in sections:
        out_lines.append(f"## {title}")
        out_lines.extend(render_table(sec_rows))
        notes = render_notes(sec_rows)
        if notes:
            out_lines.append("")
            out_lines.append("Footnotes:")
            out_lines.extend(notes)
        out_lines.append("")

    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines))


if __name__ == "__main__":
    main()
