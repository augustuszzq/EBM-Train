#!/usr/bin/env python3
"""Build paper-facing current-CIFAR tables and lightweight SVG figures.

This script intentionally uses only the Python standard library so collaborators
can regenerate the assets without pandas, matplotlib, or cluster-specific
packages. The canonical inputs are the ready CSVs under results/current_cifar.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


LABELS = {
    "single_P1_terminal": "Single terminal",
    "single_P2_equal": "Single P2 equal",
    "single_P2_deepest": "Single P2 deepest",
    "ddp_P1_terminal": "DDP terminal",
    "pipe_P4_equal": "Pipeline P4 equal",
    "pipe_P4_deepest": "Pipeline P4 deepest",
    "ddp_P1_terminal_K25": "DDP K=25",
    "ddp_P1_terminal_K50": "DDP K=50",
    "ddp_P1_terminal_K100": "DDP K=100",
    "pipe_P4_equal_K100": "Pipeline P4 equal K=100",
    "pipe_P4_deepest_K100": "Pipeline P4 deepest K=100",
}

COLORS = {
    "single_P1_terminal": "#7b7b7b",
    "single_P2_equal": "#2a9d8f",
    "single_P2_deepest": "#8ab17d",
    "ddp_P1_terminal": "#e76f51",
    "pipe_P4_equal": "#264653",
    "pipe_P4_deepest": "#f4a261",
    "ddp_P1_terminal_K25": "#b5179e",
    "ddp_P1_terminal_K50": "#7209b7",
    "ddp_P1_terminal_K100": "#e76f51",
    "pipe_P4_equal_K100": "#264653",
    "pipe_P4_deepest_K100": "#f4a261",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", "NA", "nan"} else math.nan


def mean(values: Iterable[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return statistics.mean(vals) if vals else math.nan


def std(values: Iterable[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return statistics.stdev(vals) if len(vals) > 1 else math.nan


def median(values: Iterable[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return statistics.median(vals) if vals else math.nan


def fmt(value: float, digits: int = 6) -> str:
    if value is None or math.isnan(value):
        return "NA"
    return f"{value:.{digits}f}"


def family_summary(
    runs: list[dict[str, str]],
    *,
    phase_group: str,
    horizon_steps: int,
    families: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family in families:
        group = [
            r
            for r in runs
            if r["phase_group"] == phase_group
            and r["family_id"] == family
            and int(r["horizon_steps"]) == horizon_steps
        ]
        if not group:
            continue
        best = [as_float(r, "best_fid_inception") for r in group]
        final = [as_float(r, "final_fid_inception") for r in group]
        rows.append(
            {
                "phase_group": phase_group,
                "family_id": family,
                "label": LABELS.get(family, family),
                "horizon_steps": horizon_steps,
                "runs": len(group),
                "seeds": ",".join(sorted({r["seed"] for r in group}, key=lambda x: int(float(x)))),
                "best_fid_mean": fmt(mean(best), 3),
                "best_fid_std": fmt(std(best), 3),
                "best_fid_median": fmt(median(best), 3),
                "final_fid_mean": fmt(mean(final), 3),
                "final_fid_std": fmt(std(final), 3),
                "final_fid_median": fmt(median(final), 3),
            }
        )
    return rows


def trajectory_coverage(
    runs: list[dict[str, str]], points: list[dict[str, str]]
) -> list[dict[str, object]]:
    by_run: dict[str, list[int]] = defaultdict(list)
    for point in points:
        by_run[point["logical_run_id"]].append(int(float(point["step"])))

    rows: list[dict[str, object]] = []
    for run in sorted(runs, key=lambda r: r["logical_run_id"]):
        steps = sorted(by_run.get(run["logical_run_id"], []))
        horizon = int(run["horizon_steps"])
        expected = horizon // 5000
        complete = (
            len(steps) == expected
            and bool(steps)
            and steps[0] == 5000
            and steps[-1] == horizon
            and all((b - a) == 5000 for a, b in zip(steps, steps[1:]))
        )
        rows.append(
            {
                "logical_run_id": run["logical_run_id"],
                "phase_group": run["phase_group"],
                "family_id": run["family_id"],
                "horizon_steps": horizon,
                "seed": run["seed"],
                "points": len(steps),
                "expected_points": expected,
                "first_step": steps[0] if steps else "",
                "last_step": steps[-1] if steps else "",
                "complete_5k_grid": "yes" if complete else "no",
            }
        )
    return rows


def html_escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return (dst_min + dst_max) / 2
    return dst_min + (value - src_min) * (dst_max - dst_min) / (src_max - src_min)


def svg_line_plot(
    path: Path,
    *,
    title: str,
    subtitle: str,
    points: list[dict[str, str]],
    runs: list[dict[str, str]],
    phase_group: str,
    horizon_steps: int,
    families: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    run_meta = {
        r["logical_run_id"]: r
        for r in runs
        if r["phase_group"] == phase_group
        and int(r["horizon_steps"]) == horizon_steps
        and r["family_id"] in families
    }
    grouped: dict[str, dict[str, list[tuple[int, float]]]] = {
        family: defaultdict(list) for family in families
    }
    for point in points:
        meta = run_meta.get(point["logical_run_id"])
        if not meta:
            continue
        grouped[meta["family_id"]][point["logical_run_id"]].append(
            (int(float(point["step"])), float(point["fid_inception"]))
        )

    all_values = [
        fid for family in families for pts_by_run in grouped[family].values() for _, fid in pts_by_run
    ]
    if not all_values:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>\n")
        return
    y_min = max(0.0, min(all_values) - 5.0)
    y_max = max(all_values) + 5.0
    width, height = 1100, 650
    left, right, top, bottom = 90, 240, 80, 90
    plot_w = width - left - right
    plot_h = height - top - bottom

    def sx(step: int) -> float:
        return scale(step, 0, horizon_steps, left, left + plot_w)

    def sy(fid: float) -> float:
        return scale(fid, y_min, y_max, top + plot_h, top)

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='#fbfaf7'/>",
        f"<text x='{left}' y='34' font-family='Arial, sans-serif' font-size='22' font-weight='700' fill='#1f2933'>{html_escape(title)}</text>",
        f"<text x='{left}' y='58' font-family='Arial, sans-serif' font-size='13' fill='#52616b'>{html_escape(subtitle)}</text>",
        f"<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' stroke='#39434d' stroke-width='1.2'/>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' stroke='#39434d' stroke-width='1.2'/>",
    ]
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        y = top + frac * plot_h
        value = y_max - frac * (y_max - y_min)
        lines.append(f"<line x1='{left}' y1='{y:.1f}' x2='{left + plot_w}' y2='{y:.1f}' stroke='#e2e8ee'/>")
        lines.append(
            f"<text x='{left - 12}' y='{y + 4:.1f}' text-anchor='end' font-family='Arial, sans-serif' font-size='11' fill='#52616b'>{value:.0f}</text>"
        )
    for step in range(0, horizon_steps + 1, 100000):
        x = sx(step)
        lines.append(
            f"<text x='{x:.1f}' y='{top + plot_h + 28}' text-anchor='middle' font-family='Arial, sans-serif' font-size='11' fill='#52616b'>{step // 1000}k</text>"
        )
    lines.append(
        f"<text x='{left + plot_w / 2}' y='{height - 25}' text-anchor='middle' font-family='Arial, sans-serif' font-size='12' fill='#39434d'>training step</text>"
    )
    lines.append(
        f"<text x='24' y='{top + plot_h / 2}' text-anchor='middle' font-family='Arial, sans-serif' font-size='12' fill='#39434d' transform='rotate(-90 24 {top + plot_h / 2})'>FID</text>"
    )

    legend_y = top + 10
    for idx, family in enumerate(families):
        color = COLORS.get(family, "#333333")
        label = LABELS.get(family, family)
        per_run = grouped[family]
        for pts in per_run.values():
            pts = sorted(pts)
            d = " ".join(f"{sx(s):.1f},{sy(fid):.1f}" for s, fid in pts)
            lines.append(f"<polyline points='{d}' fill='none' stroke='{color}' stroke-width='1.1' opacity='0.22'/>")

        by_step: dict[int, list[float]] = defaultdict(list)
        for pts in per_run.values():
            for step, fid in pts:
                by_step[step].append(fid)
        mean_pts = [(step, mean(vals)) for step, vals in sorted(by_step.items())]
        d = " ".join(f"{sx(s):.1f},{sy(fid):.1f}" for s, fid in mean_pts)
        lines.append(f"<polyline points='{d}' fill='none' stroke='{color}' stroke-width='3.0'/>")

        ly = legend_y + idx * 25
        lx = left + plot_w + 35
        lines.append(f"<line x1='{lx}' y1='{ly}' x2='{lx + 28}' y2='{ly}' stroke='{color}' stroke-width='3'/>")
        lines.append(
            f"<text x='{lx + 36}' y='{ly + 4}' font-family='Arial, sans-serif' font-size='12' fill='#1f2933'>{html_escape(label)} (n={len(per_run)})</text>"
        )

    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")


def svg_bar_plot(
    path: Path,
    *,
    title: str,
    subtitle: str,
    rows: list[dict[str, object]],
    value_key: str,
    value_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [float(r[value_key]) for r in rows if str(r[value_key]) != "NA"]
    y_max = max(values) * 1.15 if values else 1.0
    width, height = 980, 520
    left, right, top, bottom = 90, 70, 82, 115
    plot_w = width - left - right
    plot_h = height - top - bottom
    bar_gap = 24
    bar_w = (plot_w - bar_gap * (len(rows) - 1)) / max(1, len(rows))

    lines = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='#fbfaf7'/>",
        f"<text x='{left}' y='34' font-family='Arial, sans-serif' font-size='22' font-weight='700' fill='#1f2933'>{html_escape(title)}</text>",
        f"<text x='{left}' y='58' font-family='Arial, sans-serif' font-size='13' fill='#52616b'>{html_escape(subtitle)}</text>",
        f"<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' stroke='#39434d' stroke-width='1.2'/>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' stroke='#39434d' stroke-width='1.2'/>",
    ]
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        y = top + plot_h - frac * plot_h
        value = frac * y_max
        lines.append(f"<line x1='{left}' y1='{y:.1f}' x2='{left + plot_w}' y2='{y:.1f}' stroke='#e2e8ee'/>")
        lines.append(
            f"<text x='{left - 12}' y='{y + 4:.1f}' text-anchor='end' font-family='Arial, sans-serif' font-size='11' fill='#52616b'>{value:.0f}</text>"
        )

    for idx, row in enumerate(rows):
        family = str(row["family_id"])
        value = float(row[value_key])
        x = left + idx * (bar_w + bar_gap)
        y = top + plot_h - (value / y_max) * plot_h
        color = COLORS.get(family, "#333333")
        lines.append(
            f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' height='{top + plot_h - y:.1f}' fill='{color}' opacity='0.88'/>"
        )
        lines.append(
            f"<text x='{x + bar_w / 2:.1f}' y='{y - 8:.1f}' text-anchor='middle' font-family='Arial, sans-serif' font-size='12' fill='#1f2933'>{value:.1f}</text>"
        )
        label = html_escape(row.get("label", family))
        lines.append(
            f"<text x='{x + bar_w / 2:.1f}' y='{top + plot_h + 24}' text-anchor='middle' font-family='Arial, sans-serif' font-size='11' fill='#1f2933'>{label}</text>"
        )
        lines.append(
            f"<text x='{x + bar_w / 2:.1f}' y='{top + plot_h + 42}' text-anchor='middle' font-family='Arial, sans-serif' font-size='10' fill='#52616b'>n={row.get('runs', '')}</text>"
        )
    lines.append(
        f"<text x='25' y='{top + plot_h / 2}' text-anchor='middle' font-family='Arial, sans-serif' font-size='12' fill='#39434d' transform='rotate(-90 25 {top + plot_h / 2})'>{html_escape(value_label)}</text>"
    )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")


def write_readme(out_dir: Path) -> None:
    readme = """# Paper-ready current-CIFAR assets

This folder is generated by:

```bash
python3.11 scripts/current/build_current_cifar_paper_assets.py
```

Canonical inputs:

- `../training_time_by_run.csv`
- `../fid_vs_step_with_wallclock.csv`

The generated tables group by `family_id` for paper-facing quality claims.
This avoids splitting equivalent `pipe_P4_equal` rows across raw `regime`
names such as `pipeline` and `strict_pipeline`.

Generated assets:

- `main_500k_family_summary.csv`: family-level 500k mean/std/median table.
- `main_300k_o3_summary.csv`: O3 300k seed top-up table.
- `shallow_chain_o5_summary.csv`: O5 shallow-chain mechanism controls.
- `trajectory_coverage.csv`: 5k-grid completeness audit for all logical runs.
- `fid_vs_step_main_500k.svg`: seed-level 500k trajectory figure.
- `pipeline_equal_vs_deepest_500k.svg`: mechanism ablation figure.
- `shallow_chain_control_best_fid.svg`: shallow-chain control figure.
"""
    (out_dir / "README.md").write_text(readme)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-subdir", default="results/current_cifar/paper_ready")
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    current = repo / "results" / "current_cifar"
    out_dir = repo / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = read_csv(current / "training_time_by_run.csv")
    points = read_csv(current / "fid_vs_step_with_wallclock.csv")

    summary_fields = [
        "phase_group",
        "family_id",
        "label",
        "horizon_steps",
        "runs",
        "seeds",
        "best_fid_mean",
        "best_fid_std",
        "best_fid_median",
        "final_fid_mean",
        "final_fid_std",
        "final_fid_median",
    ]
    main_500k = family_summary(
        runs,
        phase_group="O4",
        horizon_steps=500000,
        families=[
            "single_P1_terminal",
            "single_P2_equal",
            "single_P2_deepest",
            "ddp_P1_terminal",
            "pipe_P4_equal",
            "pipe_P4_deepest",
        ],
    )
    write_csv(out_dir / "main_500k_family_summary.csv", main_500k, summary_fields)

    main_300k = family_summary(
        runs,
        phase_group="O3",
        horizon_steps=300000,
        families=[
            "single_P1_terminal",
            "single_P2_equal",
            "single_P2_deepest",
            "ddp_P1_terminal",
            "pipe_P4_equal",
            "pipe_P2_equal",
        ],
    )
    write_csv(out_dir / "main_300k_o3_summary.csv", main_300k, summary_fields)

    o5 = family_summary(
        runs,
        phase_group="O5",
        horizon_steps=300000,
        families=[
            "ddp_P1_terminal_K25",
            "ddp_P1_terminal_K50",
            "ddp_P1_terminal_K100",
            "pipe_P4_equal_K100",
            "pipe_P4_deepest_K100",
        ],
    )
    write_csv(out_dir / "shallow_chain_o5_summary.csv", o5, summary_fields)

    coverage = trajectory_coverage(runs, points)
    write_csv(
        out_dir / "trajectory_coverage.csv",
        coverage,
        [
            "logical_run_id",
            "phase_group",
            "family_id",
            "horizon_steps",
            "seed",
            "points",
            "expected_points",
            "first_step",
            "last_step",
            "complete_5k_grid",
        ],
    )

    svg_line_plot(
        out_dir / "fid_vs_step_main_500k.svg",
        title="Current CIFAR 500k FID trajectories",
        subtitle="Thin lines are seeds; thick lines are family means. Source: fid_vs_step_with_wallclock.csv.",
        points=points,
        runs=runs,
        phase_group="O4",
        horizon_steps=500000,
        families=[
            "single_P1_terminal",
            "single_P2_equal",
            "ddp_P1_terminal",
            "pipe_P4_equal",
            "pipe_P4_deepest",
        ],
    )

    ablation_rows = [r for r in main_500k if r["family_id"] in {"ddp_P1_terminal", "pipe_P4_equal", "pipe_P4_deepest"}]
    svg_bar_plot(
        out_dir / "pipeline_equal_vs_deepest_500k.svg",
        title="Intermediate energies matter beyond pipeline schedule",
        subtitle="Same K=100 setting; equal weighting is separated from deepest-only.",
        rows=ablation_rows,
        value_key="best_fid_mean",
        value_label="Best FID mean",
    )

    svg_bar_plot(
        out_dir / "shallow_chain_control_best_fid.svg",
        title="Shallow terminal chains do not explain the gain",
        subtitle="O5 mechanism controls; K=25 and K=50 terminal baselines collapse.",
        rows=o5,
        value_key="best_fid_mean",
        value_label="Best FID mean",
    )

    manifest_rows = [
        {"path": "main_500k_family_summary.csv", "type": "table", "source": "training_time_by_run.csv"},
        {"path": "main_300k_o3_summary.csv", "type": "table", "source": "training_time_by_run.csv"},
        {"path": "shallow_chain_o5_summary.csv", "type": "table", "source": "training_time_by_run.csv"},
        {"path": "trajectory_coverage.csv", "type": "audit", "source": "fid_vs_step_with_wallclock.csv"},
        {"path": "fid_vs_step_main_500k.svg", "type": "figure", "source": "fid_vs_step_with_wallclock.csv"},
        {"path": "pipeline_equal_vs_deepest_500k.svg", "type": "figure", "source": "training_time_by_run.csv"},
        {"path": "shallow_chain_control_best_fid.svg", "type": "figure", "source": "training_time_by_run.csv"},
    ]
    write_csv(out_dir / "asset_manifest.csv", manifest_rows, ["path", "type", "source"])
    write_readme(out_dir)

    complete = [r for r in coverage if r["complete_5k_grid"] == "yes"]
    print(f"Wrote paper-ready assets to {out_dir.relative_to(repo)}")
    print(f"Complete 5k trajectories: {len(complete)} / {len(coverage)}")


if __name__ == "__main__":
    main()
