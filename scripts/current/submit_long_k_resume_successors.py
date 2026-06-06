#!/usr/bin/env python3
"""Submit bounded Long-K resume successors.

This is scheduler plumbing only. It does not change the training method.

For each logical Long-K run, the script checks:
  1. whether the target step is already reached,
  2. whether an original or resume job is active,
  3. whether a successor resume is already active.

If a run is incomplete and has no successor, it submits either:
  - an immediate resume from the latest checkpoint, or
  - an afterany-dependent successor using RESUME_CKPT=latest.
"""

import argparse
import csv
import getpass
import os
import re
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


ROOT = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/runs_long_k_scaling")
SUBMITTED = ROOT / "summaries" / "long_k_l1_l2_submitted.csv"
RESUMES = ROOT / "summaries" / "long_k_resume_attempts.csv"


def qstat_states() -> Dict[str, str]:
    user = os.environ.get("USER") or getpass.getuser()
    out = subprocess.check_output(["qstat", "-u", user]).decode("utf-8", "replace")
    states: Dict[str, str] = {}
    for line in out.splitlines():
        if ".polaris-pbs" not in line:
            continue
        parts = line.split()
        if not parts:
            continue
        qid = parts[0].split(".")[0]
        state = ""
        for tok in reversed(parts):
            if tok in {"R", "Q", "H", "E", "B", "S", "W"}:
                state = tok
                break
        if state:
            states[qid] = state
    return states


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def map_path(raw: str) -> Path:
    p = Path(raw)
    if p.exists():
        return p
    if raw.startswith("/eagle/"):
        p2 = Path("/lus/eagle/projects") / Path(raw[7:])
        if p2.exists():
            return p2
    return p


def target_from_exp(exp_id: str) -> Optional[int]:
    m = re.search(r"_s(\d+)k(?:_|$)", exp_id)
    if not m:
        return None
    return int(m.group(1)) * 1000


def latest_logged_step(run_dir: Path) -> Optional[int]:
    metrics = run_dir / "metrics_rank0.csv"
    if not metrics.exists():
        return None
    try:
        last = subprocess.check_output(["tail", "-n", "1", str(metrics)]).decode("utf-8", "replace").strip()
    except Exception:
        return None
    if not last or last.startswith("step,"):
        return None
    try:
        return int(float(last.split(",", 1)[0]))
    except Exception:
        return None


def latest_ckpt(run_dir: Path) -> Optional[Path]:
    candidates: List[Tuple[int, Path]] = []
    for p in (run_dir / "checkpoints").glob("ckpt_step*.pt"):
        m = re.search(r"ckpt_step(\d+)\.pt$", p.name)
        if not m:
            continue
        candidates.append((int(m.group(1)), p))
    for _, p in sorted(candidates, reverse=True):
        # Interrupted walltime can leave a partially written PyTorch zip. Avoid
        # repeatedly resubmitting from an unreadable newest checkpoint.
        if zipfile.is_zipfile(p):
            return p
    return None


def active_resume_by_exp(resume_rows: Iterable[dict], states: Dict[str, str]) -> Dict[str, List[str]]:
    active: Dict[str, List[str]] = {}
    for row in resume_rows:
        exp = row.get("logical_exp_id", "")
        jid = (row.get("resume_job_id", "") or "").split(".")[0]
        if jid and states.get(jid) in {"R", "Q", "H", "W"}:
            active.setdefault(exp, []).append(jid)
    return active


def append_resume(row: dict) -> None:
    RESUMES.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "logical_exp_id",
        "resume_job_id",
        "resume_from_ckpt",
        "submitted_at",
        "pbs_path",
        "run_dir",
        "notes",
    ]
    exists = RESUMES.exists()
    with RESUMES.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def submit_resume(
    exp_id: str,
    pbs_path: Path,
    run_dir: Path,
    resume_value: str,
    dependency_job: Optional[str],
    dry_run: bool,
) -> str:
    cmd = ["qsub"]
    if dependency_job:
        cmd += ["-W", "depend=afterany:%s" % dependency_job]
    cmd += ["-v", "RESUME_CKPT=%s" % resume_value, str(pbs_path)]
    if dry_run:
        return "DRY_RUN:" + " ".join(cmd)
    out = subprocess.check_output(cmd).decode("utf-8", "replace").strip()
    append_resume(
        {
            "logical_exp_id": exp_id,
            "resume_job_id": out,
            "resume_from_ckpt": resume_value,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "pbs_path": str(pbs_path),
            "run_dir": str(run_dir),
            "notes": (
                "dependent auto successor after %s; resolves latest checkpoint at job start"
                % dependency_job
                if dependency_job
                else "immediate auto resume from latest known checkpoint"
            ),
        }
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    states = qstat_states()
    submitted = read_csv(SUBMITTED)
    resumes = read_csv(RESUMES)
    active_resumes = active_resume_by_exp(resumes, states)

    candidates = []
    for row in submitted:
        exp = row["exp_id"]
        target = target_from_exp(exp)
        if target is None:
            continue
        run_dir = map_path(row["train_run_dir"])
        pbs_path = map_path(row["pbs_path"])
        step = latest_logged_step(run_dir)
        if step is not None and step >= target - 1:
            continue
        if active_resumes.get(exp):
            continue
        orig_job = (row.get("train_job_id", "") or "").split(".")[0]
        orig_state = states.get(orig_job, "not_in_q")
        ckpt = latest_ckpt(run_dir)
        if orig_state in {"R", "Q", "H", "W"}:
            candidates.append((exp, pbs_path, run_dir, "latest", orig_job, step, target))
        elif ckpt is not None:
            candidates.append((exp, pbs_path, run_dir, str(ckpt), None, step, target))

    submitted_count = 0
    for exp, pbs_path, run_dir, resume_value, dep, step, target in candidates:
        if submitted_count >= int(args.max_new):
            break
        job_id = submit_resume(
            exp_id=exp,
            pbs_path=pbs_path,
            run_dir=run_dir,
            resume_value=resume_value,
            dependency_job=dep,
            dry_run=bool(args.dry_run),
        )
        print("%s step=%s/%s dep=%s resume=%s job=%s" % (exp, step, target, dep or "", resume_value, job_id))
        submitted_count += 1

    print("submitted_count=%d" % submitted_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
