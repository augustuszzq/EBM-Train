# Seed1 FID Trajectory Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Evaluate the existing `seed1` DDP and pipeline `300k` checkpoints every `5000` steps, record the full 60-point FID trajectory for each run, and summarize whether the final FID matches the previously recorded result plus where the best/decline region occurs.

**Architecture:** Add one reusable local analysis script that discovers `artifacts/checkpoints/ckpt_step*.pt`, runs the existing canonical `eval_generate.py` and `eval_metrics.py` for each selected checkpoint, and writes one trajectory CSV/JSON/Markdown summary per run plus one combined summary. Keep the training runs unchanged and reuse the exact historical eval hyperparameters (`K_eval=100`, `step_size=1.0`, `noise_std=0.01`, `langevin_sign=1.0`, `seed=1`, `no_clamp_x=True`) so the final-point comparison is meaningful.

**Tech Stack:** Python 3, existing EBM eval scripts, pytest, local GPU execution.

### Task 1: Add failing tests for checkpoint trajectory support

**Files:**
- Create: `tests/test_eval_fid_trajectory.py`
- Inspect: `scripts/eval_generate.py`
- Inspect: `scripts/eval_metrics.py`

**Step 1: Write the failing tests**

Add tests for:
- discovering `ckpt_step*.pt` from `artifacts/checkpoints/`
- selecting every checkpoint at a fixed interval
- summarizing trajectory rows into `best_step`, `best_fid`, `final_step`, `final_fid`, and `final_match_delta`

**Step 2: Run test to verify it fails**

Run:

```bash
python3 -m pytest /lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/tests/test_eval_fid_trajectory.py -q
```

Expected: FAIL because the new trajectory module does not exist yet.

### Task 2: Implement the minimal trajectory evaluator

**Files:**
- Create: `scripts/eval_fid_trajectory.py`
- Test: `tests/test_eval_fid_trajectory.py`

**Step 1: Implement checkpoint discovery and summary helpers**

Implement pure functions for:
- parsing checkpoint steps
- collecting and sorting checkpoints from `artifacts/checkpoints/`
- building trajectory summaries from row dictionaries

**Step 2: Implement CLI execution**

Add CLI support for:
- `--run-dir`
- `--label`
- `--out-dir`
- `--step-interval`
- `--eval-images`
- `--eval-batch`
- `--device`
- `--skip-existing`

For each checkpoint:
- call `scripts/eval_generate.py`
- call `scripts/eval_metrics.py`
- store outputs under a trajectory-specific eval directory
- append a row to a trajectory CSV/JSON

**Step 3: Run the tests to verify they pass**

Run:

```bash
python3 -m pytest /lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/tests/test_eval_fid_trajectory.py -q
```

Expected: PASS.

### Task 3: Verify script syntax and run the real seed1 evaluations

**Files:**
- Verify: `scripts/eval_fid_trajectory.py`

**Step 1: Syntax check**

Run:

```bash
python3 -m py_compile /lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/eval_fid_trajectory.py
```

Expected: no output, exit `0`.

**Step 2: Launch DDP seed1 trajectory**

Run the script on:

```text
/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_dual/modea_fullk_align128/ddp_fullk_s300k_seed1_syncinit_20260324_165322
```

using the historical eval settings.

**Step 3: Launch pipeline seed1 trajectory**

Run the script on:

```text
/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_pipeline_strict/pipeline_strict_s300k_b001_lr1e-4_seed1_syncinit_20260324_165322
```

using the same eval settings.

### Task 4: Produce the comparison summary

**Files:**
- Output: `runs_analysis/seed1_fid_trajectory/combined_summary.json`
- Output: `runs_analysis/seed1_fid_trajectory/combined_summary.md`

**Step 1: Confirm final-point alignment**

For both runs, compare:
- trajectory final FID at `step299999`
- previously recorded final FID in the original run eval directory

**Step 2: Identify where quality bottoms out or starts worsening**

Summarize:
- `best_step`
- `best_fid`
- `final_step`
- `final_fid`
- first step entering key thresholds such as `<70` and `<60` when applicable

**Step 3: Report outcome**

Return the two 60-point trajectories plus a concise conclusion about:
- whether the final values match the historical records
- where the major improvement region occurs
- whether either curve later degrades again
