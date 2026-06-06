# Ablation Replay And 500k Submission Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Backfill checkpoint-by-checkpoint replay evaluation for completed ablation runs, render replay-based summary tables, and submit the requested 7 long-queue `500k` training jobs.

**Architecture:** Reuse the existing `eval_fid_trajectory.py` worker and `pbs_eval_fid_trajectory.pbs` wrapper. Add a replay manifest/collector layer that knows which completed runs already have trajectory outputs, which ones are partial and must be left alone, and which missing ones should be submitted to the `capacity` queue. For `500k`, reuse the generic `submit_ablation_phase1.sh` machinery with a dedicated manifest and `capacity` queue overrides.

**Tech Stack:** Python 3, CSV/JSON, existing ablation manifests, existing PBS wrappers, `qsub`, Markdown rendering.

### Task 1: Add replay helper tests first

**Files:**
- Create: `tests/test_ablation_replay.py`
- Create: `tests/test_ablation_500k_manifest.py`
- Create: `scripts/ablation_replay_common.py`
- Create: `scripts/ablation_500k_manifest.py`

**Step 1: Write failing tests**

Cover:
- replay row building from submitted manifests and existing trajectory overrides
- trajectory status detection (`done`, `partial`, `missing`)
- final replay summary including `best_fid`, `best_step`, and `final_minus_best`
- `500k` manifest row generation returning exactly the 7 requested experiments with `save_every=5000`

**Step 2: Run tests to verify failure**

Run:

```bash
PYTHONPATH=/eagle/lc-mpi/Zhiqing python3 -m pytest \
  /lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/tests/test_ablation_replay.py \
  /lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/tests/test_ablation_500k_manifest.py -q
```

Expected: FAIL because the helper modules do not exist yet.

### Task 2: Implement replay manifest + collector + renderer

**Files:**
- Create: `scripts/ablation_replay_common.py`
- Create: `scripts/build_ablation_replay_manifest.py`
- Create: `scripts/submit_ablation_replay.sh`
- Create: `scripts/collect_ablation_replay.py`
- Create: `scripts/render_ablation_replay_tables.py`
- Create: `experiments/ablation_replay_manifest.csv`
- Create: `experiments/ablation_replay_submitted.csv`

**Step 1: Build replay manifest**

Read:
- `experiments/ablation_manifest_phase1_submitted.csv`
- `experiments/ablation_manifest_phase2_submitted.csv`

Include all completed runs with checkpoints, but preserve current progress:
- already finished main seed trajectories reuse their existing `runs_analysis/seed*_fid_trajectory/*_local` directories
- currently running local A1 replay jobs reuse their existing `runs_analysis/ablation_fid_trajectory/...` directories
- all missing replay outputs point to `runs_ablation_replay/trajectories/<exp_id>/`

**Step 2: Submit only missing replay jobs**

Use `qsub` with:
- wrapper: `scripts/pbs_eval_fid_trajectory.pbs`
- queue: `capacity`
- walltime: `02:00:00` for `20k`/4-point runs, `06:00:00` for `300k`/60-point runs

Do not resubmit rows with existing or in-progress local replay outputs.

**Step 3: Collect replay outputs**

Write:
- `runs_ablation_replay/ablation_replay_points.csv`
- `runs_ablation_replay/ablation_final_summary.csv`
- `runs_ablation_replay/ablation_tables_filled.md`

The points CSV must include:
- `fid_inception`
- `fid_feature`
- `unique_ratio`
- `step`
- `seed`
- `exp_id`

The final summary must include:
- `final_fid`
- `best_fid`
- `best_step`
- `final_minus_best`
- `status`

The Markdown must render:
- baseline alignment
- weighted-sum
- pipeline factor
- K sweep
- step-size

If `abs(final_fid - best_fid) >= 5`, add a footnote line naming the best checkpoint.

### Task 3: Implement 500k manifest + submitter

**Files:**
- Create: `scripts/ablation_500k_manifest.py`
- Create: `scripts/submit_ablation_500k.sh`
- Create: `scripts/collect_ablation_500k.py`
- Create: `scripts/render_ablation_500k_report.py`
- Create: `experiments/ablation_manifest_500k.csv`
- Create: `experiments/ablation_manifest_500k_submitted.csv`

**Step 1: Generate exactly 7 rows**

Experiments:
- `single_fullk_K100_seed1_500k`
- `ddp_fullk_K100_seed1_500k`
- `ddp_fullk_K100_seed2_500k`
- `ddp_fullk_K100_seed3_500k`
- `pipe_strict_P4_K100_beta001_lr1e4_seed1_500k`
- `pipe_strict_P4_K100_beta001_lr1e4_seed2_500k`
- `pipe_strict_P4_K100_beta001_lr1e4_seed3_500k`

Use `queue=capacity`, keep `save_every=5000`, and leave eval mode as local.

**Step 2: Submit the 7 train jobs**

Reuse `scripts/submit_ablation_phase1.sh` with:
- `--manifest experiments/ablation_manifest_500k.csv`
- `--submitted-manifest experiments/ablation_manifest_500k_submitted.csv`
- `--runs-root /eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation_500k`
- `--payload-subdir 500k`
- `--eval-mode local`

### Task 4: Verify and report

**Step 1: Run focused tests**

Run:

```bash
PYTHONPATH=/eagle/lc-mpi/Zhiqing python3 -m pytest \
  /lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/tests/test_ablation_replay.py \
  /lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/tests/test_ablation_500k_manifest.py -q
```

**Step 2: Run syntax checks**

Run:

```bash
python3 -m py_compile \
  /lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/ablation_replay_common.py \
  /lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/build_ablation_replay_manifest.py \
  /lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/collect_ablation_replay.py \
  /lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/render_ablation_replay_tables.py \
  /lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/ablation_500k_manifest.py \
  /lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/collect_ablation_500k.py \
  /lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/render_ablation_500k_report.py
```

**Step 3: Verify generated outputs**

Confirm:
- replay manifest exists
- replay submitted manifest exists
- replay points/final/table files exist
- `500k` submitted manifest has exactly 7 train job ids
