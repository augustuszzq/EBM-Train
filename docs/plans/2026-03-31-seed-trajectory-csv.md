# Seed Trajectory CSV Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Export seed trajectory re-evaluation points into one reusable CSV so `seed1/seed2` can be written now and `seed3` can be appended later by rerunning the same script.

**Architecture:** Add a small collector that scans known trajectory output directories, reads each `step*/metrics_compare.json`, and emits a flat CSV with one row per evaluated checkpoint. Keep the CSV schema simple and stable so later analysis can sort or filter by `seed`, `method`, and `step`.

**Tech Stack:** Python 3, `pathlib`, `csv`, existing trajectory output directories under `runs_analysis/`.

### Task 1: Add a failing test for aggregated trajectory export

**Files:**
- Create: `tests/test_collect_seed_fid_trajectories.py`
- Create: `scripts/collect_seed_fid_trajectories.py`

**Step 1: Write the failing test**

Write a focused test that creates fake `seed1/seed2` trajectory directories with `metrics_compare.json` files, runs the collector helper, and asserts:
- all rows are exported
- `seed`, `method`, `step`, and `fid_inception` columns are present
- rows are sorted by seed, method, then step

**Step 2: Run test to verify it fails**

Run:

```bash
python3 -m pytest /lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/tests/test_collect_seed_fid_trajectories.py -q
```

Expected: FAIL because the collector module does not exist yet.

### Task 2: Implement the minimal collector

**Files:**
- Create: `scripts/collect_seed_fid_trajectories.py`

**Step 1: Implement a small reusable API**

Add helpers to:
- discover rows from one trajectory directory
- aggregate rows across multiple seeds/methods
- write a CSV

Keep the output schema minimal:
- `seed`
- `method`
- `label`
- `step`
- `fid_inception`
- `fid_feature`
- `unique_ratio`
- `trajectory_dir`
- `metrics_path`

**Step 2: Run the focused test**

Run:

```bash
python3 -m pytest /lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/tests/test_collect_seed_fid_trajectories.py -q
```

Expected: PASS.

### Task 3: Generate the real CSV for current trajectories

**Files:**
- Update runtime output under `runs_analysis/`

**Step 1: Run the collector against real data**

Generate:

```text
/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_analysis/seed_fid_trajectory_points.csv
```

The first run should contain the completed `seed1` + `seed2` trajectories. After `seed3` finishes, rerun the same command to refresh the CSV to the full `360` rows.

**Step 2: Verify row counts**

Confirm the current CSV contains `240` rows from:
- `seed1 ddp`
- `seed1 pipeline`
- `seed2 ddp`
- `seed2 pipeline`

Later, confirm rerunning the collector after `seed3` completion updates the CSV to `360` rows.
