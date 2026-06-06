# Active/Archive Layout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorganize the repo so the current EBM workflow lives in dedicated active folders, legacy code moves into archive folders, and all live script paths are updated to the new layout.

**Architecture:** Keep run outputs and data directories untouched. Move active training/eval/PBS/submit scripts into `scripts/current`, move legacy scripts into `scripts/archive`, move currently used launchers into `launchers/current`, and archive old launchers under `launchers/archive`. Update all active references and only keep tests aligned with the active layout.

**Tech Stack:** Bash file moves, Python path updates, PBS launch scripts, pytest smoke tests.

### Task 1: Define active vs archive file sets

**Files:**
- Inspect: `scripts/*`
- Inspect: `run_hsn*.sh`
- Inspect: `run_ssh*.sh`
- Inspect: `tests/*`

**Step 1: Record active code set**

Active code includes the current three training modes, eval scripts, ablation/replay tooling, and current launchers.

**Step 2: Record archive code set**

Archive code includes old stream runtimes, deprecated baseline/pipeline launchers, plotting one-offs, and tests that only target archived code.

### Task 2: Create new directory structure

**Files:**
- Create: `scripts/current/`
- Create: `scripts/archive/`
- Create: `launchers/current/`
- Create: `launchers/archive/`
- Create: `tests/archive/`

**Step 1: Create the directories**

Create the new active/archive folders without touching `runs_*` or experiment outputs.

**Step 2: Move files**

Move active scripts and active launchers into their `current` folders. Move legacy scripts, legacy launchers, and archived tests into their `archive` folders.

### Task 3: Update active code paths

**Files:**
- Modify: active files under `scripts/current/`
- Modify: active files under `launchers/current/`
- Modify: active tests under `tests/`

**Step 1: Update `PROJECT_DIR/scripts/...` references**

Replace active script references with `PROJECT_DIR/scripts/current/...`.

**Step 2: Update Python package imports**

Replace active package imports from `polaris_ebm.scripts...` to `polaris_ebm.scripts.current...`.

**Step 3: Update launcher references**

Replace root launcher references with `launchers/current/...` and make wrappers resolve the repo root correctly from the new directory.

### Task 4: Validate the new layout

**Files:**
- Verify: `scripts/current/*.py`
- Verify: `scripts/current/*.sh`
- Verify: `scripts/current/*.pbs`
- Verify: `launchers/current/*.sh`
- Verify: active tests in `tests/`

**Step 1: Run syntax validation**

Run `bash -n` on active shell/PBS scripts and `python3 -m py_compile` on active Python entrypoints.

**Step 2: Run targeted tests**

Run only active-layout tests that exercise the moved scripts and launchers.

**Step 3: Add a short layout note**

Summarize the new `current` vs `archive` structure and the canonical entrypoints for single baseline, ddp strict, and pipeline strict.
