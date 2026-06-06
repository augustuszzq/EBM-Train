# Resume 500k Baselines Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Resume the four failed 500k baseline/DDP training jobs from their latest checkpoints, then document why PBS terminated them.

**Architecture:** First inspect whether the current ablation launcher and training entrypoints already support checkpoint resume. If they do, reuse the existing path and only resubmit with resume-specific environment/config. If they do not, add the smallest possible resume hook, verify it locally, and then resubmit the four interrupted jobs while separately collecting PBS scheduler evidence.

**Tech Stack:** PBS/qsub, bash launchers, Python training scripts, JSON manifest/job metadata.

### Task 1: Inspect existing resume support

**Files:**
- Read: `scripts/pbs_run_ablation_case.pbs`
- Read: `scripts/ebm_train_sync_mode_a.py`
- Read: `scripts/ebm_train_single_pipe_emul.py`
- Read: `scripts/pbs_repro_orig_single_fid.sh`

**Step 1:** Search for checkpoint/resume flags or loading logic.

**Step 2:** Identify whether single and ddp/fullk paths can resume without code changes.

### Task 2: Add minimal resume support only if missing

**Files:**
- Modify only if needed: `scripts/pbs_run_ablation_case.pbs`
- Modify only if needed: `scripts/ebm_train_sync_mode_a.py`
- Test only if needed: `tests/test_pbs_run_ablation_case_launcher.py`

**Step 1:** Write/extend a targeted test for resume argument propagation.

**Step 2:** Run the targeted test and confirm it fails before the change.

**Step 3:** Implement the smallest resume hook.

**Step 4:** Re-run the targeted test and confirm it passes.

### Task 3: Resubmit interrupted runs from latest checkpoints

**Files:**
- Read/write metadata via: `experiments/ablation_manifest_500k_submitted.csv`

**Step 1:** Detect the latest checkpoint for each failed run.

**Step 2:** Submit four resumed PBS jobs with updated run dirs and walltimes.

**Step 3:** Record new job IDs and resume sources back into submission metadata.

### Task 4: Collect scheduler kill evidence

**Files:**
- Read: PBS history via `qstat -xf`
- Optionally write summary note to `docs/`

**Step 1:** Capture exit codes, walltimes, and scheduler comments for the failed jobs.

**Step 2:** Compare failed jobs against the successful pipeline jobs.

**Step 3:** Summarize the most likely scheduler-side root cause.
