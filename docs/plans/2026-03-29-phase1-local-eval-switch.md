# Phase1 Local Eval Switch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stop spending PBS queue capacity on Phase 1 eval jobs by switching future submissions to train-only mode and running eval locally against completed run directories.

**Architecture:** Keep the current training PBS path intact. Extract the eval body into a reusable shell entrypoint that both the PBS wrapper and a new local batch-eval script can call. Update the Phase 1 submitter to support incremental resubmission and an eval submission mode that can skip PBS eval jobs entirely.

**Tech Stack:** Bash, Python 3, PBS/qsub, existing `eval_generate.py` and `eval_metrics.py`.

### Task 1: Add a reusable eval entrypoint

**Files:**
- Create: `scripts/run_eval_ablation_case.sh`
- Modify: `scripts/pbs_eval_ablation_case.pbs`

**Step 1: Write the failing test**

Create a launcher test that asserts the PBS eval wrapper delegates to a reusable runtime script.

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest /lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/tests/test_ablation_eval_launchers.py -q`

**Step 3: Write minimal implementation**

Move the current eval body into `scripts/run_eval_ablation_case.sh` and make the PBS script source environment then `exec` the reusable runtime script.

**Step 4: Run test to verify it passes**

Run the same pytest target and verify pass.

### Task 2: Make submitter incremental and train-only capable

**Files:**
- Modify: `scripts/submit_ablation_phase1.sh`
- Test: `tests/test_submit_ablation_phase1.py`

**Step 1: Write the failing tests**

Add tests for:
- preserving existing submitted rows instead of recreating the whole file
- `--eval-mode local` skipping PBS eval submission
- persisting `train_job_id` even if eval submission fails

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest /lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/tests/test_submit_ablation_phase1.py -q`

**Step 3: Write minimal implementation**

Update the submitter to:
- load and merge an existing submitted manifest when present
- support `--eval-mode pbs|local`
- leave `eval_job_id` empty in local mode
- persist partial train submission state before exiting on eval submission failure

**Step 4: Run tests to verify they pass**

Run the same pytest target and verify pass.

### Task 3: Add a local Phase 1 batch eval script

**Files:**
- Create: `scripts/run_local_eval_phase1.py`
- Modify: `scripts/collect_ablation_phase1.py` only if needed for new status handling

**Step 1: Write the failing test**

Add a test that verifies completed train rows can be selected for local eval and rows with existing `metrics_compare.json` are skipped.

**Step 2: Run test to verify it fails**

Run the focused pytest target.

**Step 3: Write minimal implementation**

Create a script that:
- reads `experiments/ablation_manifest_phase1_submitted.csv`
- selects rows with finished train and missing eval output
- sets env vars and calls `scripts/run_eval_ablation_case.sh`
- supports `--exp-id`, `--group`, `--max-cases`, `--cuda-visible-devices`, and `--dry-run`

**Step 4: Run test to verify it passes**

Run the focused pytest target and verify pass.

### Task 4: Apply the operational switch

**Files:**
- Modify: `experiments/ablation_manifest_phase1_submitted.csv`

**Step 1: Cancel queued eval jobs**

Cancel pending or held `abev*` Phase 1 jobs so queue slots go to train jobs.

**Step 2: Sync the manifest**

Clear canceled `eval_job_id` values for cases that will now be evaluated locally. Preserve train job ids and run dirs.

**Step 3: Resume remaining train submissions**

Use the updated submitter in local-eval mode to fill remaining `train_job_id` gaps.

**Step 4: Verify**

Run:
- `bash -n /lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/pbs_eval_ablation_case.pbs /lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/run_eval_ablation_case.sh`
- `python3 -m pytest /lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/tests/test_ablation_eval_launchers.py /lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/tests/test_submit_ablation_phase1.py -q`
- queue / manifest sanity checks
