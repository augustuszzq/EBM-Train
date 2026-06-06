# ImageNet-32 Phase A Preemptable Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a reproducible `preemptable` submission path for real ImageNet-32 conditional Phase A runs, including dataset preparation, 1k/5k/20k training cases, and machine-readable bookkeeping.

**Architecture:** Reuse the active three-mode training entrypoints and the generic ablation PBS launcher, but first fix benchmark config propagation so ImageNet-32 configs actually reach runtime selection. Add one dedicated dataset-materialization PBS job, then submit a Phase A manifest of single/DDP/pipeline cases that depend on data prep completion. Keep eval local for now; this phase is about train-side semantic validation on real data.

**Tech Stack:** Bash/PBS, Python CSV manifest generation, existing `ebm_train_sync_mode_a.py`, existing `ebm_train_single_pipe_emul.py`, HF-backed ImageNet-32 materialization helpers.

### Task 1: Fix benchmark config propagation in the generic PBS launcher

**Files:**
- Modify: `/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/current/pbs_run_ablation_case.pbs`
- Test: `/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/tests/test_pbs_run_ablation_case_launcher.py`

**Step 1: Write the failing test**

Add assertions that `CONFIG` is exported and forwarded into both single and distributed train argument lists.

**Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=/eagle/lc-mpi/Zhiqing OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/kevienzzq/.conda/envs/llm-env/bin/python -m pytest tests/test_pbs_run_ablation_case_launcher.py::test_ablation_pbs_propagates_benchmark_config_when_present -q
```

**Step 3: Write minimal implementation**

Add `CONFIG="${CONFIG:-}"`, export it, persist it into `config_resolved.json`, and append `--config "${CONFIG}"` to both single and distributed launch paths.

**Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH=/eagle/lc-mpi/Zhiqing OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/kevienzzq/.conda/envs/llm-env/bin/python -m pytest tests/test_pbs_run_ablation_case_launcher.py -q
```

### Task 2: Add restart-safe ImageNet-32 dataset-prep submission entrypoint

**Files:**
- Create: `/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/current/pbs_materialize_imagenet32.pbs`
- Modify: `/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/current/export_hf_imagenet32.py`
- Modify: `/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/current/imagenet32_driver.py`
- Test: `/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/tests/test_imagenet32_data.py`
- Test: `/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/tests/test_imagenet32_scripts.py`

**Step 1: Write the failing test**

Cover tiny materialization limits and driver passthrough for `max_examples`, plus shell-level expectations for the dataset-prep script if needed.

**Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=/eagle/lc-mpi/Zhiqing OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/kevienzzq/.conda/envs/llm-env/bin/python -m pytest tests/test_imagenet32_data.py tests/test_imagenet32_scripts.py -q
```

**Step 3: Write minimal implementation**

Keep one dataset-prep job that materializes `train` and `val` into `/eagle/lc-mpi/Zhiqing/polaris_ebm/data/imagenet32`, writes `_SUCCESS.json`, and is safe to rerun.

**Step 4: Run test to verify it passes**

Run the same pytest slice and:
```bash
/home/kevienzzq/.conda/envs/llm-env/bin/python -m py_compile scripts/current/imagenet32_data.py scripts/current/export_hf_imagenet32.py scripts/current/imagenet32_driver.py
```

### Task 3: Add Phase A manifest and submitter

**Files:**
- Create: `/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/current/imagenet32_phasea_manifest.py`
- Create: `/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/current/submit_imagenet32_phasea.sh`
- Create: `/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/experiments/imagenet32_phasea_manifest.csv`
- Create: `/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/experiments/imagenet32_phasea_submitted.csv`
- Create: `/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/docs/imagenet32_phasea_plan.md`

**Step 1: Write the failing test**

Add tests that the manifest rows cover `single/ddp/pipeline` at `1k/5k/20k`, and that the submitter emits a dataset-prep job plus dependent train jobs in dry-run mode.

**Step 2: Run test to verify it fails**

Run targeted pytest for the new manifest/submitter tests.

**Step 3: Write minimal implementation**

Reuse the generic ablation submitter shape:
- one materialization `qsub`
- train rows with `CONFIG` pointing to the right ImageNet-32 YAML
- local eval mode for now
- dependency `afterok:<materialize_job_id>` on all Phase A train jobs

**Step 4: Run test to verify it passes**

Run the new targeted pytest slice plus `bash -n` on the new PBS/submit shell.

### Task 4: Dry-run and submit

**Files:**
- Modify if needed: `/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/docs/imagenet32_benchmark_status.md`

**Step 1: Run dry-run verification**

Run:
```bash
bash -n scripts/current/pbs_materialize_imagenet32.pbs scripts/current/submit_imagenet32_phasea.sh
PYTHONPATH=/eagle/lc-mpi/Zhiqing /home/kevienzzq/.conda/envs/llm-env/bin/python scripts/current/imagenet32_phasea_manifest.py --out /tmp/imagenet32_phasea_manifest.csv
bash scripts/current/submit_imagenet32_phasea.sh --dry-run
```

**Step 2: Submit if clean**

Submit the dataset-prep job and dependent Phase A jobs to `preemptable`.

**Step 3: Record outputs**

Update the status markdown with:
- materialization target path
- job ids
- manifest paths
- what still remains local-only vs preemptable-ready
