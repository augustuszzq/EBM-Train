# ImageNet-32 DRL-Backbone Validation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a clean, separate ImageNet-32 external-validation path that uses a DRL-style ResNet energy backbone with the existing completion-aware EBM objective and strict pipeline semantics.

**Architecture:** Keep the existing CIFAR/ImageNet trainers untouched. Add a DRL ResNet energy module, then add thin DRL-specific training entrypoints that import the existing trainers, replace only their `build_energy_model` binding, and call the original `main()`.

**Tech Stack:** PyTorch, existing `scripts/current` training runtime, PBS scripts, pytest.

### Task 1: DRL Backbone Model

**Files:**
- Create: `scripts/current/drl_resnet_energy.py`
- Test: `tests/test_drl_resnet_energy.py`

**Steps:**
1. Write tests for `DRLResNetEnergy.forward(x)` returning shape `[B]`, accepting ignored labels as `forward(x, y)`, optional 16x16 attention, and spectral normalization.
2. Run targeted pytest and verify it fails because the module does not exist.
3. Implement the minimal DRL-style ResNet energy backbone: conv stem, residual/downsample blocks, optional self-attention, global pooling, scalar head.
4. Re-run targeted pytest and verify it passes.

### Task 2: DRL Training Wrappers

**Files:**
- Create: `scripts/current/ebm_train_imagenet32_drl_sync.py`
- Create: `scripts/current/ebm_train_imagenet32_drl_single.py`
- Test: `tests/test_imagenet32_drl_wrappers.py`

**Steps:**
1. Write tests that verify the wrappers expose a DRL model builder and do not require modifying the original trainer modules.
2. Run targeted pytest and verify it fails because wrappers do not exist.
3. Implement wrappers that patch the imported trainer module's `build_energy_model` to return `DRLResNetEnergy` from the YAML `model:` block.
4. Re-run targeted pytest and verify it passes.

### Task 3: Configs, Manifest, And Launch Scripts

**Files:**
- Create: `configs/imagenet32_drl_ddp_k100.yaml`
- Create: `configs/imagenet32_drl_pipeline_p4_equal_k100.yaml`
- Create: `configs/imagenet32_drl_ddp_smoke.yaml`
- Create: `configs/imagenet32_drl_pipeline_p4_equal_smoke.yaml`
- Create: `scripts/current/imagenet32_drl_manifest.py`
- Create: `scripts/current/pbs_run_imagenet32_drl_case.pbs`
- Create: `scripts/current/submit_imagenet32_drl.sh`
- Create: `scripts/run_imagenet32_drl_ddp_smoke.sh`
- Create: `scripts/run_imagenet32_drl_pipeline_smoke.sh`
- Test: `tests/test_imagenet32_drl_manifest.py`

**Steps:**
1. Write tests that verify the manifest contains Phase 0 smoke DDP and pipeline rows, points to DRL configs, uses ImageNet-32, and defaults later phases to `submit=no`.
2. Run targeted pytest and verify it fails because the manifest module does not exist.
3. Add configs and scripts. The PBS script should mirror existing run-dir artifacts but call the new DRL wrappers.
4. Run `bash -n` on scripts, `py_compile` on Python entrypoints, and targeted pytest.

### Task 4: Verification

**Steps:**
1. Run the targeted DRL tests.
2. Run syntax checks for new shell/PBS scripts.
3. Report the exact files added and the commands run.
