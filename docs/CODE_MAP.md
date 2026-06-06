# Code Map

This file gives collaborators a short path from scientific question to source
file.

## Method Semantics

The method is defined by a completion-aware weighted negative-energy objective:

```text
L_t = f_pos - sum_s alpha_s * f_neg(x_{t,s})
```

The weights apply to scalar energy terms. No image-space stage averaging is
allowed.

Relevant files:

- `scripts/current/ebm_train_sync_mode_a.py`
- `scripts/current/ebm_train_single_pipe_emul.py`
- `scripts/current/mode_a_contract.py`
- `tests/test_mode_a_contract.py`
- `tests/test_mode_a_langevin.py`

## Single/DDP/Pipeline Training

- Single full-K baseline:
  `scripts/current/ebm_train_baseline_single.py`
- DDP full-K and strict pipeline:
  `scripts/current/ebm_train_sync_mode_a.py`
- Single-GPU pipeline emulation:
  `scripts/current/ebm_train_single_pipe_emul.py`
- PBS wrapper:
  `scripts/current/pbs_run_ablation_case.pbs`

## Current CIFAR Paper Registry

- Build the canonical registry:
  `scripts/current/build_master_experiment_registry.py`
- Canonical run registry:
  `scripts/current/build_canonical_run_registry.py`
- FID x step collector:
  `scripts/current/collect_current_cifar_fid_vs_step.py`
- Seed trajectory collector:
  `scripts/current/collect_seed_fid_trajectories.py`

Lightweight registry outputs are copied under:

```text
provenance/runs_final_bundle/
```

## Long-K Scaling

- Main manifest generator:
  `scripts/current/long_k_multinode_sweep_manifest.py`
- 300k continuation generator/resumer:
  `scripts/current/long_k_multinode_300k_continuations.py`
- M0 resume helper:
  `scripts/current/submit_long_k_multinode_sweep_resumes.py`
- Batch scaling manifest:
  `scripts/current/long_k_batch_scaling_manifest.py`
- Wall-clock collector:
  `scripts/current/collect_long_k_wallclock.py`

Generated configs, PBS files, and summary CSVs are copied under:

```text
provenance/runs_long_k_multinode_sweep/
provenance/runs_long_k_batch_scaling/
provenance/runs_long_k_scaling/
```

## Overhead Attribution

These are lower priority than the main long-K branch, but the code is kept:

- `scripts/current/ebm_overhead_microbench.py`
- `scripts/current/long_k_overhead_microbench_manifest.py`
- `scripts/current/long_k_overhead_attribution_manifest.py`

Generated lightweight artifacts are under:

```text
provenance/runs_long_k_overhead_microbench/
provenance/runs_long_k_overhead_attribution/
```

## ImageNet-32 Conditional and DRL-Style External Validation

Conditional infrastructure:

- `scripts/current/conditional_chain.py`
- `scripts/current/conditional_replay.py`
- `scripts/current/conditional_sampler.py`
- `scripts/current/conditional_model.py`
- `scripts/current/imagenet32_data.py`

DRL-style backbone path:

- `scripts/current/drl_resnet_energy.py`
- `scripts/current/ebm_train_imagenet32_drl_sync.py`
- `scripts/current/ebm_train_imagenet32_drl_single.py`
- `scripts/current/imagenet32_drl_manifest.py`
- `scripts/current/imagenet32_drl_stability_manifest.py`
- `scripts/current/submit_imagenet32_drl_prime20k_successors.py`

Evaluation:

- `scripts/current/eval_imagenet32_fid.py`
- `scripts/current/eval_imagenet32_conditional_acc.py`
- `scripts/current/aggregate_imagenet32_results.py`

## Launchers

Current HSN launch wrappers:

- `launchers/current/run_hsn.sh`
- `launchers/current/run_hsn_baseline.sh`
- `launchers/current/run_hsn_pipeline.sh`

Archive launchers are intentionally excluded from this clean repo.

