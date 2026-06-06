# Active Experiment Ledgers

This file explains the active experiment families represented in the clean
repository. It is a provenance map, not a live queue snapshot.

For live queue status, check the original Polaris workspace:

```bash
qstat -u "$USER"
screen -ls
```

Then inspect the relevant watcher logs in the original run directory.

## Ledger Boundaries

Keep these ledgers separate:

| Ledger | Contents | Paper usage |
| --- | --- | --- |
| `current_cifar_paper` | Original CIFAR EBM model/objective, O1-O5 accounting, FID trajectories, wall-clock summaries. | Main paper evidence. |
| `historical_exploratory` | Old beta families, K=400 exploratory controls, smoke/debug/resume fragments, replay-only artifacts. | Do not count as current mainline. |
| `external_benchmark` | ImageNet-32 conditional and DRL-style backbone validation. | External validation only. |
| `long_k_scaling` | Original CIFAR model with larger Langevin depth and larger GPU/batch settings. | Scaling/mechanism evidence, separate from O1-O5 unless explicitly merged. |
| `overhead_attribution` | Sampling/training/communication microbenchmarks. | Supports system bottleneck analysis. |

When reporting counts, specify the unit:

- experiment family
- logical run: family x seed x horizon
- scheduler attempt or resume fragment

## Current CIFAR Paper Ledger

Canonical groups:

| Group | Purpose |
| --- | --- |
| `O1` | Single-side objective sweep. |
| `O2` | Distributed paired sweep. |
| `O3` | Seed top-up for current mainline. |
| `O4` | 500k long-horizon runs. |
| `O5` | Compute-matched shallow-chain controls. |

Primary provenance:

```text
provenance/runs_final_bundle/
```

Most important files:

| File | Meaning |
| --- | --- |
| `master_experiment_registry.csv` | One row per logical run. |
| `scheduler_attempts.csv` | One row per scheduler attempt or resume fragment. |
| `paper_manifest_current.md` | Human-readable paper-ledger status. |
| `manifest_diff_vs_claimed_plan.md` | Difference between filesystem state and claimed O1-O5 plan. |
| `current_cifar_fid_vs_step.csv` | FID-vs-step trajectory table. |
| `current_cifar_fid_vs_step_with_wallclock.csv` | FID trajectory with wall-clock alignment where available. |
| `current_cifar_walltime_all_62_runs.csv` | Wall-clock records by run. |
| `current_cifar_walltime_by_family.csv` | Aggregated wall-clock by family. |

Rebuild scripts:

```text
scripts/current/build_master_experiment_registry.py
scripts/current/collect_current_cifar_fid_vs_step.py
scripts/current/collect_seed_fid_trajectories.py
```

## Long-K Scaling

The long-K branch uses the original stable CIFAR model and objective. It does
not use DRL-style ImageNet code.

Current purpose:

- test whether larger negative-chain budgets make strict pipeline more useful
- compare full-K DDP against strict pipeline under larger `K`
- track wall-clock, GPU-hours, and throughput under larger batch/GPU settings

Primary provenance:

```text
provenance/runs_long_k_multinode_sweep/
provenance/runs_long_k_batch_scaling/
provenance/runs_long_k_scaling/
```

Important scripts:

```text
scripts/current/long_k_multinode_sweep_manifest.py
scripts/current/long_k_multinode_300k_continuations.py
scripts/current/submit_long_k_multinode_sweep_resumes.py
scripts/current/long_k_batch_scaling_manifest.py
scripts/current/collect_long_k_wallclock.py
```

Operational note:

- Recent preemptable long-K PBS files were configured for `72:00:00` walltime.
- If active jobs are interrupted, treat resume fragments as scheduler attempts,
  not separate logical runs.
- Add wall-clock across all attempts belonging to the same logical run before
  comparing system performance.

## Overhead Attribution

Purpose:

- determine how much of end-to-end time is sampling, training, synchronization,
  or communication
- avoid claiming that DDP communication alone explains the bottleneck unless
  the microbenchmarks support it

Primary provenance:

```text
provenance/runs_long_k_overhead_microbench/
provenance/runs_long_k_overhead_attribution/
```

Important scripts:

```text
scripts/current/ebm_overhead_microbench.py
scripts/current/long_k_overhead_microbench_manifest.py
scripts/current/long_k_overhead_attribution_manifest.py
```

## ImageNet-32 And DRL-Style External Validation

This ledger is external validation. It should not be mixed into the current
CIFAR O1-O5 totals.

ImageNet-32 conditional infrastructure:

```text
scripts/current/imagenet32_data.py
scripts/current/conditional_chain.py
scripts/current/conditional_replay.py
scripts/current/conditional_sampler.py
scripts/current/conditional_model.py
```

DRL-style backbone path:

```text
scripts/current/drl_resnet_energy.py
scripts/current/ebm_train_imagenet32_drl_sync.py
scripts/current/ebm_train_imagenet32_drl_single.py
scripts/current/imagenet32_drl_manifest.py
scripts/current/imagenet32_drl_stability_manifest.py
```

Current interpretation boundary:

- The DRL path adapts a ResNet-style energy backbone only.
- It does not use recovery likelihood.
- It does not use diffusion noise ladders.
- It is not the original CIFAR paper method.
- Early DRL-style ports were unstable under the ordinary EBM objective.
- Prime/temperature/loss-scale diagnostics are external-validation
  stabilization work, not replacement CIFAR evidence.

## What To Update When New Results Arrive

For current CIFAR:

1. rebuild the master registry
2. rebuild FID-vs-step
3. rebuild wall-clock summaries
4. update `provenance/runs_final_bundle/`

For long-K:

1. collect all scheduler attempts for each logical run
2. sum attempt wall-clock
3. update long-K wall-clock and FID summary CSVs
4. record whether the result is M0 screening, M1 300k, or longer-horizon work

For external ImageNet/DRL:

1. keep it under the external-benchmark ledger
2. record energy sign, scale, head type, prime/temperature/loss-scale settings
3. report stability separately from FID quality
