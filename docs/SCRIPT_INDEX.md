# Script Index

This index classifies `scripts/current/` by role. It is meant to reduce the
first-reader cost of a directory with many active and retained experiment
scripts.

Status labels:

- `primary`: main code path for the current method or paper accounting.
- `support`: reusable helper, collector, evaluator, or launcher.
- `external`: ImageNet-32 / DRL-style external validation path.
- `appendix`: useful for secondary analysis or older experiment sections.
- `retained`: kept for provenance or reproducibility, but not the first file to
  read.

## Primary Training And Method Semantics

| File | Status | Role |
| --- | --- | --- |
| `mode_a_contract.py` | primary | Shared method-contract checks for loss signs, stage weights, and schedule semantics. |
| `ebm_train_sync_mode_a.py` | primary | DDP full-K and strict multi-GPU pipeline training entrypoint. |
| `ebm_train_baseline_single.py` | primary | Single-GPU strict full-K baseline. |
| `ebm_train_single_pipe_emul.py` | primary | Single-GPU serialized multi-stage objective emulation. |
| `benchmark_config.py` | support | Shared benchmark/config parsing helpers. |
| `benchmark_runtime.py` | support | Shared runtime/logging helpers. |
| `ablation_common.py` | support | Shared ablation utilities retained by current experiment scripts. |
| `ablation_replay_common.py` | support | Shared replay/FID trajectory utilities. |
| `ablation_replay_queue.py` | support | Replay backfill queue helper. |

## Current CIFAR Registry, FID, And Paper Tables

| File | Status | Role |
| --- | --- | --- |
| `build_master_experiment_registry.py` | primary | Builds the canonical logical-run registry for current CIFAR paper accounting. |
| `build_canonical_run_registry.py` | support | Builds normalized run registry records. |
| `collect_current_cifar_fid_vs_step.py` | primary | Produces the current CIFAR FID-vs-step table. |
| `collect_seed_fid_trajectories.py` | primary | Aggregates seed-level FID trajectories. |
| `backfill_current_cifar_trajectories.py` | support | Backfills missing current CIFAR trajectory points. |
| `run_blocked_current_cifar_fid_backfill.py` | support | Driver for blocked FID backfills. |
| `run_blocked_current_cifar_fid_backfill_and_collect.sh` | support | Shell wrapper for blocked backfill plus collection. |
| `run_current_cifar_trajectory_backfill_mpi.sh` | support | MPI wrapper for CIFAR trajectory backfill. |
| `run_manual_two_lane_backfill.py` | support | Manual two-lane local FID backfill runner. |
| `collect_experiment_bundle.py` | support | General experiment-bundle collector. |

## Evaluation Utilities

| File | Status | Role |
| --- | --- | --- |
| `eval_generate.py` | support | Generate samples for evaluation. |
| `eval_metrics.py` | support | Evaluation metrics helpers. |
| `eval_fid_trajectory.py` | support | FID trajectory evaluator. |
| `run_local_eval_phase1.py` | retained | Local phase-1 evaluation runner retained for reproducibility. |
| `run_local_eval_phase2.py` | retained | Local phase-2 evaluation runner retained for reproducibility. |
| `run_eval_ablation_case.sh` | support | Local/shell eval wrapper for ablation cases. |
| `pbs_eval_ablation_case.pbs` | support | PBS eval wrapper for ablation cases. |
| `pbs_eval_fid_trajectory.pbs` | support | PBS FID trajectory wrapper. |

## Objective-First, Pipeline-Then-Weighting, And Ablation Scripts

| File | Status | Role |
| --- | --- | --- |
| `collect_objective_first_followup.py` | primary | Collects objective-first follow-up results. |
| `objective_first_followup_live_queue.py` | support | Live queue helper for objective-first follow-up jobs. |
| `run_objective_first_followup_final_eval.py` | support | Runs final eval for objective-first follow-up. |
| `run_objective_first_followup_final_eval_mpi.sh` | support | MPI wrapper for objective-first final eval. |
| `submit_objective_first_followup.sh` | support | Submits objective-first follow-up jobs. |
| `collect_pipeline_then_weighting.py` | primary | Collects pipeline-then-weighting experiment results. |
| `collect_pipeline_then_weighting_replay.py` | support | Collects replay trajectory points for pipeline-then-weighting. |
| `build_pipeline_then_weighting_replay_manifest.py` | support | Builds replay manifest for pipeline-then-weighting. |
| `launch_pipeline_then_weighting_local.py` | support | Local launcher for pipeline-then-weighting. |
| `pipeline_then_weighting_live_queue.py` | support | Live queue helper for pipeline-then-weighting. |
| `run_pipeline_then_weighting_final_eval.py` | support | Final eval runner for pipeline-then-weighting. |
| `submit_pipeline_then_weighting.sh` | support | Submits pipeline-then-weighting manifest rows. |
| `pbs_eval_pipeline_then_weighting_case.pbs` | support | PBS eval wrapper for pipeline-then-weighting. |
| `pbs_run_pipeline_then_weighting_case.pbs` | support | PBS train wrapper for pipeline-then-weighting. |
| `ablation_500k_manifest.py` | appendix | 500k ablation manifest helper. |
| `build_ablation_500k_replay_manifest.py` | appendix | Builds replay manifest for 500k ablations. |
| `build_ablation_replay_manifest.py` | appendix | Builds replay manifest for ablations. |
| `collect_ablation_500k.py` | appendix | Collects 500k ablation summaries. |
| `collect_ablation_500k_trajectory.py` | appendix | Collects 500k ablation trajectories. |
| `collect_ablation_phase1.py` | appendix | Collects phase-1 ablation summaries. |
| `collect_ablation_phase2.py` | appendix | Collects phase-2 ablation summaries. |
| `collect_ablation_replay.py` | appendix | Collects ablation replay results. |
| `render_ablation_500k_report.py` | appendix | Renders 500k ablation report. |
| `render_ablation_phase1_report.py` | appendix | Renders phase-1 report. |
| `render_ablation_phase2_report.py` | appendix | Renders phase-2 report. |
| `render_ablation_replay_tables.py` | appendix | Renders replay tables. |
| `submit_ablation_500k.sh` | appendix | Submits 500k ablations. |
| `submit_ablation_phase1.sh` | appendix | Submits phase-1 ablations. |
| `submit_ablation_phase2.sh` | appendix | Submits phase-2 ablations. |
| `submit_ablation_replay.sh` | appendix | Submits ablation replay jobs. |
| `pbs_run_ablation_case.pbs` | appendix | PBS train wrapper for ablation case. |

## Long-K Scaling And Batch Scaling

| File | Status | Role |
| --- | --- | --- |
| `long_k_multinode_sweep_manifest.py` | primary | Generates long-K multi-node sweep configs and PBS files. |
| `long_k_multinode_300k_continuations.py` | primary | Generates/submits 300k continuation jobs for long-K sweep. |
| `submit_long_k_multinode_sweep_resumes.py` | primary | Resubmits unfinished long-K sweep fragments. |
| `submit_long_k_resume_successors.py` | support | Submits successor/resume jobs for long-K runs. |
| `watch_long_k_multinode_sweep.sh` | support | Watcher for long-K sweep jobs. |
| `watch_long_k_multinode_300k.sh` | support | Watcher for long-K 300k jobs. |
| `watch_long_k_successors.sh` | support | Watcher for long-K successor jobs. |
| `long_k_batch_scaling_manifest.py` | primary | Generates batch-size scaling configs. |
| `collect_long_k_wallclock.py` | primary | Aggregates long-K wall-clock and GPU-hour records. |
| `long_k_scaling_manifest.py` | retained | Earlier long-K scaling manifest generator. |

## Overhead Attribution

| File | Status | Role |
| --- | --- | --- |
| `ebm_overhead_microbench.py` | appendix | Microbenchmark for separating training/sampling/communication costs. |
| `long_k_overhead_microbench_manifest.py` | appendix | Generates overhead microbenchmark jobs. |
| `long_k_overhead_attribution_manifest.py` | appendix | Generates overhead attribution jobs. |
| `watch_long_k_overhead_attribution_submit.sh` | appendix | Watcher/submission helper for overhead attribution. |

## ImageNet-32 Conditional And DRL-Style External Validation

| File | Status | Role |
| --- | --- | --- |
| `imagenet32_data.py` | external | ImageNet-32 dataset loader/cache interface. |
| `conditional_chain.py` | external | Conditional chain-state abstraction. |
| `conditional_replay.py` | external | Label-aware replay buffer. |
| `conditional_sampler.py` | external | Conditional Langevin sampler. |
| `conditional_model.py` | external | Conditional model path. |
| `imagenet32_driver.py` | external | ImageNet-32 driver/helper. |
| `export_hf_imagenet32.py` | external | Hugging Face ImageNet-32 export/materialization helper. |
| `pbs_materialize_imagenet32.pbs` | external | PBS wrapper for ImageNet-32 materialization. |
| `imagenet32_phasea_manifest.py` | external | Phase-A ImageNet-32 manifest generator. |
| `submit_imagenet32_phasea.sh` | external | Submits ImageNet-32 Phase-A jobs. |
| `drl_resnet_energy.py` | external | DRL-style ResNet energy backbone. |
| `ebm_train_imagenet32_drl_sync.py` | external | DDP/pipeline ImageNet-32 DRL-style training entrypoint. |
| `ebm_train_imagenet32_drl_single.py` | external | Single-GPU ImageNet-32 DRL-style training entrypoint. |
| `eval_generate_imagenet32_drl.py` | external | DRL-specific sample generator wrapper for ImageNet-32 checkpoints. |
| `eval_imagenet32_drl_local.sh` | external | Local DRL sample generation plus rough FID wrapper. |
| `imagenet32_drl_manifest.py` | external | DRL-style ImageNet-32 manifest generator. |
| `imagenet32_drl_stability_manifest.py` | external | DRL-style stability diagnostic manifest generator. |
| `pbs_run_imagenet32_drl_case.pbs` | external | PBS train wrapper for one ImageNet-32 DRL-style case. |
| `setup_drl_env.sh` | external | One-command environment setup for DRL/ImageNet-32 on a new server. |
| `materialize_imagenet32_local.sh` | external | Local ImageNet-32 materialization wrapper using `.env.drl`. |
| `run_imagenet32_drl_smoke_local.sh` | external | Local `torchrun` smoke launcher for DRL DDP or pipeline. |
| `submit_imagenet32_drl.sh` | external | Submits DRL-style ImageNet-32 jobs. |
| `submit_imagenet32_drl_prime20k_successors.py` | external | Submits prime/scale successor diagnostics. |
| `watch_imagenet32_drl_prime20k.sh` | external | Watcher for DRL prime/scale diagnostics. |
| `launch_imagenet32_drl_local_phase1_two_node.sh` | external | Local/two-node DRL launcher. |
| `eval_imagenet32_fid.py` | external | ImageNet-32 FID evaluator. |
| `eval_imagenet32_conditional_acc.py` | external | ImageNet-32 conditional-accuracy evaluator. |
| `aggregate_imagenet32_results.py` | external | Aggregates ImageNet-32 results. |

## PBS, Submission, And Retained Helpers

| File | Status | Role |
| --- | --- | --- |
| `pbs_modea_fullk_align_singlefid.pbs` | retained | PBS wrapper for full-K alignment/single-FID check. |
| `pbs_pipeline_stage2b001_convergence_eval.pbs` | retained | PBS wrapper for older stage/beta convergence eval. |
| `pbs_repro_orig_single_fid.sh` | retained | Reproduction helper for original single FID. |
| `__init__.py` | support | Makes `scripts/current` importable as a package. |

If a script is not listed as `primary`, do not use it as the first entrypoint
unless the corresponding document or manifest explicitly requires it.
