# Results

This directory contains lightweight, collaborator-facing result tables. It is intentionally separate from raw run folders, scheduler logs, checkpoints, and generated image dumps.

## Layout

- `current_cifar/`: canonical CIFAR paper-result tables, including FID-vs-step trajectories and endpoint summaries.
- `long_k/`: original CIFAR long-K scaling summaries. These files are currently rolling placeholders until the long-K jobs finish and the collector is rerun.
- `batch_scaling/`: K400 batch-size scaling summaries. These files are currently rolling placeholders until the collector is rerun.
- `multinode_scaling/`: 16/32-GPU long-K multinode summaries. These files are currently rolling placeholders until the collector is rerun.

## Primary Tables

- `current_cifar/fid_vs_step.csv`: canonical 5k-spaced FID trajectory table for current CIFAR paper runs.
- `current_cifar/fid_vs_step_with_wallclock.csv`: same trajectory table with wall-clock and GPU-hour columns.
- `current_cifar/training_time_by_run.csv`: one row per logical run with total per-run training wall-clock and GPU-hours.
- `current_cifar/training_time_comparison_by_run.csv`: concise per-run comparison table with regime, GPU count, wall-clock, GPU-hours, final FID, and best FID.
- `current_cifar/training_time_regime_summary.csv`: grouped timing summary for quickly comparing single-GPU, DDP, and pipeline regimes.
- `current_cifar/training_time_totals.csv`: aggregate training duration totals by ledger, phase group, regime, and paper role.
- `current_cifar/seed_fid_trajectory_points.csv`: seed-level FID trajectory points.
- `current_cifar/ablation_final_summary.csv`: final/best summary for current CIFAR ablations.
- `current_cifar/ablation_500k_summary.csv`: 500k long-horizon summary.
- `current_cifar/pipeline_then_weighting_replay_summary.csv`: replay summary for the pipeline-then-weighting analysis.

## Training Time Semantics

Training-time totals are summed per logical run. They are not calendar makespan, because many jobs ran concurrently or through resume fragments. Use `train_gpu_hours` or `total_train_gpu_hours` when comparing compute usage across GPU counts.

## Update Policy

Future collectors should write final CSV outputs into the appropriate subdirectory here, using stable filenames:

- `fid_vs_step.csv`
- `fid_vs_step_with_wallclock.csv` when wall-clock/GPU-hour data is available
- `wallclock_summary.csv`
- `best_final_summary.csv`

Do not commit raw checkpoints, raw PBS logs, full run directories, or image dumps into `results/`.
