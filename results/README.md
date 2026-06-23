# Results

This directory contains lightweight, collaborator-facing result tables. It is intentionally separate from raw run folders, scheduler logs, checkpoints, and generated image dumps.

## Layout

- `current_cifar/`: canonical CIFAR paper-result tables, including FID-vs-step trajectories and endpoint summaries.
- `current_cifar/paper_ready/`: paper-facing current-CIFAR tables and SVG figures generated from the canonical current-CIFAR CSVs.
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

## Current CIFAR Paper-Ready Assets

Use these for paper drafting before reaching back into raw run directories:

- `current_cifar/paper_ready/main_500k_family_summary.csv`: 500k family-level mean/std/median table grouped by `family_id`.
- `current_cifar/paper_ready/main_300k_o3_summary.csv`: 300k O3 seed top-up table.
- `current_cifar/paper_ready/shallow_chain_o5_summary.csv`: O5 shallow-chain control table.
- `current_cifar/paper_ready/trajectory_coverage.csv`: complete 5k-grid audit for all 62 current-CIFAR logical runs.
- `current_cifar/paper_ready/fid_vs_step_main_500k.svg`: seed-level and mean 500k trajectory figure.
- `current_cifar/paper_ready/pipeline_equal_vs_deepest_500k.svg`: mechanism ablation figure.
- `current_cifar/paper_ready/shallow_chain_control_best_fid.svg`: shallow-chain control figure.

Regenerate them with:

```bash
python3.11 scripts/current/build_current_cifar_paper_assets.py
```

The paper-ready summaries intentionally group quality results by `family_id`.
This prevents the same canonical method family from being split by raw runtime
labels such as `pipeline` versus `strict_pipeline`.

## Training Time Semantics

Training-time totals are summed per logical run. They are not calendar makespan, because many jobs ran concurrently or through resume fragments. Use `train_gpu_hours` or `total_train_gpu_hours` when comparing compute usage across GPU counts.

## Current CIFAR Type-Average Training Time

These are actual per-logical-run training wall-clock averages from `current_cifar/training_time_regime_summary.csv`.

| horizon | role | regime | P | weighting | K | runs | mean wall-clock hours |
|---:|---|---|---:|---|---:|---:|---:|
| 300k | control | ddp_strict | 1 | terminal | 25 | 1 | 3.55 |
| 300k | control | ddp_strict | 1 | terminal | 50 | 1 | 5.18 |
| 300k | control | ddp_strict | 1 | terminal | 100 | 2 | 8.40 |
| 300k | headline | ddp_strict | 1 | terminal | 100 | 4 | 8.38 |
| 300k | appendix | pipeline | 4 | deepest | 100 | 2 | 3.59 |
| 300k | control | pipeline | 4 | equal | 100 | 1 | 3.55 |
| 300k | control | single_emulation | 2 | deepest | 100 | 1 | 8.46 |
| 300k | headline | single_emulation | 2 | deepest | 100 | 2 | 8.48 |
| 300k | control | single_emulation | 2 | equal | 100 | 1 | 8.49 |
| 300k | headline | single_emulation | 2 | equal | 100 | 2 | 8.52 |
| 300k | control | single_emulation | 4 | deepest | 100 | 1 | 8.75 |
| 300k | control | single_emulation | 4 | equal | 100 | 1 | 8.76 |
| 300k | appendix | single_emulation | 8 | deepest | 100 | 1 | 9.34 |
| 300k | appendix | single_emulation | 8 | equal | 100 | 1 | 9.11 |
| 300k | appendix | single_emulation | 16 | deepest | 100 | 1 | 10.54 |
| 300k | appendix | single_emulation | 16 | equal | 100 | 1 | 10.52 |
| 300k | control | single_strict | 1 | terminal | 100 | 1 | 8.31 |
| 300k | headline | single_strict | 1 | terminal | 100 | 2 | 6.89 |
| 300k | control | strict_pipeline | 2 | deepest | 100 | 1 | 4.95 |
| 300k | control | strict_pipeline | 2 | equal | 100 | 3 | 4.90 |
| 300k | control | strict_pipeline | 4 | deepest | 100 | 2 | 3.28 |
| 300k | control | strict_pipeline | 4 | equal | 100 | 2 | 3.58 |
| 300k | headline | strict_pipeline | 4 | equal | 100 | 4 | 3.33 |
| 300k | appendix | strict_pipeline | 8 | deepest | 100 | 1 | 19.69 |
| 300k | appendix | strict_pipeline | 8 | equal | 100 | 1 | 3.20 |
| 300k | appendix | strict_pipeline | 16 | deepest | 100 | 1 | 23.53 |
| 300k | appendix | strict_pipeline | 16 | equal | 100 | 1 | 22.73 |
| 500k | headline | ddp_strict | 1 | terminal | 100 | 3 | 13.17 |
| 500k | headline | pipeline | 4 | equal | 100 | 2 | 5.94 |
| 500k | headline | single_emulation | 2 | deepest | 100 | 3 | 13.91 |
| 500k | headline | single_emulation | 2 | equal | 100 | 3 | 14.13 |
| 500k | headline | single_strict | 1 | terminal | 100 | 3 | 13.87 |
| 500k | control | strict_pipeline | 4 | deepest | 100 | 3 | 5.97 |
| 500k | headline | strict_pipeline | 4 | equal | 100 | 3 | 5.98 |

## Long-K M1 300k Validation Time

These are long-K validation wall-clock estimates for the 300k target. Completed runs are still shown using the same `300k * recent median iter_ms` estimate so the table has one consistent logical-run timing convention. The `latest step` column records the current observed progress at the time this table was prepared.

| run | latest step | progress | median iter ms | wall-clock hours |
|---|---:|---:|---:|---:|
| M1_S16_ddp_fullk_K400_b512_s20k_seed1_to300k | 290362 | 96.8% | 1672.4 | 139.36 |
| M1_S16_ddp_fullk_K800_b512_s20k_seed1_to300k | 130196 | 43.4% | 2958.7 | 246.56 |
| M1_S16_ddp_fullk_K1600_b512_s20k_seed1_to300k | 66817 | 22.3% | 5624.2 | 468.69 |
| M1_S16_pipe_P16_equal_K400_b512_s20k_seed1_to300k | 299999 | 100.0% | 419.6 | 34.97 |
| M1_S16_pipe_P16_equal_K800_b512_s20k_seed1_to300k | 299999 | 100.0% | 528.1 | 44.01 |
| M1_S16_pipe_P16_equal_K1600_b512_s20k_seed1_to300k | 50667 | 16.9% | 681.9 | 56.82 |
| M1_S16_single_fullk_K400_b512_s20k_seed1_to300k | 265021 | 88.3% | 2039.0 | 169.92 |
| M1_S16_single_fullk_K800_b512_s20k_seed1_to300k | 110302 | 36.8% | 4048.3 | 337.36 |
| M1_S16_single_fullk_K1600_b512_s20k_seed1_to300k | 61383 | 20.5% | 7949.5 | 662.46 |
| M1_S32_ddp_fullk_K400_b1024_s20k_seed1_to300k | 75820 | 25.3% | 1741.5 | 145.12 |
| M1_S32_ddp_fullk_K800_b1024_s20k_seed1_to300k | 20144 | 6.7% | 3092.9 | 257.74 |
| M1_S32_ddp_fullk_K1600_b1024_s20k_seed1_to300k | 30113 | 10.0% | 5772.0 | 481.00 |
| M1_S32_pipe_P32_equal_K400_b1024_s20k_seed1_to300k | 100200 | 33.4% | 441.3 | 36.77 |
| M1_S32_pipe_P32_equal_K800_b1024_s20k_seed1_to300k | 299999 | 100.0% | 483.3 | 40.28 |
| M1_S32_pipe_P32_equal_K1600_b1024_s20k_seed1_to300k | 299999 | 100.0% | 582.6 | 48.55 |
| M1_S32_single_fullk_K400_b1024_s20k_seed1_to300k | 115089 | 38.4% | 3906.4 | 325.53 |
| M1_S32_single_fullk_K800_b1024_s20k_seed1_to300k | 65081 | 21.7% | 7693.9 | 641.16 |
| M1_S32_single_fullk_K1600_b1024_s20k_seed1_to300k | 25032 | 8.3% | 15333.8 | 1277.82 |

## Update Policy

Future collectors should write final CSV outputs into the appropriate subdirectory here, using stable filenames:

- `fid_vs_step.csv`
- `fid_vs_step_with_wallclock.csv` when wall-clock/GPU-hour data is available
- `wallclock_summary.csv`
- `best_final_summary.csv`

Do not commit raw checkpoints, raw PBS logs, full run directories, or image dumps into `results/`.
