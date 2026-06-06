# Current CIFAR Walltime By Experiment Type

Source: each canonical run directory `job_info.json` for `train_walltime_seconds`; GPU count from `manifest_row.json`.
Grouping unit: `phase_group + family_id + horizon_steps + K + train_mode + gpu_count`. Registry spelling variants such as `pipeline` vs `strict_pipeline` are merged when the family/mode/GPU shape match.

- logical runs read: 62
- logical runs with walltime: 62
- missing walltime: 0
- grouped experiment types: 36

| phase | family | horizon | K | mode | GPUs | seeds | mean h | std h | min-max h | mean GPU-h | h / 100k |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| O1 | single_P16_deepest | 300000 | 100 | single_pipe_emul | 1 | 1 (1) | 10.54 | 0.00 | 10.54-10.54 | 10.54 | 3.51 |
| O1 | single_P16_equal | 300000 | 100 | single_pipe_emul | 1 | 1 (1) | 10.52 | 0.00 | 10.52-10.52 | 10.52 | 3.51 |
| O1 | single_P1_terminal | 300000 | 100 | single_fullk | 1 | 1 (1) | 8.31 | 0.00 | 8.31-8.31 | 8.31 | 2.77 |
| O1 | single_P2_deepest | 300000 | 100 | single_pipe_emul | 1 | 1 (1) | 8.46 | 0.00 | 8.46-8.46 | 8.46 | 2.82 |
| O1 | single_P2_equal | 300000 | 100 | single_pipe_emul | 1 | 1 (1) | 8.49 | 0.00 | 8.49-8.49 | 8.49 | 2.83 |
| O1 | single_P4_deepest | 300000 | 100 | single_pipe_emul | 1 | 1 (1) | 8.75 | 0.00 | 8.75-8.75 | 8.75 | 2.92 |
| O1 | single_P4_equal | 300000 | 100 | single_pipe_emul | 1 | 1 (1) | 8.76 | 0.00 | 8.76-8.76 | 8.76 | 2.92 |
| O1 | single_P8_deepest | 300000 | 100 | single_pipe_emul | 1 | 1 (1) | 9.34 | 0.00 | 9.34-9.34 | 9.34 | 3.11 |
| O1 | single_P8_equal | 300000 | 100 | single_pipe_emul | 1 | 1 (1) | 9.11 | 0.00 | 9.11-9.11 | 9.11 | 3.04 |
| O2 | ddp_P1_terminal | 300000 | 100 | ddp_fullk | 4 | 1 (1) | 8.39 | 0.00 | 8.39-8.39 | 33.56 | 2.80 |
| O2 | pipe_P16_deepest | 300000 | 100 | pipe_strict | 16 | 1 (1) | 23.53 | 0.00 | 23.53-23.53 | 376.54 | 7.84 |
| O2 | pipe_P16_equal | 300000 | 100 | pipe_strict | 16 | 1 (1) | 22.73 | 0.00 | 22.73-22.73 | 363.72 | 7.58 |
| O2 | pipe_P2_deepest | 300000 | 100 | pipe_strict | 2 | 1 (1) | 4.95 | 0.00 | 4.95-4.95 | 9.90 | 1.65 |
| O2 | pipe_P2_equal | 300000 | 100 | pipe_strict | 2 | 1 (1) | 4.92 | 0.00 | 4.92-4.92 | 9.84 | 1.64 |
| O2 | pipe_P4_deepest | 300000 | 100 | pipe_strict | 4 | 1 (1) | 3.59 | 0.00 | 3.59-3.59 | 14.37 | 1.20 |
| O2 | pipe_P4_equal | 300000 | 100 | pipe_strict | 4 | 1 (1) | 3.55 | 0.00 | 3.55-3.55 | 14.21 | 1.18 |
| O2 | pipe_P8_deepest | 300000 | 100 | pipe_strict | 8 | 1 (1) | 19.69 | 0.00 | 19.69-19.69 | 157.50 | 6.56 |
| O2 | pipe_P8_equal | 300000 | 100 | pipe_strict | 8 | 1 (1) | 3.20 | 0.00 | 3.20-3.20 | 25.61 | 1.07 |
| O3 | ddp_P1_terminal | 300000 | 100 | ddp_fullk | 4 | 4 (2 3 4 5) | 8.38 | 0.02 | 8.36-8.40 | 33.51 | 2.79 |
| O3 | pipe_P2_equal | 300000 | 100 | pipe_strict | 2 | 2 (2 3) | 4.89 | 0.00 | 4.89-4.89 | 9.77 | 1.63 |
| O3 | pipe_P4_deepest | 300000 | 100 | pipe_strict | 4 | 2 (2 3) | 3.59 | 0.01 | 3.58-3.60 | 14.36 | 1.20 |
| O3 | pipe_P4_equal | 300000 | 100 | pipe_strict | 4 | 4 (2 3 4 5) | 3.33 | 0.57 | 2.48-3.63 | 13.32 | 1.11 |
| O3 | single_P1_terminal | 300000 | 100 | single_fullk | 1 | 2 (2 3) | 6.89 | 2.05 | 5.43-8.34 | 6.89 | 2.30 |
| O3 | single_P2_deepest | 300000 | 100 | single_pipe_emul | 1 | 2 (2 3) | 8.48 | 0.05 | 8.44-8.52 | 8.48 | 2.83 |
| O3 | single_P2_equal | 300000 | 100 | single_pipe_emul | 1 | 2 (2 3) | 8.52 | 0.00 | 8.52-8.52 | 8.52 | 2.84 |
| O4 | ddp_P1_terminal | 500000 | 100 | ddp_fullk | 4 | 3 (1 2 3) | 13.17 | 1.30 | 11.66-13.94 | 52.66 | 2.63 |
| O4 | pipe_P4_deepest | 500000 | 100 | pipe_strict | 4 | 3 (1 2 3) | 5.97 | 0.02 | 5.95-5.99 | 23.88 | 1.19 |
| O4 | pipe_P4_equal | 500000 | 100 | pipe_strict | 4 | 5 (1 2 3 4 5) | 5.96 | 0.03 | 5.91-5.99 | 23.86 | 1.19 |
| O4 | single_P1_terminal | 500000 | 100 | single_fullk | 1 | 3 (1 2 3) | 13.87 | 0.04 | 13.83-13.89 | 13.87 | 2.77 |
| O4 | single_P2_deepest | 500000 | 100 | single_pipe_emul | 1 | 3 (1 2 3) | 13.91 | 0.29 | 13.58-14.08 | 13.91 | 2.78 |
| O4 | single_P2_equal | 500000 | 100 | single_pipe_emul | 1 | 3 (1 2 3) | 14.13 | 0.05 | 14.09-14.18 | 14.13 | 2.83 |
| O5 | ddp_P1_terminal_K100 | 300000 | 100 | ddp_fullk | 4 | 1 (1) | 8.40 | 0.00 | 8.40-8.40 | 33.62 | 2.80 |
| O5 | ddp_P1_terminal_K25 | 300000 | 25 | ddp_fullk | 4 | 1 (1) | 3.55 | 0.00 | 3.55-3.55 | 14.20 | 1.18 |
| O5 | ddp_P1_terminal_K50 | 300000 | 50 | ddp_fullk | 4 | 1 (1) | 5.18 | 0.00 | 5.18-5.18 | 20.73 | 1.73 |
| O5 | pipe_P4_deepest_K100 | 300000 | 100 | pipe_strict | 4 | 1 (1) | 2.97 | 0.00 | 2.97-2.97 | 11.89 | 0.99 |
| O5 | pipe_P4_equal_K100 | 300000 | 100 | pipe_strict | 4 | 2 (1 2) | 3.58 | 0.03 | 3.55-3.60 | 14.31 | 1.19 |
