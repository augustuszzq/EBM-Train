# Phase 2 300k Summary

## Anchors
| exp_id | train_mode | ws | P | K | eps | weight | FID | feature-FID | unique | walltime | status |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| single_fullk_K100_seed1_300k | single_fullk | 1 | 1 | 100 | 1.0 | uniform | 61.651 | 0.542 | 1.000 | 07:39:42 | reference |
| ddp_fullk_K100_seed1_300k | ddp_fullk | 4 | 0 | 100 | 1.0 | uniform | 66.857 | 0.436 | 1.000 | 08:26:34 | reference |
| pipe_strict_P4_K100_beta001_lr1e4_seed1_300k | pipe_strict | 4 | 4 | 100 | 1.0 | last2_beta | 55.995 | 0.450 | 1.000 | 03:38:59 | reference |

## Reproducibility Seed Sweep
| exp_id | train_mode | ws | P | K | eps | weight | FID | feature-FID | unique | walltime | status |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| ddp_fullk_K100_seed2_300k | ddp_fullk | 4 | 0 | 100 | 1.0 | uniform | 58.293 | 0.272 | 1.000 | 08:26:02 | done |
| pipe_strict_P4_K100_beta001_lr1e4_seed2_300k | pipe_strict | 4 | 4 | 100 | 1.0 | last2_beta | 66.047 | 0.687 | 1.000 | 03:35:51 | done |
| ddp_fullk_K100_seed3_300k | ddp_fullk | 4 | 0 | 100 | 1.0 | uniform | 65.133 | 1.184 | 1.000 | 08:24:00 | done |
| pipe_strict_P4_K100_beta001_lr1e4_seed3_300k | pipe_strict | 4 | 4 | 100 | 1.0 | last2_beta | 71.189 | 0.873 | 1.000 | 03:37:06 | done |

## High-K Shortcut Control
| exp_id | train_mode | ws | P | K | eps | weight | FID | feature-FID | unique | walltime | status |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| ddp_fullk_K400_seed1_300k | ddp_fullk | 4 | 0 | 400 | 1.0 | uniform | 436.258 | 6.912 | 1.000 | 27:29:30 | done |
| pipe_strict_P4_K400_beta001_lr1e4_seed1_300k | pipe_strict | 4 | 4 | 400 | 1.0 | last2_beta | 391.778 | 6.578 | 1.000 | 08:25:16 | done |

## Single-GPU Mechanism
| exp_id | train_mode | ws | P | K | eps | weight | FID | feature-FID | unique | walltime | status |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| single_pipe_emul_P4_K100_deep_seed1_300k | single_pipe_emul | 1 | 4 | 100 | 1.0 | deep_only | 60.619 | 0.800 | 1.000 | 07:56:40 | done |
| single_pipe_emul_P4_K100_beta001_seed1_300k | single_pipe_emul | 1 | 4 | 100 | 1.0 | last2_beta | 89.392 | 2.323 | 1.000 | 07:56:35 | done |

## Pipeline Stage-Count Extension
| exp_id | train_mode | ws | P | K | eps | weight | FID | feature-FID | unique | walltime | status |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| pipe_strict_P2_K100_beta001_lr1e4_seed1_300k | pipe_strict | 2 | 2 | 100 | 1.0 | last2_beta | 83.666 | 0.657 | 1.000 | 04:54:08 | done |
| pipe_strict_P8_K100_beta001_lr1e4_seed1_300k | pipe_strict | 8 | 8 | 100 | 1.0 | last2_beta | 76.615 | 2.014 | 1.000 | 19:22:51 | done |
