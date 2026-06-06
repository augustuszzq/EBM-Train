# Phase 1 Ablation Summary

## Baseline Alignment
| exp_id | train_mode | ws | P | K | eps | weight | FID | feature-FID | unique | walltime | status |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| A0_single_strict_300k_ref | single_fullk | 1 | 1 | 100 | 1.0 | uniform | 61.651 | 0.542 | 1.000 | 07:39:42 | reference |
| A0_ddp_strict_300k_ref | ddp_fullk | 4 | 0 | 100 | 1.0 | uniform | 66.857 | 0.436 | 1.000 | 08:26:34 | reference |
| A0_pipeline_strict_300k_ref | pipe_strict | 4 | 4 | 100 | 1.0 | last2_beta | 55.995 | 0.450 | 1.000 | 03:38:59 | reference |
| A1_single_fullk_k100_s20k | single_fullk | 1 | 1 | 100 | 1.0 | uniform | 97.456 | 0.322 | 1.000 | 00:30:55 | done |
| A3_ddp_fullk_k100_s20k | ddp_fullk | 4 | 0 | 100 | 1.0 | uniform | 118.981 | 0.530 | 1.000 | 00:34:02 | done |
| A3_pipe_last2b001_p4_k100_s20k | pipe_strict | 4 | 4 | 100 | 1.0 | last2_beta | 123.036 | 1.490 | 1.000 | 00:14:46 | done |

## Weighted-Sum On Single GPU
| exp_id | train_mode | ws | P | K | eps | weight | FID | feature-FID | unique | walltime | status |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| A1_single_fullk_k100_s20k | single_fullk | 1 | 1 | 100 | 1.0 | uniform | 97.456 | 0.322 | 1.000 | 00:30:55 | done |
| A1_single_pipe_uniform_p4_k100_s20k | single_pipe_emul | 1 | 4 | 100 | 1.0 | uniform | 90.515 | 1.882 | 1.000 | 00:31:47 | done |
| A1_single_pipe_deep_p4_k100_s20k | single_pipe_emul | 1 | 4 | 100 | 1.0 | deep_only | 124.038 | 5.540 | 1.000 | 00:32:01 | done |
| A1_single_pipe_last2b001_p4_k100_s20k | single_pipe_emul | 1 | 4 | 100 | 1.0 | last2_beta | 132.846 | 2.735 | 1.000 | 00:31:58 | done |

## Pipeline Stage-Count
| exp_id | train_mode | ws | P | K | eps | weight | FID | feature-FID | unique | walltime | status |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| A2_pipe_uniform_p2_k100_s20k | pipe_strict | 2 | 2 | 100 | 1.0 | uniform | 108.216 | 4.531 | 1.000 | 00:20:00 | done |
| A2_pipe_uniform_p4_k100_s20k | pipe_strict | 4 | 4 | 100 | 1.0 | uniform | 88.167 | 4.967 | 1.000 | 00:14:38 | done |
| A2_pipe_uniform_p8_k100_s20k | pipe_strict | 8 | 8 | 100 | 1.0 | uniform | 99.061 | 1.948 | 1.000 | 01:19:50 | done |
| A2_pipe_last2b001_p2_k100_s20k | pipe_strict | 2 | 2 | 100 | 1.0 | last2_beta | 114.222 | 1.840 | 1.000 | 00:19:59 | done |
| A2_pipe_last2b001_p4_k100_s20k | pipe_strict | 4 | 4 | 100 | 1.0 | last2_beta | 106.223 | 2.695 | 1.000 | 00:15:02 | done |
| A2_pipe_last2b001_p8_k100_s20k | pipe_strict | 8 | 8 | 100 | 1.0 | last2_beta | 127.719 | 3.242 | 1.000 | 01:19:03 | done |

## K Sweep
| exp_id | train_mode | ws | P | K | eps | weight | FID | feature-FID | unique | walltime | status |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| A3_ddp_fullk_k30_s20k | ddp_fullk | 4 | 0 | 30 | 1.0 | uniform | 372.824 | 16.663 | 1.000 | 00:15:41 | done |
| A3_pipe_last2b001_p4_k30_s20k | pipe_strict | 4 | 4 | 30 | 1.0 | last2_beta | 167.250 | 13.756 | 0.996 | 00:09:58 | done |
| A3_ddp_fullk_k60_s20k | ddp_fullk | 4 | 0 | 60 | 1.0 | uniform | 143.821 | 8.907 | 0.999 | 00:23:45 | done |
| A3_pipe_last2b001_p4_k60_s20k | pipe_strict | 4 | 4 | 60 | 1.0 | last2_beta | 124.302 | 12.206 | 1.000 | 00:12:01 | done |
| A3_ddp_fullk_k100_s20k | ddp_fullk | 4 | 0 | 100 | 1.0 | uniform | 118.981 | 0.530 | 1.000 | 00:34:02 | done |
| A3_pipe_last2b001_p4_k100_s20k | pipe_strict | 4 | 4 | 100 | 1.0 | last2_beta | 123.036 | 1.490 | 1.000 | 00:14:46 | done |
| A3_ddp_fullk_k150_s20k | ddp_fullk | 4 | 0 | 150 | 1.0 | uniform | 334.974 | 3.229 | 1.000 | 00:47:28 | done |
| A3_pipe_last2b001_p4_k150_s20k | pipe_strict | 4 | 4 | 150 | 1.0 | last2_beta | 327.738 | 4.245 | 1.000 | 00:18:03 | done |
| A3_ddp_fullk_k200_s20k | ddp_fullk | 4 | 0 | 200 | 1.0 | uniform | 371.295 | 5.117 | 1.000 | 01:00:45 | done |
| A3_pipe_last2b001_p4_k200_s20k | pipe_strict | 4 | 4 | 200 | 1.0 | last2_beta | 370.405 | 5.399 | 1.000 | 00:21:21 | done |
| A3_ddp_fullk_k400_s20k | ddp_fullk | 4 | 0 | 400 | 1.0 | uniform | 433.114 | 7.025 | 1.000 | 01:51:21 | done |
| A3_pipe_last2b001_p4_k400_s20k | pipe_strict | 4 | 4 | 400 | 1.0 | last2_beta | 421.738 | 7.142 | 1.000 | 00:34:43 | done |

## Step-Size Sweep
| exp_id | train_mode | ws | P | K | eps | weight | FID | feature-FID | unique | walltime | status |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| A4_ddp_fullk_eps001_k100_s20k | ddp_fullk | 4 | 0 | 100 | 0.01 | uniform | 118.942 | 0.948 | 1.000 | 00:34:35 | done |
| A4_pipe_last2b001_p4_eps001_k100_s20k | pipe_strict | 4 | 4 | 100 | 0.01 | last2_beta | 125.185 | 1.600 | 1.000 | 00:15:13 | done |
| A4_ddp_fullk_eps002_k100_s20k | ddp_fullk | 4 | 0 | 100 | 0.02 | uniform | 111.660 | 0.585 | 1.000 | 00:34:36 | done |
| A4_pipe_last2b001_p4_eps002_k100_s20k | pipe_strict | 4 | 4 | 100 | 0.02 | last2_beta | 115.645 | 5.018 | 1.000 | 00:15:15 | done |
| A4_ddp_fullk_eps005_k100_s20k | ddp_fullk | 4 | 0 | 100 | 0.05 | uniform | 113.463 | 0.488 | 1.000 | 00:34:34 | done |
| A4_pipe_last2b001_p4_eps005_k100_s20k | pipe_strict | 4 | 4 | 100 | 0.05 | last2_beta | 115.558 | 3.418 | 1.000 | 00:15:06 | done |
| A4_ddp_fullk_eps010_k100_s20k | ddp_fullk | 4 | 0 | 100 | 0.1 | uniform | 114.957 | 0.496 | 1.000 | 00:34:27 | done |
| A4_pipe_last2b001_p4_eps010_k100_s20k | pipe_strict | 4 | 4 | 100 | 0.1 | last2_beta | 112.574 | 1.706 | 1.000 | 00:15:07 | done |
| A4_ddp_fullk_eps020_k100_s20k | ddp_fullk | 4 | 0 | 100 | 0.2 | uniform | 117.380 | 0.419 | 1.000 | 00:34:33 | done |
| A4_pipe_last2b001_p4_eps020_k100_s20k | pipe_strict | 4 | 4 | 100 | 0.2 | last2_beta | 149.042 | 2.813 | 1.000 | 00:15:11 | done |
