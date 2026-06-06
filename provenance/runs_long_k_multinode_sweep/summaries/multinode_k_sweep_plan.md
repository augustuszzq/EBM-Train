# Multi-Node Long-K Sweep

Scope: original CIFAR EBM only; no DRL backbone and no ImageNet-32.

Objective semantics are unchanged: pipeline weights are applied to negative energy scalars, never to image tensors.

This first phase is a 20k screen for K=400/800/1600 at 16-GPU and 32-GPU scale.

| exp_id | mode | scale | nodes | world | batch | local | K | P | step_size | walltime | job_id |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| S16_single_fullk_K400_b512_s20k_seed1 | single_fullk | S16 | 1 | 1 | 512 | 512 | 400 | 1 | 0.0025 | 36:00:00 | 7173318.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov |
| S16_ddp_fullk_K400_b512_s20k_seed1 | ddp_fullk | S16 | 4 | 16 | 512 | 32 | 400 | 1 | 0.0025 | 24:00:00 | 7173319.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov |
| S16_pipe_P16_equal_K400_b512_s20k_seed1 | pipe_strict | S16 | 4 | 16 | 512 | 32 | 400 | 16 | 0.0025 | 24:00:00 | 7173320.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov |
| S16_single_fullk_K800_b512_s20k_seed1 | single_fullk | S16 | 1 | 1 | 512 | 512 | 800 | 1 | 0.00125 | 36:00:00 | 7173321.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov |
| S16_ddp_fullk_K800_b512_s20k_seed1 | ddp_fullk | S16 | 4 | 16 | 512 | 32 | 800 | 1 | 0.00125 | 36:00:00 | 7173322.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov |
| S16_pipe_P16_equal_K800_b512_s20k_seed1 | pipe_strict | S16 | 4 | 16 | 512 | 32 | 800 | 16 | 0.00125 | 36:00:00 | 7173323.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov |
| S16_single_fullk_K1600_b512_s20k_seed1 | single_fullk | S16 | 1 | 1 | 512 | 512 | 1600 | 1 | 0.000625 | 36:00:00 | 7173324.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov |
| S16_ddp_fullk_K1600_b512_s20k_seed1 | ddp_fullk | S16 | 4 | 16 | 512 | 32 | 1600 | 1 | 0.000625 | 36:00:00 | 7173325.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov |
| S16_pipe_P16_equal_K1600_b512_s20k_seed1 | pipe_strict | S16 | 4 | 16 | 512 | 32 | 1600 | 16 | 0.000625 | 36:00:00 | 7173326.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov |
| S32_single_fullk_K400_b1024_s20k_seed1 | single_fullk | S32 | 1 | 1 | 1024 | 1024 | 400 | 1 | 0.0025 | 36:00:00 | 7173327.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov |
| S32_ddp_fullk_K400_b1024_s20k_seed1 | ddp_fullk | S32 | 8 | 32 | 1024 | 32 | 400 | 1 | 0.0025 | 24:00:00 | 7173328.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov |
| S32_pipe_P32_equal_K400_b1024_s20k_seed1 | pipe_strict | S32 | 8 | 32 | 1024 | 32 | 400 | 32 | 0.0025 | 24:00:00 | 7173329.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov |
| S32_single_fullk_K800_b1024_s20k_seed1 | single_fullk | S32 | 1 | 1 | 1024 | 1024 | 800 | 1 | 0.00125 | 36:00:00 | 7173330.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov |
| S32_ddp_fullk_K800_b1024_s20k_seed1 | ddp_fullk | S32 | 8 | 32 | 1024 | 32 | 800 | 1 | 0.00125 | 36:00:00 | 7173331.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov |
| S32_pipe_P32_equal_K800_b1024_s20k_seed1 | pipe_strict | S32 | 8 | 32 | 1024 | 32 | 800 | 32 | 0.00125 | 36:00:00 | 7173332.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov |
| S32_single_fullk_K1600_b1024_s20k_seed1 | single_fullk | S32 | 1 | 1 | 1024 | 1024 | 1600 | 1 | 0.000625 | 36:00:00 | 7173333.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov |
| S32_ddp_fullk_K1600_b1024_s20k_seed1 | ddp_fullk | S32 | 8 | 32 | 1024 | 32 | 1600 | 1 | 0.000625 | 36:00:00 | 7173334.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov |
| S32_pipe_P32_equal_K1600_b1024_s20k_seed1 | pipe_strict | S32 | 8 | 32 | 1024 | 32 | 1600 | 32 | 0.000625 | 36:00:00 | 7173335.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov |

Notes:
- S16 uses 4 nodes / 16 GPUs for DDP and pipeline; S32 uses 8 nodes / 32 GPUs.
- Single runs use the corresponding global batch on one GPU and may require resume for K=800/1600.
- Step size follows 0.01 * 100 / K.
