# Long-K Scaling Phase L0 Report

Scope: original CIFAR EBM backbone/dataset only. No DRL backbone. No ImageNet-32.

| exp_id | mode | K | P | step_size | submit | job_id | status |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| L0_ddp_fullk_K200_s20k_seed1 | ddp_fullk | 200 | 1 | 0.005 | yes | 7127689.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | submitted |
| L0_pipe_P8_equal_K200_s20k_seed1 | pipe_strict | 200 | 8 | 0.005 | yes | 7127690.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | submitted |
| L0_ddp_fullk_K400_s20k_seed1 | ddp_fullk | 400 | 1 | 0.0025 | yes | 7127691.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | submitted |
| L0_pipe_P8_equal_K400_s20k_seed1 | pipe_strict | 400 | 8 | 0.0025 | yes | 7127692.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | submitted |
| L0_single_fullk_K400_s20k_seed1 | single_fullk | 400 | 1 | 0.0025 | yes | 7127693.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | submitted |
| L0_single_emul_P8_equal_K400_s20k_seed1 | single_pipe_emul | 400 | 8 | 0.0025 | yes | 7127694.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov | submitted |

Success criteria: finite loss, no crash, at least one FID point after eval/backfill, expected max_abs_chain, wall-clock and GPU-hour logging.
