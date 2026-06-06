# Long-K Batch-Scaling Plan

Scope: original CIFAR model/dataset only; no DRL backbone and no ImageNet-32.

Motivation: the previous K400 pipeline used global batch 64 on 8 GPUs, giving local batch 8. This likely made the run communication/launch-overhead dominated. This bundle first tests global batch 256, giving local batch 32 on 8 GPUs.

| exp_id | mode | world | global batch | local batch | K | P | steps | submit | job_id |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| B0_single_fullk_K400_b256_s20k_seed1 | single_fullk | 1 | 256 | 256 | 400 | 1 | 20000 | yes | 7162478.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov |
| B0_single_emul_P8_equal_K400_b256_s20k_seed1 | single_pipe_emul | 1 | 256 | 256 | 400 | 8 | 20000 | yes | 7162479.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov |
| B0_ddp_fullk_K400_b256_s20k_seed1 | ddp_fullk | 8 | 256 | 32 | 400 | 1 | 20000 | yes | 7162480.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov |
| B0_pipe_P8_equal_K400_b256_s20k_seed1 | pipe_strict | 8 | 256 | 32 | 400 | 8 | 20000 | yes | 7162481.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov |
| B0opt_ddp_fullk_K400_b128_s20k_seed1 | ddp_fullk | 8 | 128 | 16 | 400 | 1 | 20000 | no |  |
| B0opt_pipe_P8_equal_K400_b128_s20k_seed1 | pipe_strict | 8 | 128 | 16 | 400 | 8 | 20000 | no |  |
| B0opt_ddp_fullk_K400_b512_s20k_seed1 | ddp_fullk | 8 | 512 | 64 | 400 | 1 | 20000 | no |  |
| B0opt_pipe_P8_equal_K400_b512_s20k_seed1 | pipe_strict | 8 | 512 | 64 | 400 | 8 | 20000 | no |  |

Gate for next phase: compare step time, throughput, finite loss, max_abs_chain, and at least one FID point. If b256 is stable and pipeline overhead improves, extend the same rows to 300k.
