# Phase 1 Local Eval Backlog

Snapshot date: 2026-03-29 UTC

This file records the Phase 1 screening cases that currently do not have `eval/metrics_compare.json`.
After the train queue drains, these cases should be evaluated locally with:

```bash
python3 /lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/run_local_eval_phase1.py \
  --submitted-manifest /lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/experiments/ablation_manifest_phase1_submitted.csv \
  --cuda-visible-devices 0
```

Current counts:

- Missing eval cases: `14`
- Already ready for local eval: `6`
- Still training or queued: `8`

## Ready For Local Eval

1. `A2_pipe_uniform_p8_k100_s20k`
   - Group: `A2`
   - Train mode: `pipe_strict`
   - K: `100`
   - Step size: `1.0`
   - Train status: `succeeded`
   - Run dir: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation/A2/A2_pipe_uniform_p8_k100_s20k_20260329_145326`
2. `A3_ddp_fullk_k200_s20k`
   - Group: `A3`
   - Train mode: `ddp_fullk`
   - K: `200`
   - Step size: `1.0`
   - Train status: `succeeded`
   - Run dir: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation/A3/A3_ddp_fullk_k200_s20k_20260329_152523`
3. `A3_pipe_last2b001_p4_k400_s20k`
   - Group: `A3`
   - Train mode: `pipe_strict`
   - K: `400`
   - Step size: `1.0`
   - Train status: `succeeded`
   - Run dir: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation/A3/A3_pipe_last2b001_p4_k400_s20k_20260329_160221`
4. `A4_ddp_fullk_eps001_k100_s20k`
   - Group: `A4`
   - Train mode: `ddp_fullk`
   - K: `100`
   - Step size: `0.01`
   - Train status: `succeeded`
   - Run dir: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation/A4/A4_ddp_fullk_eps001_k100_s20k_20260329_161307`
5. `A4_pipe_last2b001_p4_eps001_k100_s20k`
   - Group: `A4`
   - Train mode: `pipe_strict`
   - K: `100`
   - Step size: `0.01`
   - Train status: `succeeded`
   - Run dir: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation/A4/A4_pipe_last2b001_p4_eps001_k100_s20k_20260329_161307`
6. `A4_pipe_last2b001_p4_eps002_k100_s20k`
   - Group: `A4`
   - Train mode: `pipe_strict`
   - K: `100`
   - Step size: `0.02`
   - Train status: `succeeded`
   - Run dir: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation/A4/A4_pipe_last2b001_p4_eps002_k100_s20k_20260329_161307`

## Still Training Or Queued

1. `A3_ddp_fullk_k400_s20k`
   - Group: `A3`
   - Train mode: `ddp_fullk`
   - K: `400`
   - Step size: `1.0`
   - Train status: `running`
   - Run dir: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation/A3/A3_ddp_fullk_k400_s20k_20260329_152523`
2. `A4_ddp_fullk_eps002_k100_s20k`
   - Group: `A4`
   - Train mode: `ddp_fullk`
   - K: `100`
   - Step size: `0.02`
   - Train status: `running`
   - Run dir: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation/A4/A4_ddp_fullk_eps002_k100_s20k_20260329_161307`
3. `A4_ddp_fullk_eps005_k100_s20k`
   - Group: `A4`
   - Train mode: `ddp_fullk`
   - K: `100`
   - Step size: `0.05`
   - Train status: `running`
   - Run dir: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation/A4/A4_ddp_fullk_eps005_k100_s20k_20260329_161307`
4. `A4_pipe_last2b001_p4_eps005_k100_s20k`
   - Group: `A4`
   - Train mode: `pipe_strict`
   - K: `100`
   - Step size: `0.05`
   - Train status: `running`
   - Run dir: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation/A4/A4_pipe_last2b001_p4_eps005_k100_s20k_20260329_162236`
5. `A4_ddp_fullk_eps010_k100_s20k`
   - Group: `A4`
   - Train mode: `ddp_fullk`
   - K: `100`
   - Step size: `0.1`
   - Train status: `running`
   - Run dir: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation/A4/A4_ddp_fullk_eps010_k100_s20k_20260329_162236`
6. `A4_pipe_last2b001_p4_eps010_k100_s20k`
   - Group: `A4`
   - Train mode: `pipe_strict`
   - K: `100`
   - Step size: `0.1`
   - Train status: `running`
   - Run dir: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation/A4/A4_pipe_last2b001_p4_eps010_k100_s20k_20260329_164242`
7. `A4_ddp_fullk_eps020_k100_s20k`
   - Group: `A4`
   - Train mode: `ddp_fullk`
   - K: `100`
   - Step size: `0.2`
   - Train status: `unknown`
   - Run dir: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation/A4/A4_ddp_fullk_eps020_k100_s20k_20260329_164242`
8. `A4_pipe_last2b001_p4_eps020_k100_s20k`
   - Group: `A4`
   - Train mode: `pipe_strict`
   - K: `100`
   - Step size: `0.2`
   - Train status: `unknown`
   - Run dir: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation/A4/A4_pipe_last2b001_p4_eps020_k100_s20k_20260329_164242`
