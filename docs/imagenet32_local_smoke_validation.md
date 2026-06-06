# ImageNet-32 Conditional Local Smoke Validation

Date: 2026-04-03 UTC

## Local synthetic benchmark fixture

- Materialized local smoke dataset:
  - `/eagle/lc-mpi/Zhiqing/polaris_ebm/tmp_imagenet32_smoke_data`
- Temporary smoke configs:
  - `/eagle/lc-mpi/Zhiqing/polaris_ebm/tmp_imagenet32_smoke_configs/imagenet32_single_smoke.yaml`
  - `/eagle/lc-mpi/Zhiqing/polaris_ebm/tmp_imagenet32_smoke_configs/imagenet32_ddp_smoke.yaml`
  - `/eagle/lc-mpi/Zhiqing/polaris_ebm/tmp_imagenet32_smoke_configs/imagenet32_pipeline_smoke.yaml`

The fixture uses a tiny materialized ImageNet-32-style layout with train/validation splits and `labels.csv`, enough to validate conditional `(x, y)` flow and checkpoint-compatible eval.

## Real HF-backed benchmark fixture

- Real Hugging Face source verified:
  - `ChocolateDave/imagenet-32`
- Tiny real-data materialization used for local validation:
  - `/eagle/lc-mpi/Zhiqing/polaris_ebm/data/imagenet32_real_smoke`
- Materialized counts:
  - `train = 256`
  - `val = 128`
- Important split fix:
  - the real dataset uses `train` / `val`
  - benchmark config defaults and loader aliasing were updated so `validation` is normalized to `val`

## Validated locally

### 1. single strict conditional smoke

- Train entrypoint:
  - [ebm_train_single_pipe_emul.py](/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/current/ebm_train_single_pipe_emul.py)
- Run dir:
  - [single_conditional_smoke](/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_imagenet32_smoke/single_conditional_smoke)
- Result:
  - 4-step train completed
  - checkpoints written at `step1` and `step3`

### 2. DDP strict conditional smoke

- Train entrypoint:
  - [ebm_train_sync_mode_a.py](/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/current/ebm_train_sync_mode_a.py)
- Run dir:
  - [ddp_conditional_smoke](/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_imagenet32_smoke/ddp_conditional_smoke)
- Result:
  - 2-GPU local `torchrun` completed
  - conditional DDP train loop stayed finite
  - rank-local checkpoint state files were written

### 3. strict pipeline conditional smoke

- Train entrypoint:
  - [ebm_train_sync_mode_a.py](/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/current/ebm_train_sync_mode_a.py)
- Run dir:
  - [pipeline_conditional_smoke](/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_imagenet32_smoke/pipeline_conditional_smoke)
- Result:
  - 2-GPU local pipeline completed
  - stage-to-stage conditional signature checks did not fire
  - checkpoints written at `step1` and `step3`

### 4. pipeline conditional resume smoke

- Resume run dir:
  - [pipeline_conditional_smoke_resume](/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_imagenet32_smoke/pipeline_conditional_smoke_resume)
- Resume source:
  - [ckpt_step1.pt](/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_imagenet32_smoke/pipeline_conditional_smoke/checkpoints/ckpt_step1.pt)
- Result:
  - resumed from `start_step=2`
  - completed to `step3`
  - rank-local resume state path was used

## Eval path validated locally

### single conditional eval

- Eval dir:
  - [single eval_smoke](/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_imagenet32_smoke/single_conditional_smoke/eval_smoke)
- Outputs present:
  - `grid.png`
  - `samples.pt`
  - `stats.json`
  - `conditional_fid.json`
  - `conditional_acc.json`

### pipeline conditional eval

- Eval dir:
  - [pipeline eval_smoke](/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_imagenet32_smoke/pipeline_conditional_smoke/eval_smoke)
- Outputs present:
  - `grid.png`
  - `samples.pt`
  - `stats.json`
  - `conditional_fid.json`
  - `conditional_acc.json`

## Current conclusion

The repo now has a locally validated conditional smoke slice for:

- single strict
- DDP strict
- strict pipeline
- checkpoint-compatible eval
- pipeline resume

This is sufficient to move the next debugging stage to real ImageNet-32 materialized data and then to `preemptable` Phase A runs.

## Real-data smoke validated locally

### 5. single strict conditional smoke on real ImageNet-32 slice

- Run dir:
  - [single_conditional_real_smoke](/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_imagenet32_smoke_real/single_conditional_real_smoke)
- Result:
  - 20-step local train completed on the real materialized slice
  - checkpoints written at `step9` and `step19`

### 6. DDP strict conditional smoke on real ImageNet-32 slice

- Run dir:
  - [ddp_conditional_real_smoke](/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_imagenet32_smoke_real/ddp_conditional_real_smoke)
- Result:
  - 2-GPU local `torchrun` completed on real ImageNet-32 data
  - conditional replay/DDP path stayed finite
  - rank-local checkpoint state files were written

### 7. strict pipeline conditional smoke on real ImageNet-32 slice

- Run dir:
  - [pipeline_conditional_real_smoke](/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_imagenet32_smoke_real/pipeline_conditional_real_smoke)
- Result:
  - 2-GPU local pipeline completed on real ImageNet-32 data
  - stage-to-stage conditional signature checks did not fire
  - checkpoints written at `step9` and `step19`

### 8. real-data eval path validated locally

- Eval dir:
  - [pipeline eval_real_smoke](/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_imagenet32_smoke_real/pipeline_conditional_real_smoke/eval_real_smoke)
- Outputs present:
  - `grid.png`
  - `samples.pt`
  - `stats.json`
  - `conditional_fid.json`
  - `conditional_acc.json`
- Smoke metrics:
  - `final_fid = 539.98`
  - `top1_conditional_acc = 0.0`

These smoke metrics are not quality targets. They only confirm that the real-data conditional generation/eval bookkeeping works end to end on materialized ImageNet-32 samples.
