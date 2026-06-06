# ImageNet-32 Phase A Plan

## Goal

Run the first real `preemptable` ImageNet-32 conditional validation package after local
real-data smoke succeeded for:

- `single strict`
- `DDP strict`
- `strict pipeline`

The Phase A package is intended to validate:

- end-to-end conditional training stability
- checkpoint writing at short horizons
- ImageNet-32 data materialization and distributed restart-safety
- compatibility of the shared conditional chain-state contract with all three execution modes

## Dataset preparation

The training runs depend on a single `preemptable` data-prep job:

- `scripts/current/pbs_materialize_imagenet32.pbs`

That job materializes:

- `train`
- `val`

from `ChocolateDave/imagenet-32` into:

- `/eagle/lc-mpi/Zhiqing/polaris_ebm/data/imagenet32`

The export is restart-safe because each split writes `_SUCCESS.json` and subsequent runs
reuse the cached result.

## Training matrix

Phase A submits 9 train jobs:

- horizons: `1k`, `5k`, `20k`
- modes:
  - `single_fullk`
  - `ddp_fullk`
  - `pipe_strict (P=4, K=100, beta=0.01)`

All rows use:

- benchmark: `imagenet32`
- queue: `preemptable`
- `K=100`
- `lr=1e-4`
- `save_every=1000`
- `vis_every=1000`
- `data_dir=/eagle/lc-mpi/Zhiqing/polaris_ebm/data/imagenet32`

## Submission flow

Submission is handled by:

- `scripts/current/submit_imagenet32_phasea.sh`

The submitter:

1. writes `experiments/imagenet32_phasea_manifest.csv`
2. submits the ImageNet-32 materialization job
3. submits 9 dependent train jobs through `pbs_run_ablation_case.pbs`
4. records job ids in `experiments/imagenet32_phasea_submitted.csv`

## Current status

Before Phase A submission:

- helper/config/conditional replay infrastructure is in place
- local real-data smoke passed for single / DDP / pipeline
- real-data eval smoke wrote both `conditional_fid.json` and `conditional_acc.json`

The next checkpoint after this document is the first real `preemptable` Phase A submission.
