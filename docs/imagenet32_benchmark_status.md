# ImageNet-32 Conditional Benchmark Status

## Implemented

- Shared conditional `ChainState` contract in [conditional_chain.py](/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/current/conditional_chain.py).
- Benchmark-aware config loading and pipeline contract checks in [benchmark_config.py](/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/current/benchmark_config.py).
- Label-bound replay semantics in [conditional_replay.py](/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/current/conditional_replay.py).
- Conditional energy model and sampler in [conditional_model.py](/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/current/conditional_model.py) and [conditional_sampler.py](/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/current/conditional_sampler.py).
- ImageNet-32 metadata, HF materialization scaffold, and local dataset loader in [imagenet32_data.py](/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/current/imagenet32_data.py).
- Benchmark runtime helpers wired into `eval_generate`, `eval_metrics`, `ebm_train_sync_mode_a`, and `ebm_train_single_pipe_emul`.
- Conditional local state checkpoint helpers wired into [ebm_train_sync_mode_a.py](/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/current/ebm_train_sync_mode_a.py) and [ebm_train_single_pipe_emul.py](/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/current/ebm_train_single_pipe_emul.py).
- Conditional replay is now active for:
  - non-pipeline conditional runs in `ebm_train_sync_mode_a`
  - stage-0 chain restarts in conditional pipeline mode
  - single-process `single_pipe_emul` stage-0 replay path
- Strict pipeline conditional transport now sends a lightweight state signature and asserts that received `(y, chain_id, steps_done, valid)` metadata matches that signature.
- Single-sample FID eval entrypoint in [eval_imagenet32_fid.py](/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/current/eval_imagenet32_fid.py).
- Conditional-faithfulness eval entrypoint in [eval_imagenet32_conditional_acc.py](/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/current/eval_imagenet32_conditional_acc.py).
- Root wrappers for ImageNet-32 configs and eval scripts in [scripts](/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts).

## Verified

- Config files load and validate.
- Conditional chain/replay/model/data helpers pass unit tests.
- Hugging Face access to `ChocolateDave/imagenet-32` works in `llm-env`.
- The real dataset split naming is now aligned with HF metadata:
  - `train`
  - `val`
- `imagenet32_data.py` now normalizes `validation`/`valid` aliases to `val`.
- Tiny real-data materialization is now supported for smoke validation via `max_examples`.
- `eval_generate` writes requested labels into `samples.pt` for conditional benchmarks.
- `eval_metrics` can load real samples from a materialized ImageNet-32 validation split.
- The new ImageNet-32 FID and conditional-accuracy scripts pass unit tests.
- Rank-local conditional resume state can override the main checkpoint payload for distributed resumes.
- Chain-state helper payloads can round-trip `(x, y, chain_id, steps_done, valid)`.
- Pipeline conditional transport tags/signatures are unit-tested.
- Local real-data smoke runs now pass for:
  - single strict
  - DDP strict
  - strict pipeline
- A real-data pipeline smoke checkpoint has completed an end-to-end local eval that wrote:
  - `grid.png`
  - `samples.pt`
  - `stats.json`
  - `conditional_fid.json`
  - `conditional_acc.json`
- The first real `preemptable` Phase A package has now been submitted:
  - materialize job: `6986198`
  - train jobs:
    - `6986199` `imagenet32_single_strict_k100_s1k`
    - `6986200` `imagenet32_ddp_strict_k100_s1k`
    - `6986201` `imagenet32_pipeline_strict_p4_k100_s1k`
    - `6986202` `imagenet32_single_strict_k100_s5k`
    - `6986203` `imagenet32_ddp_strict_k100_s5k`
    - `6986204` `imagenet32_pipeline_strict_p4_k100_s5k`
    - `6986205` `imagenet32_single_strict_k100_s20k`
    - `6986206` `imagenet32_ddp_strict_k100_s20k`
    - `6986207` `imagenet32_pipeline_strict_p4_k100_s20k`
- Current queue state at submission time (`2026-04-03T02:16:05Z`):
  - materialize job in `Q`
  - all 9 train jobs in `H` waiting on the materialize dependency

## Still Pending Before Phase A Smoke Runs

- Implement a real ImageNet-32 conditional accuracy protocol choice beyond the generic torchvision-classifier fallback.
- Extend aggregation to produce step/wall-clock/GPU-hours plots and bootstrap summaries.
- Wait for the full `train` / `val` materialization job to complete and release the 9 dependent Phase A jobs.
- Run the submitted real ImageNet-32 Phase A ladder to completion on:
  - `1k`
  - `5k`
  - `20k`

## Current Interpretation

The repo is no longer CIFAR-only at the helper layer, and the evaluation path now has concrete ImageNet-32 conditional entrypoints. The training path preserves local conditional state across checkpoint/resume, carries `(x, y, meta)` across pipeline stages, and now survives real-data local smoke for single / DDP / pipeline. The project has crossed the threshold from helper-layer bring-up into real queued experimentation: the first full ImageNet-32 materialization job and dependent `1k -> 5k -> 20k` Phase A smoke runs are now in `preemptable`.
