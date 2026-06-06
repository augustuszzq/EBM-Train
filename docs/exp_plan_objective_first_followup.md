# Objective-First Follow-Up Experiment Plan

This document turns the current CIFAR evidence into a new experiment package built around the updated paper logic:

1. define the completion-aware weighted energy objective first
2. validate that objective on the single-GPU emulation path
3. validate strict pipeline as the faithful efficient realization of the same objective
4. compare weighting families and long-horizon behavior only after the objective layer is clean

## Current reading of the completed replay bundle

The replay-complete `pipeline_then_weighting` bundle supports a different story from the older `beta001`-centric narrative:

- `uniform` is the strongest trajectory improver
  - it reaches strong checkpoints earlier
  - it produces the strongest best-checkpoint results in both single emulation and pipeline P4
- `deep` is a stronger endpoint-oriented weighting
  - it is not the best trajectory shaper
  - but it gives the best `pipeline P4` fixed endpoint in the current seed-1 pilot
- `beta001` is not the mainline winner under either view
  - it is not the strongest endpoint configuration
  - it is not the strongest trajectory configuration

This means the next package must not spend its main budget on more `beta` sweep runs. Instead, it should promote `uniform` and `deep` into the new mainline candidates.

## Phase structure

### O1: objective-first single-side sweep

Purpose:
- test whether the objective family itself remains useful before invoking the multi-GPU system story

Configs:
- `single_fullk` baseline control
- `single_pipe_emul`
  - `P = 2, 4, 8, 16`
  - weighting in `{uniform, deep_only}`
  - `K = 100`
  - `steps = 300k`

Outputs:
- final endpoint FID
- best checkpoint FID
- best step
- first reach threshold(s)
- late-window mean
- trajectory plots

### O2: faithful strict-pipeline realization

Purpose:
- compare `single_pipe_emul(P, weighting)` against `pipe_strict(P, weighting)` under the same objective definition
- isolate the system contribution from the objective contribution

Configs:
- `ddp_fullk` baseline control
- `pipe_strict`
  - `P = 2, 4, 8, 16`
  - weighting in `{uniform, deep_only}`
  - `K = 100`
  - `steps = 300k`

Outputs:
- paired comparison table by `(P, weighting)`
- final / best / threshold metrics
- wall-clock and GPU-hours
- throughput and step-time

### O3: seed expansion

Purpose:
- move seed budget from `beta001` to the new mainline candidates

Initial scaffold in the manifest:
- `ddp_fullk K=100` seeds `2-5`
- `single_pipe_emul P2 deep` seeds `2-3`
- `single_pipe_emul P4 uniform` seeds `2-3`
- `pipe_strict P4 uniform` seeds `2-5`
- `pipe_strict P4 deep` seeds `2-5`

These are intentionally provisional. If O1/O2 show different winners, the manifest should be revised rather than forcing the current `P4` placeholders.

### O4: 500k long horizon

Purpose:
- rerun long-horizon only for the new objective-family winners
- avoid borrowing `beta001` long-horizon results for a new story they do not support

Initial scaffold:
- single baseline `K=100`
- DDP baseline `K=100`
- `single_pipe_emul P2 deep`
- `single_pipe_emul P4 uniform`
- `pipe_strict P4 uniform`
- `pipe_strict P4 deep`

### O5: compute-matched controls

Purpose:
- replace the uninformative `K=400` mechanism test

Configs:
- `ddp_fullk K = 25, 50, 100`
- selected `pipe_strict K = 100` candidates

Required comparisons:
- equal wall-clock
- equal GPU-hours
- equal model-evaluation budget

## Reporting contract

Every follow-up summary should treat trajectory metrics as first-class outputs:

- `final_fid`
- `best_fid`
- `best_step`
- `first_leq_70`
- `first_leq_65`
- `first_leq_60`
- `late_window_mean_fid`
- `walltime`
- `gpu_hours`

For paired comparisons, the main table should be:

| P | weighting | single emulation | strict pipeline | gap |
| --- | --- | --- | --- | --- |

where each cell includes at least:
- final FID
- best FID
- best step
- first threshold reach
- wall-clock / GPU-hours

## ImageNet-32 handoff

ImageNet-32 remains the external validation benchmark. Its first job is not to crown a lowest final FID winner. Its first job is to check whether the same split persists under conditional state:

- does `uniform` still look like a trajectory improver?
- does `deep` still look like an endpoint winner?
- is strict pipeline faithful to the matched single-emulation objective?

Any ImageNet paired table should include:
- final / best FID
- FID vs step / wall-clock / GPU-hours
- conditional accuracy
- per-class average accuracy
- fixed-class sample grids

## Notes on the manifest

The companion manifest `experiments/objective_first_followup_manifest.csv` is intentionally conservative:

- it defaults every row to `submit=no`
- it encodes the current best guess for likely winners
- it separates objective screening, paired pipeline realization, seed expansion, 500k follow-up, and compute-matched controls

This package is intended to be edited after O1/O2 close the remaining `P × {uniform, deep}` gaps.
