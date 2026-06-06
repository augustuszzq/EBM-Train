# DRL-Backbone ImageNet-32 Stability Restart

This restart keeps the paper method unchanged:

- No recovery likelihood objective.
- No diffusion noise ladder.
- No adaptive weighting.
- The training loss remains the completion-aware diagonal stage-output objective.
- Stage outputs are combined only as negative energy scalars, never in image space.

## Observed Failure

The unstable run was DRL-backbone ImageNet-32 strict pipeline P4 equal with `K=100`, slices `[25,25,25,25]`, `lr=1e-4`, `step_size=0.01`, `noise_std=0.01`, and `no_clamp_x=true`.

The pipeline run exploded around step 20: energy grew rapidly and `max_abs_chain` moved past 4 and then 7. The DDP full-K reference with the same backbone/data was initially stable, which points to the interaction between unbounded chains, large Langevin updates, high LR, and equal weighting over partially completed shallow stages.

## Implemented Controls

- ImageNet-32 DRL scripts no longer pass `--no_clamp_x`; chain states are clamped to `[-1, 1]` after every Langevin update by default.
- Diagnostic logging now includes weighted `f_neg_total`, per-stage negative energies, per-stage chain norms, active stage set, alpha weights, optimizer update status, and parameter gradient norm.
- Crash guards can stop runs early on extreme energy, unclamped chain blow-up, non-finite values, or large parameter gradient norm.
- `--skip_optimizer_until_full_diagonal` fills the pipeline diagonal before the first optimizer update. This changes only the fill phase, not the steady-state objective.
- `--grad_clip_norm` is an explicit optional override for global gradient clipping. It is enabled only in the safest fallback diagnostic row.

## Diagnostic Matrix

All rows start at 500 steps and must pass before any 150k/300k external validation run is launched.

| ID | Mode | Weight | LR | Step Size | Noise | Extra |
| - | - | - | -: | -: | -: | - |
| A | DDP full-K | terminal | `5e-5` | `0.005` | `0.005` | clamp |
| B | Pipeline P4 | equal | `5e-5` | `0.005` | `0.005` | clamp, skip fill optimizer |
| C | Pipeline P4 | equal | `2.5e-5` | `0.002` | `0.005` | conservative |
| D | Pipeline P4 | deepest | `5e-5` | `0.005` | `0.005` | shallow-stage control |
| E | Pipeline P4 | equal | `1e-5` | `0.001` | `0.002` | grad clip 10 |

## Decision Rules

- If B is stable for 2k steps, extend B to 10k and then 50k.
- If B fails but C is stable, use C as the Phase 1 candidate.
- If D is stable while equal weighting fails, equal-weight shallow stages are the likely instability source; do not change the main method without a separate stabilization experiment.
- If all pipeline diagnostics fail but A is stable, inspect stage energies, chain norms, diagonal aggregation, and fill handling before launching longer jobs.
- If A also fails, validate the DRL backbone on CIFAR before continuing ImageNet-32.

## Stability Criteria

A diagnostic run is stable if, for 2k steps:

- `max_abs_chain <= 1.05` with clamp enabled.
- `abs(f_pos)` and `abs(f_neg_total)` remain below 100.
- No monotonic energy explosion appears.
- Parameter gradient norm remains finite and does not spike toward the crash threshold.
- Loss remains finite.
