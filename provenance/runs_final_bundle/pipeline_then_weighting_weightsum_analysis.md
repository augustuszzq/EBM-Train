# Pipeline-Then-Weighting Analysis

Data sources:

- `pipeline_then_weighting_replay_points.csv`
- `pipeline_then_weighting_replay_summary.csv`

This note focuses on two questions:

1. Relative to the matching baseline, does weighted-sum show faster convergence?
2. Is the mid-training strength of `uniform` just an accidental spike, or a persistent trend?

## Reference baselines

- Single baseline: `E1_single_fullk_K100_s300k`
  - final FID: `62.761`
  - best FID: `53.551 @ 280k`
- DDP baseline: `E1_ddp_fullk_K100_s300k`
  - final FID: `69.015`
  - best FID: `56.314 @ 170k`

## Single emulation: weighting vs single baseline

### Endpoint view

- `uniform`
  - final: `64.708`
  - best: `52.487 @ 150k`
- `deep`
  - final: `98.172`
  - best: `61.004 @ 190k`
- `beta=0.001`
  - final: `75.033`
  - best: `62.778 @ 285k`

Takeaway:

- `uniform` is the only single-emulation weighting that is competitive with the single baseline.
- `deep` and `beta=0.001` are both worse at the fixed `300k` endpoint.

### Convergence-speed view

- Single baseline first reaches `<= 62.761` at `145k`
- `uniform` first reaches `<= 62.761` at `80k`
- `deep` first reaches `<= 62.761` at `190k`
- `beta=0.001` never reaches `<= 62.761`

- Single baseline best-level threshold `<= 53.551`
  - baseline reaches it at `280k`
  - `uniform` reaches it at `150k`
  - `deep` never reaches it
  - `beta=0.001` never reaches it

### Is `uniform` just an accidental spike?

Evidence that it is not just a one-off lucky point:

- `uniform` has `11` checkpoints at or below the single baseline final FID
- its longest consecutive streak below the baseline final is `2` checkpoints
- its best point `52.487 @ 150k` is better than the single baseline best `53.551`

But it is also not a stable endpoint improver:

- after `150k`, it rebounds substantially
- late-window mean (`200k+`) is `69.669`, worse than the single baseline late mean `65.890`
- final endpoint is still worse than the single baseline final

Conclusion:

- In single emulation, `uniform` behaves like a genuine early-trajectory improver, not a pure random spike.
- But that improvement does not persist cleanly to the final endpoint.

## Pipeline P4: weighting vs DDP baseline

### Endpoint view

- `deep`
  - final: `63.510`
  - best: `61.569 @ 280k`
- `uniform`
  - final: `67.776`
  - best: `48.713 @ 285k`
- `beta=0.001`
  - final: `68.048`
  - best: `60.687 @ 270k`

Takeaway:

- For the fixed endpoint, `deep` is the best of the three.
- `uniform` and `beta=0.001` both finish worse than `deep`.

### Convergence-speed view

- DDP baseline first reaches `<= 69.015` at `70k`
- `uniform` reaches `<= 69.015` at `55k`
- `deep` reaches `<= 69.015` at `145k`
- `beta=0.001` reaches `<= 69.015` at `145k`

- DDP baseline best-level threshold `<= 56.314`
  - baseline reaches it at `170k`
  - `uniform` reaches it at `195k`
  - `deep` never reaches it
  - `beta=0.001` never reaches it

### Is `uniform` just an accidental spike?

Evidence that it is not:

- `uniform` has `28` checkpoints at or below the DDP baseline final FID
- its longest consecutive streak below the DDP baseline final is `7` checkpoints
- late-window mean (`200k+`) is `65.781`, which is better than the DDP baseline late mean `68.907`
- its best point `48.713 @ 285k` is dramatically stronger than the DDP baseline best `56.314`

So the mid/late strength of pipeline `uniform` is not just a single noisy dip. It is repeated and sustained over a nontrivial region of training.

However:

- its endpoint still rebounds from `48.713` to `67.776`
- `deep` still wins on fixed-endpoint quality

Conclusion:

- In pipeline mode, `uniform` is a strong trajectory improver and best-checkpoint improver.
- `deep` is the stronger endpoint configuration.

## What this says about weighted-sum

The replay evidence does not support the simple claim:

- "weighted-sum improves final endpoint quality"

Instead it supports a more specific story:

- `uniform` can improve the optimization trajectory, especially in terms of reaching strong checkpoints earlier or more often
- `deep` is more aligned with fixed-endpoint quality in the pipeline setting
- `beta=0.001` is not the strongest weighting in this batch; it is outperformed by `uniform` on trajectory quality and by `deep` on endpoint quality

## Practical interpretation

- If the metric is fixed endpoint at `300k`, prefer `deep` in pipeline mode.
- If the metric is best checkpoint or entering a strong FID regime earlier, `uniform` is the most compelling weighting in this batch.
- For single emulation, `uniform` clearly has early-training value, but that value is not stable enough to dominate the final endpoint.
