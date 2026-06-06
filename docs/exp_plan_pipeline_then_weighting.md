# Pipeline Then Weighting Experiment Plan

## Paper Story

This experiment package reorganizes the method around the algorithmic objective first and
the multi-GPU system second.

At global training step `t`, with `P` completion stages and negative states `x_{t,s}`, the
training target is:

```text
L_t = f_pos - Σ_s α_s f_neg(x_{t,s})
```

The central semantic contract is fixed:

- weighting happens on the negative scalar energy terms
- image tensors from different stages are never averaged
- the next sampling state is never formed by averaging stage images
- sampling influences training through `x_{t,s} -> f_neg(x_{t,s})`
- training influences later sampling through updated parameters `θ_t -> θ_{t+1}`

The multi-GPU strict pipeline is therefore treated as an efficient implementation of this
completion-aware weighted energy objective, not as the primary source of the objective
itself.

## Why E1 Comes Before E2

### E1: Pipeline Factor First

E1 fixes the weighting to the cleanest possible objective:

- `deep_only`
- deepest-stage one-hot for each `P`

This isolates the completion-aware objective and asks three questions:

1. Is the objective feasible on a single GPU without any communication?
2. Does multi-GPU strict pipeline faithfully reproduce the same objective?
3. Which stage count `P` is the best systems/quality operating point?

Because the weighting is fixed, any difference in E1 is attributed mainly to completion
depth structure and execution semantics rather than mixture refinement.

### E2: Weighted-Sum Factor After Best `P`

Only after E1 identifies the best stage count `P*` do we study weighting families:

- `uniform`
- `deep_only`
- late-stage `last2_beta`

This makes the interpretation cleaner:

- E1 answers whether completion-aware staging and pipeline execution are useful at all.
- E2 answers whether the weighting family is an additional refinement once the best
  completion structure is fixed.

## Scientific Questions

### E1: Pipeline Factor With Deep-Only Objective

- Does the deep-only completion-aware objective work on a single GPU?
- Does strict pipeline preserve the same objective on multiple GPUs?
- Is `P=4` still the best stage count, or does the best `P` shift?

### E2: Weighting Factor At Fixed Best `P`

- Does `uniform` outperform `deep_only`?
- Does late-stage weighting outperform both `uniform` and `deep_only`?
- Is weighted-sum merely a refinement, or a distinct source of gain once `P*` is fixed?

## Immediate Execution Plan

### Current Horizon Decision

The current execution decision for this package is to run both the submitted E1 rows and
the prepared E2 rows at `300k` rather than a short screening horizon. E1 remains the only
phase submitted immediately; E2 stays prepared with `submit=no`.

### Submit Now

Submit all E1 rows marked `submit=yes`:

- baseline controls
- single emulation deep-only for `P=2/4/8/16`
- strict pipeline deep-only for `P=2/4/8/16`

### Prepare But Do Not Submit Yet

Write all E2 rows into the manifest, but keep:

```text
submit=no
```

until E1 determines the best stage count.
