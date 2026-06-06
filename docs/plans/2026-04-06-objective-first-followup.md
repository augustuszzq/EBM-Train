# Objective-First Follow-Up Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Re-center the CIFAR benchmark on an objective-first story where `uniform` is tested as a trajectory improver, `deep` is tested as an endpoint-oriented weighting, and strict pipeline is evaluated as the faithful efficient realization of those objectives.

**Architecture:** Treat the current replay-complete `pipeline_then_weighting` bundle as the seed-1 pilot, freeze it in a canonical registry, then build a new follow-up experiment package around `P × {uniform, deep}`. The follow-up package should pair single emulation and strict pipeline under matched objective definitions, then expand seeds, long-horizon runs, and compute-matched controls only for the surviving winners.

**Tech Stack:** Existing CIFAR train/eval scripts in `scripts/current/`, CSV manifests in `experiments/`, PBS launchers, replay trajectory collectors, and the `runs_final_bundle/` summary bundle.

### Task 1: Freeze the canonical registry

**Files:**
- Create: `runs_final_bundle/canonical_run_registry.csv`
- Create: `scripts/current/build_canonical_run_registry.py`
- Test: `tests/test_build_canonical_run_registry.py`

**Step 1: Treat the registry as the only source of truth**

Include completed rows from:
- `runs_final_bundle/phase1_summary.csv`
- `runs_final_bundle/phase2_summary.csv`
- `runs_final_bundle/ablation_500k_summary.csv`
- `runs_final_bundle/pipeline_then_weighting_current.csv`
- `runs_final_bundle/imagenet32_phasea_current.csv`

**Step 2: Record identity fields explicitly**

For every row, record:
- source table / experiment package
- benchmark and conditionality
- story bucket
- train mode / execution semantics
- objective family
- `P`, `K`, `steps`, `lr`, `step_size`, weighting, `beta`, seed
- run / eval / trajectory paths
- current status and inclusion status
- config signature and config hash

**Step 3: Mark what belongs to the current paper story**

Use:
- `include_in_current_story = yes` for completed `pipeline_then_weighting` and `imagenet32_phasea` rows
- `include_in_current_story = context_only` for older screening / legacy mainline / 500k rows
- `include_in_current_story = no` for historical failed rows

**Step 4: Verify**

Run:
```bash
PYTHONPATH=/eagle/lc-mpi/Zhiqing/polaris_ebm python3 -m pytest \
  /eagle/lc-mpi/Zhiqing/polaris_ebm/tests/test_build_canonical_run_registry.py -q
python3 /eagle/lc-mpi/Zhiqing/polaris_ebm/scripts/current/build_canonical_run_registry.py
```

Expected:
- test pass
- `runs_final_bundle/canonical_run_registry.csv` written cleanly

### Task 2: Build the objective-first CIFAR follow-up manifest

**Files:**
- Create: `experiments/objective_first_followup_manifest.csv`
- Create: `docs/exp_plan_objective_first_followup.md`

**Step 1: Freeze the questions**

Write the manifest around:
- `single_fullk` baseline controls
- `ddp_fullk` baseline controls
- `single_pipe_emul` for `P = 2, 4, 8, 16`
- `pipe_strict` for `P = 2, 4, 8, 16`
- weighting limited to `uniform` and `deep_only`

**Step 2: Separate phases**

Include explicit phases:
- `O1`: single-side `P × {uniform, deep}` objective sweep at `300k`
- `O2`: paired strict-pipeline faithful realization at the same configs
- `O3`: seed expansion for winners
- `O4`: `500k` long-horizon only for the new winners
- `O5`: compute-matched controls

**Step 3: Keep beta out of the mainline**

Do not allocate new mainline budget to `last2_beta` in this manifest. Keep beta-family rows only if needed as `legacy_context = yes`.

### Task 3: Add paired trajectory-first reporting

**Files:**
- Create: `scripts/current/collect_objective_first_followup.py`
- Create: `runs_final_bundle/objective_first_followup_template.md`

**Step 1: Promote trajectory metrics to first-class outputs**

Every summary table must include:
- final FID
- best FID
- best step
- first reach threshold(s)
- late-window mean
- wall-clock
- GPU-hours

**Step 2: Produce paired tables**

Require at least:
- `single emulation` vs `strict pipeline`
- matched by `P`, weighting, `K`, seed

### Task 4: Seed expansion for the new mainline

**Files:**
- Modify: `experiments/objective_first_followup_manifest.csv`
- Create: `docs/seed_budget_objective_first.md`

**Step 1: Move seed budget from beta to uniform/deep**

Minimum follow-up target:
- `DDP strict K=100`: 5 seeds
- `single_pipe_emul P* uniform`: 3 seeds
- `single_pipe_emul P* deep`: 3 seeds
- `pipe_strict P* uniform`: 5 seeds
- `pipe_strict P* deep`: 5 seeds

**Step 2: Allow different winners**

If single-side and pipeline-side prefer different `P*`, keep both. Do not force them into the same mainline config.

### Task 5: Long-horizon follow-up

**Files:**
- Modify: `experiments/objective_first_followup_manifest.csv`
- Create: `docs/objective_first_500k_scope.md`

**Step 1: Only rerun 500k for the new winners**

Required minimum:
- `single_fullk K=100` baseline
- best `single_pipe_emul` config under the new story
- best `ddp_fullk K=100`
- best `pipe_strict` config under the new story

**Step 2: Keep both endpoint and best-checkpoint reporting**

For all `500k` rows report:
- final endpoint
- best checkpoint
- best step
- late-window mean

### Task 6: Replace the high-K mechanism control

**Files:**
- Create: `experiments/objective_first_compute_matched_manifest.csv`
- Create: `docs/objective_first_compute_matched.md`

**Step 1: Drop `K=400` as the main mechanism proof**

Use instead:
- strict baseline `K = 25, 50, 100`
- pipeline winner at `K = 100`

**Step 2: Compare under matched budgets**

Every comparison must include:
- equal wall-clock
- equal GPU-hours
- equal model-evaluation budget inside Langevin

### Task 7: Tie ImageNet-32 to the new story

**Files:**
- Modify: `experiments/imagenet32_phasea_manifest.csv`
- Create: `docs/imagenet32_objective_readout.md`

**Step 1: Keep ImageNet-32 as the external validation benchmark**

The first ImageNet question is not “who has the lowest final FID?” It is:
- does `uniform` still look like a trajectory improver?
- does `deep` still look like an endpoint winner?
- is strict pipeline faithful to the matched single-emulation objective?

**Step 2: Require conditional-faithfulness metrics**

For every ImageNet paired comparison, report:
- final / best FID
- FID vs step / wall-clock / GPU-hours
- top-1 conditional accuracy
- per-class average accuracy
- fixed-class sample grids

### Task 8: Final report handoff

**Files:**
- Create: `docs/objective_first_followup_report_template.md`

**Step 1: Make the paper wording conservative**

Required framing:
- `uniform` as trajectory shaping / early-entry weighting
- `deep` as endpoint-oriented weighting
- `pipeline` as faithful efficient realization

**Step 2: Explicitly mark unsupported claims**

Do not claim:
- beta-family remains the main mechanism
- `P=4` is universally optimal before the new `P × {uniform, deep}` sweep completes
- high-K failure is mechanism proof

