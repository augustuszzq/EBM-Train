# Phase 2 300k Plan

This document records the Phase 2 long-run matrix built on top of the Phase 1 screening infrastructure.

## Anchors

These 300k references are already available and are registered as anchors only:

1. `single_fullk_K100_seed1_300k`
2. `ddp_fullk_K100_seed1_300k`
3. `pipe_strict_P4_K100_beta001_lr1e4_seed1_300k`

## Submit Now

These 8 cases are the current Phase 2 batch:

1. `ddp_fullk_K100_seed2_300k`
2. `pipe_strict_P4_K100_beta001_lr1e4_seed2_300k`
3. `ddp_fullk_K400_seed1_300k`
4. `pipe_strict_P4_K400_beta001_lr1e4_seed1_300k`
5. `single_pipe_emul_P4_K100_deep_seed1_300k`
6. `single_pipe_emul_P4_K100_beta001_seed1_300k`
7. `pipe_strict_P2_K100_beta001_lr1e4_seed1_300k`
8. `pipe_strict_P8_K100_beta001_lr1e4_seed1_300k`

## Why These 8

- `M1` checks seed robustness for the two baseline-vs-pipeline 300k headline lines.
- `M2` checks whether the pipeline gain survives when `P=4, K=400`, where each GPU slice already reaches `100` Langevin steps.
- `S1` isolates the mechanism on a single GPU without distributed communication.
- `S2` extends the stage-count factor from the 20k screening into the 300k regime.

## Eval Policy

Phase 2 follows the current local-eval policy:

- training jobs go through PBS
- eval jobs are not submitted to PBS
- completed runs are evaluated later with `scripts/run_local_eval_phase2.py`

## Files

- Manifest: `experiments/ablation_manifest_phase2.csv`
- Submitted manifest: `experiments/ablation_manifest_phase2_submitted.csv`
- Submitter: `scripts/submit_ablation_phase2.sh`
- Local eval: `scripts/run_local_eval_phase2.py`
- Collector: `scripts/collect_ablation_phase2.py`
- Report renderer: `scripts/render_ablation_phase2_report.py`

## Outputs

Phase 2 summaries are written to:

- `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation_phase2/phase2_summary.csv`
- `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation_phase2/phase2_summary.json`
- `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation_phase2/phase2_summary.md`
