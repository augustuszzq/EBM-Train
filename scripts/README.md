# Scripts Layout

- `scripts/current/`
  - Current EBM workflow entrypoints.
  - Includes the three active training modes, eval scripts, ablation submitters, collectors, and replay tooling.
- `scripts/archive/`
  - Legacy or one-off scripts that are no longer part of the current workflow.
  - Includes old stream runtimes, deprecated PBS launchers, plotting helpers, and archived experiments.

Current canonical entrypoints:

- Single baseline: `scripts/current/ebm_train_baseline_single.py`
- DDP strict / pipeline strict: `scripts/current/ebm_train_sync_mode_a.py`
- Single pipeline emulation: `scripts/current/ebm_train_single_pipe_emul.py`
- Canonical eval: `scripts/current/eval_generate.py` + `scripts/current/eval_metrics.py`
- Canonical ablation launcher: `scripts/current/pbs_run_ablation_case.pbs`
