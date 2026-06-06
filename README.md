# Polaris EBM Completion-Aware Pipeline

This is a clean collaboration snapshot of the Polaris EBM codebase. It keeps
the source code, launch scripts, configs, tests, and lightweight provenance
needed to understand and reproduce the current experiments. It intentionally
excludes checkpoints, generated samples, FID image dumps, cached datasets, and
large run directories.

## What This Repository Is For

The project studies completion-aware negative-energy objectives for image
energy-based models (EBMs), and strict multi-GPU pipeline execution as a
systems implementation of the same objective.

If this is your first time reading the repository, start with:

- `docs/START_HERE.md`
- `docs/CODE_MAP.md`
- `docs/SCRIPT_INDEX.md`
- `docs/ACTIVE_EXPERIMENTS.md`
- `docs/DRL_QUICKSTART.md` for clone-and-run DRL setup on another server

The central training objective is:

```text
L_t = f_pos - sum_s alpha_s * f_neg(x_{t,s})
```

where `x_{t,s}` is the negative chain state at completion stage `s`, and
`alpha_s` is the scalar loss weight for that stage.

Important semantic constraints:

- The weighted sum is applied to scalar negative-energy loss terms.
- Images from different stages are never averaged in image space.
- Sampling and training are coupled only through chain states and updated model
  parameters.
- Single, DDP, and pipeline modes must use the same model family, sampler,
  replay semantics, optimizer semantics, and loss sign convention.
- The intended difference across modes is execution semantics only.

## Main Execution Modes

| Mode | Meaning | Entrypoint |
| --- | --- | --- |
| `single_fullk` | Single-GPU strict full-K baseline | `scripts/current/ebm_train_baseline_single.py` |
| `ddp_fullk` | Distributed full-K strict baseline | `scripts/current/ebm_train_sync_mode_a.py --mode ddp_fullk` |
| `single_pipe_emul` | Single-GPU serialized multi-stage objective emulation | `scripts/current/ebm_train_single_pipe_emul.py` |
| `pipe_strict` | Strict multi-GPU pipeline implementation | `scripts/current/ebm_train_sync_mode_a.py --mode pipeline` |

For strict pipeline runs with total Langevin depth `K` and `P` stages, each
stage receives a slice of Langevin steps per global optimizer step:

```text
base = floor(K / P)
first K mod P stages receive base + 1 steps
remaining stages receive base steps
```

For example, `K=100, P=8` gives `[13,13,13,13,12,12,12,12]`.

## Repository Layout

| Path | Role |
| --- | --- |
| `scripts/current/` | Current training, evaluation, manifest, collector, and PBS helper code. Start here. |
| `launchers/current/` | Current HSN/multi-node launcher wrappers. |
| `configs/` | Hand-written YAML configs, mostly ImageNet-32 and DRL-style external validation. |
| `docs/` | Experiment plans, status notes, and code maps. |
| `experiments/` | Manifest and submitted-manifest CSV records. |
| `tests/` | Current unit and contract tests. Archive tests were intentionally excluded. |
| `provenance/` | Lightweight generated configs, PBS files, submitted manifests, and summary CSVs. |

Large generated run directories are not included. The original working tree
contained many `runs_*` directories with checkpoints and image artifacts; this
repository only keeps lightweight provenance needed for traceability.

For a first-reader walkthrough, see `docs/START_HERE.md`. For a script-by-script
classification of `scripts/current/`, see `docs/SCRIPT_INDEX.md`.

## Core Source Files

Current CIFAR training:

- `scripts/current/ebm_train_sync_mode_a.py`
- `scripts/current/ebm_train_baseline_single.py`
- `scripts/current/ebm_train_single_pipe_emul.py`
- `scripts/current/mode_a_contract.py`
- `scripts/current/benchmark_config.py`
- `scripts/current/benchmark_runtime.py`

Long-K and scaling experiments:

- `scripts/current/long_k_multinode_sweep_manifest.py`
- `scripts/current/long_k_multinode_300k_continuations.py`
- `scripts/current/submit_long_k_multinode_sweep_resumes.py`
- `scripts/current/long_k_batch_scaling_manifest.py`
- `scripts/current/collect_long_k_wallclock.py`
- `scripts/current/ebm_overhead_microbench.py`

Evaluation and aggregation:

- `scripts/current/eval_generate.py`
- `scripts/current/eval_metrics.py`
- `scripts/current/eval_fid_trajectory.py`
- `scripts/current/collect_current_cifar_fid_vs_step.py`
- `scripts/current/collect_seed_fid_trajectories.py`
- `scripts/current/build_master_experiment_registry.py`

ImageNet-32 and DRL-style external validation:

- `scripts/current/imagenet32_data.py`
- `scripts/current/conditional_chain.py`
- `scripts/current/conditional_replay.py`
- `scripts/current/conditional_sampler.py`
- `scripts/current/conditional_model.py`
- `scripts/current/drl_resnet_energy.py`
- `scripts/current/ebm_train_imagenet32_drl_sync.py`
- `scripts/current/ebm_train_imagenet32_drl_single.py`
- `scripts/current/eval_imagenet32_fid.py`
- `scripts/current/eval_imagenet32_conditional_acc.py`
- `scripts/current/eval_generate_imagenet32_drl.py`
- `scripts/current/eval_imagenet32_drl_local.sh`
- `scripts/current/setup_drl_env.sh`
- `scripts/current/materialize_imagenet32_local.sh`
- `scripts/current/run_imagenet32_drl_smoke_local.sh`

## Current Experiment Ledgers

The CIFAR paper accounting separates three units:

- Experiment family.
- Logical run: family x seed x horizon.
- Scheduler attempt or resume fragment.

Do not mix these units when reporting experiment counts.

Main CIFAR groups:

- `O1`: single-side objective sweep.
- `O2`: distributed paired sweep.
- `O3`: seed top-up for current mainline.
- `O4`: 500k long-horizon runs.
- `O5`: compute-matched shallow-chain controls.

External benchmarks, including ImageNet-32 and DRL-style backbone validation,
belong to a separate external-benchmark ledger and should not be mixed into
CIFAR O1-O5 totals.

To rebuild the current CIFAR registry:

```bash
python3 scripts/current/build_master_experiment_registry.py
```

Expected outputs are under `runs_final_bundle/` in the original working tree.
In this clean repository, lightweight copies are under
`provenance/runs_final_bundle/`.

## Long-K Scaling Status

The active long-K branch uses the original stable CIFAR model and objective,
not the DRL-style ImageNet backbone. The latest generated configs and PBS files
are preserved under:

```text
provenance/runs_long_k_multinode_sweep/
```

This branch contains:

- `M0`: 20k smoke/screening runs.
- `M1`: 300k continuation runs.

The most recent PBS files were updated to use `72:00:00` walltime for
preemptable jobs. If regenerating or resubmitting these jobs, check:

- `scripts/current/long_k_multinode_sweep_manifest.py`
- `scripts/current/long_k_multinode_300k_continuations.py`

## DRL-Style External Validation Status

The DRL path adapts a ResNet-style energy backbone inspired by diffusion
recovery likelihood code, but it does not use recovery likelihood, diffusion
noise ladders, or diffusion training objectives.

Important status:

- Original DRL-style port was unstable under the ordinary EBM objective.
- Later `sampler_prime`, temperature, and loss-scale variants improved DDP
  gate stability.
- DDP 20k gates have completed for selected prime/scale settings.
- Pipeline 20k gates remain incomplete and should be treated as external
  validation work, not current CIFAR paper evidence.

For clone-and-run setup on another server, see `docs/DRL_QUICKSTART.md`.

## Typical Local Validation

The clean repo is primarily a source and provenance snapshot. It does not
include datasets or checkpoints. Basic syntax validation is still useful:

Use Python 3.10 or newer. On Polaris login nodes, the default `python3` may be
too old. A known working interpreter is:

```bash
export PY=/home/kevienzzq/.conda/envs/llm-env/bin/python
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHONPATH=. "$PY" -m py_compile scripts/current/*.py polaris_ebm/__init__.py
bash -n launchers/current/run_hsn.sh \
  launchers/current/run_hsn_baseline.sh \
  launchers/current/run_hsn_pipeline.sh \
  scripts/current/*.sh \
  scripts/current/*.pbs
PYTHONPATH=. "$PY" -m pytest \
  tests/test_mode_a_contract.py \
  tests/test_drl_resnet_energy.py \
  tests/test_long_k_scaling_manifest.py \
  -q
```

Full training requires the Polaris environment, dataset paths, and PBS account
configuration used in the original working tree.

## Provenance Policy

Included:

- Source code.
- Current configs.
- Current tests.
- Experiment manifests.
- Submitted-manifest CSVs.
- Generated PBS/config files for active long-K branches.
- Lightweight summary CSV/Markdown records.

Excluded:

- Checkpoints.
- Generated samples.
- FID image dumps.
- Dataset caches.
- Large `runs_*` output directories.
- Python caches.

See `provenance/excluded_binary_artifacts.txt` for the binary artifact scan
performed during export.
