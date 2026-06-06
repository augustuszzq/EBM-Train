# Start Here

This document is the first-reader path for collaborators who have not seen the
Polaris EBM codebase before.

## Ten-Minute Orientation

Read these files in order:

1. `README.md`
2. `docs/START_HERE.md`
3. `docs/CODE_MAP.md`
4. `docs/SCRIPT_INDEX.md`
5. `docs/ACTIVE_EXPERIMENTS.md`
6. `docs/DRL_QUICKSTART.md` if you are running the ImageNet-32 DRL path

If you only inspect five source files, inspect:

1. `scripts/current/mode_a_contract.py`
2. `scripts/current/ebm_train_sync_mode_a.py`
3. `scripts/current/ebm_train_single_pipe_emul.py`
4. `scripts/current/ebm_train_baseline_single.py`
5. `scripts/current/build_master_experiment_registry.py`

These cover the method contract, distributed/pipeline training path,
single-GPU objective emulation, single full-K baseline, and paper-run registry.

## Mental Model

The project studies one objective and several execution modes.

The objective is:

```text
L_t = f_pos - sum_s alpha_s * f_neg(x_{t,s})
```

The important rule is that `alpha_s` weights scalar negative-energy loss terms.
It never weights or averages image tensors. Sampling and training communicate
only through chain states and updated model parameters.

The execution modes are:

| Mode | What changes |
| --- | --- |
| `single_fullk` | One GPU runs full `K` Langevin steps before the optimizer update. |
| `ddp_fullk` | DDP runs the same full-`K` baseline across ranks. |
| `single_pipe_emul` | One GPU serially emulates the multi-stage objective. |
| `pipe_strict` | Multiple GPUs implement the stage pipeline. |

The intended paper comparison is about execution semantics and
completion-aware negative objectives, not about changing the model family,
loss sign, replay semantics, or sampler semantics between modes.

## Repository Scope

This clean repository is a source and lightweight-provenance snapshot. It does
not include:

- checkpoints
- generated image samples
- FID image dumps
- dataset caches
- large `runs_*` directories

Use `provenance/` to inspect generated configs, submitted manifests, summary
CSVs, and lightweight run metadata. Use the original working tree or Polaris
run storage for large artifacts.

## Environment

Use Python 3.10 or newer. On Polaris login nodes, the default `python3` may be
too old for this repository. The environment used during the clean-repo check
was:

```bash
/home/kevienzzq/.conda/envs/llm-env/bin/python
```

To avoid thread-creation failures on login nodes during quick tests, set:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## Local Sanity Checks

From the repository root:

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

These checks validate imports, shell syntax, method-contract tests, the DRL
backbone module, and long-K manifest generation. They do not run full training.

## Common Tasks

| Task | Start with |
| --- | --- |
| Understand the method semantics | `scripts/current/mode_a_contract.py` and `tests/test_mode_a_contract.py` |
| Inspect DDP and strict pipeline training | `scripts/current/ebm_train_sync_mode_a.py` |
| Inspect single-GPU objective emulation | `scripts/current/ebm_train_single_pipe_emul.py` |
| Inspect single full-K baseline | `scripts/current/ebm_train_baseline_single.py` |
| Rebuild current CIFAR registry | `scripts/current/build_master_experiment_registry.py` |
| Rebuild FID-vs-step table | `scripts/current/collect_current_cifar_fid_vs_step.py` |
| Inspect long-K scaling configs | `scripts/current/long_k_multinode_sweep_manifest.py` |
| Inspect external ImageNet-32/DRL work | `scripts/current/imagenet32_drl_manifest.py` and `scripts/current/drl_resnet_energy.py` |
| Run DRL on a new server | `docs/DRL_QUICKSTART.md` |

## What Not To Mix

Do not mix these ledgers when reporting experiment counts:

- `current_cifar_paper`
- `historical_exploratory`
- `external_benchmark`

Do not mix these units:

- experiment family
- logical run: family x seed x horizon
- scheduler attempt or resume fragment

For the canonical CIFAR accounting, use:

```bash
PYTHONPATH=. "$PY" scripts/current/build_master_experiment_registry.py
```

## Before Submitting Jobs

The clean repo contains PBS templates and generated provenance, but it does not
guarantee that the current shell has the right Polaris project, queue, modules,
dataset paths, or GPU allocation.

Before submitting jobs, check:

```bash
qstat -u "$USER"
screen -ls
```

For live long-K sweeps, inspect watcher logs in the original run directory, not
only the clean repository snapshot.
