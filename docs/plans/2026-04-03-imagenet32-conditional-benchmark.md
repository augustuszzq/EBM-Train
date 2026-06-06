# ImageNet-32 Conditional Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn the current CIFAR-centric unconditional EBM repo into a general conditional-benchmark framework and make class-conditional ImageNet-32 the primary second benchmark without changing the established single/DDP/pipeline semantics.

**Architecture:** Keep the current CIFAR paths as backward-compatible defaults, then add a new shared benchmark layer that owns benchmark configs, conditional chain state, replay semantics, conditional energy models, and ImageNet-32 dataset loading/materialization. Wire existing training/eval entrypoints through that layer only when a benchmark config or conditional benchmark is requested so the legacy CIFAR experiments remain untouched.

**Tech Stack:** Python, PyTorch, torchvision, optional Hugging Face `datasets`, YAML configs, pytest, shell wrappers, PBS launchers.

### Task 1: Define the conditional benchmark contract in code

**Files:**
- Create: `scripts/current/conditional_chain.py`
- Create: `scripts/current/benchmark_config.py`
- Test: `tests/test_conditional_chain_state.py`
- Test: `tests/test_imagenet32_config.py`

**Step 1: Write failing tests for `ChainState`**

Test:
- `ChainState` must require `x` and `y`
- `y` must be integer typed and batch-aligned with `x`
- `steps_done` and `valid` metadata must round-trip through serialization
- invalid state must raise a clear error

**Step 2: Run tests to verify failure**

Run:
```bash
PYTHONPATH=/eagle/lc-mpi/Zhiqing python3 -m pytest \
  tests/test_conditional_chain_state.py \
  tests/test_imagenet32_config.py -q
```

Expected:
- import failure because the modules do not exist yet

**Step 3: Implement minimal `ChainState` and config loader**

Create:
- a dataclass-like `ChainState`
- YAML-backed config loader for benchmark/model/train/eval sections
- semantic validators for:
  - stage weights summing to 1
  - `K` vs `P` slice consistency
  - conditional benchmark requiring `num_classes > 0`

**Step 4: Re-run the tests**

Expected:
- tests pass

### Task 2: Add a conditional replay buffer with label binding

**Files:**
- Create: `scripts/current/conditional_replay.py`
- Test: `tests/test_conditional_replay.py`

**Step 1: Write failing replay tests**

Test:
- replay entries store `(x, y)` together
- sampling for `y_batch` returns matching labels when available
- restarts from noise keep the requested label instead of relabeling old states
- checkpoint save/load preserves labels and replay stats

**Step 2: Run the replay tests and verify failure**

Run:
```bash
PYTHONPATH=/eagle/lc-mpi/Zhiqing python3 -m pytest \
  tests/test_conditional_replay.py -q
```

Expected:
- import failure or missing attribute failures

**Step 3: Implement minimal replay buffer**

Implement:
- flat replay storage for `x`, `y`, `valid`, `steps_done`
- label-to-slot lookup
- sampling that prefers matching labels and otherwise restarts from noise
- checkpointable `state_dict()/load_state_dict()`

**Step 4: Re-run the replay tests**

Expected:
- tests pass

### Task 3: Add a conditional energy model and sampler API

**Files:**
- Create: `scripts/current/conditional_model.py`
- Create: `scripts/current/conditional_sampler.py`
- Test: `tests/test_conditional_energy_model.py`
- Test: `tests/test_conditional_sampler.py`

**Step 1: Write failing model/sampler tests**

Test:
- conditional model exposes `forward(x, y)`
- the same backbone is usable across single/DDP/pipeline
- Langevin sampler requires `y`
- labels remain immutable during sampling

**Step 2: Run tests to verify failure**

Run:
```bash
PYTHONPATH=/eagle/lc-mpi/Zhiqing python3 -m pytest \
  tests/test_conditional_energy_model.py \
  tests/test_conditional_sampler.py -q
```

Expected:
- import failure or signature mismatch failures

**Step 3: Implement minimal model/sampler**

Implement:
- a projection-style conditional energy head on top of the current conv family
- a `langevin_sample_conditional(model, x, y, ...)` helper
- runtime assertions for label immutability and finite outputs

**Step 4: Re-run tests**

Expected:
- tests pass

### Task 4: Add ImageNet-32 dataset integration and export path

**Files:**
- Create: `scripts/current/imagenet32_data.py`
- Create: `scripts/current/export_hf_imagenet32.py`
- Test: `tests/test_imagenet32_data.py`

**Step 1: Write failing dataset tests**

Test:
- config can select ImageNet-32 or CelebA-64
- ImageNet-32 loader reports train/val split metadata
- export path is deterministic and restart-safe
- missing `datasets` dependency yields a clear actionable error

**Step 2: Run tests to verify failure**

Run:
```bash
PYTHONPATH=/eagle/lc-mpi/Zhiqing python3 -m pytest \
  tests/test_imagenet32_data.py -q
```

Expected:
- import failure or missing behavior failures

**Step 3: Implement dataset loader/materializer**

Implement:
- optional HF-backed ImageNet-32 reader
- deterministic export/materialization helper
- CelebA-64 fallback builder
- benchmark metadata helpers for image shape and class count

**Step 4: Re-run tests**

Expected:
- tests pass

### Task 5: Wire the current training/eval entrypoints through the new benchmark layer

**Files:**
- Modify: `scripts/current/ebm_train_sync_mode_a.py`
- Modify: `scripts/current/ebm_train_baseline_single.py`
- Modify: `scripts/current/ebm_train_single_pipe_emul.py`
- Modify: `scripts/current/eval_generate.py`
- Modify: `scripts/current/eval_metrics.py`
- Test: `tests/test_imagenet32_entrypoints.py`

**Step 1: Write failing entrypoint tests**

Test:
- legacy CIFAR defaults still work without benchmark config
- new config path is accepted by all three modes
- conditional benchmark path swaps in the new model/dataset/sampler interfaces
- pipeline runtime refuses to send image-only state in conditional mode

**Step 2: Run the tests and verify failure**

Run:
```bash
PYTHONPATH=/eagle/lc-mpi/Zhiqing python3 -m pytest \
  tests/test_imagenet32_entrypoints.py -q
```

Expected:
- CLI or runtime assertion failures

**Step 3: Implement minimal wiring**

Implement:
- optional `--config` / `--benchmark` path handling
- shared benchmark/model creation helpers
- conditional eval-generate path with per-class sample scheduling
- conditional metrics path and sample-label bookkeeping

**Step 4: Re-run tests**

Expected:
- tests pass

### Task 6: Add configs, wrappers, aggregation stubs, and a smoke-test slice

**Files:**
- Create: `configs/imagenet32_single_strict.yaml`
- Create: `configs/imagenet32_ddp_strict.yaml`
- Create: `configs/imagenet32_pipeline_strict.yaml`
- Create: `configs/imagenet32_ablation.yaml`
- Create: `scripts/run_imagenet32_single_strict.sh`
- Create: `scripts/run_imagenet32_ddp_strict.sh`
- Create: `scripts/run_imagenet32_pipeline_strict.sh`
- Create: `scripts/run_imagenet32_ablation_screen.sh`
- Create: `scripts/run_imagenet32_mainline.sh`
- Create: `scripts/run_imagenet32_scaling.sh`
- Create: `scripts/eval_imagenet32_fid.sh`
- Create: `scripts/eval_imagenet32_conditional_acc.sh`
- Create: `scripts/aggregate_imagenet32_results.py`
- Test: `tests/test_imagenet32_scripts.py`

**Step 1: Write failing script/aggregation tests**

Test:
- wrappers resolve the repo root correctly
- configs load and validate
- aggregation script emits CSV/JSON placeholders with required columns

**Step 2: Run tests to verify failure**

Run:
```bash
PYTHONPATH=/eagle/lc-mpi/Zhiqing python3 -m pytest \
  tests/test_imagenet32_scripts.py -q
```

Expected:
- file-not-found failures

**Step 3: Implement minimal wrappers and aggregator skeleton**

Implement:
- thin shell wrappers around the active Python entrypoints
- config-aware aggregation scaffolding with required output column names
- a smoke-test slice for Phase A (1k/5k/20k planning only; no large run claims)

**Step 4: Run validation**

Run:
```bash
bash -n scripts/run_imagenet32_single_strict.sh \
  scripts/run_imagenet32_ddp_strict.sh \
  scripts/run_imagenet32_pipeline_strict.sh \
  scripts/run_imagenet32_ablation_screen.sh \
  scripts/run_imagenet32_mainline.sh \
  scripts/run_imagenet32_scaling.sh \
  scripts/eval_imagenet32_fid.sh \
  scripts/eval_imagenet32_conditional_acc.sh

PYTHONPATH=/eagle/lc-mpi/Zhiqing python3 -m pytest \
  tests/test_conditional_chain_state.py \
  tests/test_imagenet32_config.py \
  tests/test_conditional_replay.py \
  tests/test_conditional_energy_model.py \
  tests/test_conditional_sampler.py \
  tests/test_imagenet32_data.py \
  tests/test_imagenet32_entrypoints.py \
  tests/test_imagenet32_scripts.py -q
```

Expected:
- all new tests pass

### Task 7: Document current readiness and known missing pieces

**Files:**
- Modify: `README.md`
- Modify: `docs/ablation_phase2_plan.md` if cross-referenced
- Create: `docs/imagenet32_benchmark_status.md`

**Step 1: Record what is implemented now**

Document:
- conditional chain-state contract
- ImageNet-32 loader/export assumptions
- which scripts are smoke-test ready

**Step 2: Record what still remains for Phase A–G execution**

Document:
- dataset materialization prerequisites
- classifier-based conditional-accuracy dependency
- long-run experiment submission still pending
