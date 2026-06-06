# DRL-Style ImageNet-32 Quickstart

This page is for running the DRL-style ResNet energy backbone on a new server
after cloning this repository.

The DRL path adapts only the energy network backbone. It does not use diffusion
recovery likelihood, diffusion timesteps, or a diffusion noise ladder.

## 1. Install Environment

From the repository root:

```bash
bash scripts/current/setup_drl_env.sh
conda activate ebm-train
```

If conda is unavailable, the setup script creates `.venv` and installs
`requirements-drl.txt`:

```bash
source .venv/bin/activate
```

The default conda environment uses `pytorch-cuda=12.1`. If the target server
uses a different CUDA stack, edit `environment.yml` before running setup.

## 2. Configure Local Paths

Copy the example file:

```bash
cp .env.drl.example .env.drl
```

Edit at least:

```bash
DATA_DIR=/absolute/path/to/imagenet32
HF_CACHE_DIR=/absolute/path/to/hf_cache
CUDA_VISIBLE_DEVICES=0,1,2,3
WORLD_SIZE=4
```

The expected data layout is:

```text
${DATA_DIR}/train/labels.csv
${DATA_DIR}/train/images/<class>/<image>.png
${DATA_DIR}/val/labels.csv
${DATA_DIR}/val/images/<class>/<image>.png
```

## 3. Materialize ImageNet-32

If the target server can access Hugging Face:

```bash
bash scripts/current/materialize_imagenet32_local.sh
```

For a tiny loader smoke test, set limits before running:

```bash
TRAIN_MAX_EXAMPLES=1000 VAL_MAX_EXAMPLES=100 \
  bash scripts/current/materialize_imagenet32_local.sh
```

The source dataset is `ChocolateDave/imagenet-32`.

## 4. Run A Local DRL Smoke

Command-generation dry run:

```bash
DRY_RUN=1 bash scripts/current/run_imagenet32_drl_smoke_local.sh
```

DDP full-K smoke:

```bash
bash scripts/current/run_imagenet32_drl_smoke_local.sh
```

For a single-GPU run, use one local rank:

```bash
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 \
  bash scripts/current/run_imagenet32_drl_smoke_local.sh
```

Pipeline P4 equal smoke:

```bash
DRL_TRAIN_MODE=pipe_strict \
DRL_PIPE_STAGES=4 \
DRL_WEIGHT_MODE=uniform \
  bash scripts/current/run_imagenet32_drl_smoke_local.sh
```

The script writes runs under:

```text
runs_imagenet32_drl_local/
```

Each run contains:

- `config_local.yaml`
- `train.log`
- `checkpoints/`
- `vis/`

`config_local.yaml` is generated from the repository config but with
`benchmark.data_root` rewritten to the local `DATA_DIR`.

## 5. Generate Samples And Compute A Rough FID

After a run has at least one checkpoint:

```bash
bash scripts/current/eval_imagenet32_drl_local.sh /path/to/run_dir
```

By default this computes a 5k-image screening metric:

```text
DRL_EVAL_IMAGES=5000
DRL_EVAL_NUM_REAL=5000
DRL_SKIP_STANDARD_FID=1
```

This is useful for early sanity checks. It is not the paper-level 50k
ImageNet-32 FID protocol. For a heavier evaluation:

```bash
DRL_EVAL_IMAGES=50000 \
DRL_EVAL_NUM_REAL=50000 \
DRL_EVAL_IMAGES_PER_CLASS=50 \
DRL_SKIP_STANDARD_FID=0 \
  bash scripts/current/eval_imagenet32_drl_local.sh /path/to/run_dir
```

## 6. Stable Diagnostic Defaults

The default `.env.drl.example` values are conservative:

```text
DRL_LR=1e-7
DRL_LR_WARMUP_STEPS=10000
DRL_STEP_SIZE=5e-4
DRL_NOISE_STD=5e-4
DRL_GRAD_CLIP_NORM=1.0
```

Use these for portability checks before attempting longer runs.

## 7. Notes For Non-Polaris Servers

The local scripts do not require PBS. They use `torchrun --standalone`.

PBS files under `scripts/current/*.pbs` are Polaris-specific and include
Polaris account/module assumptions. Use them only after adapting scheduler,
module, conda, and filesystem paths.
