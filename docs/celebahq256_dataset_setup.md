# CelebA-HQ 256 Dataset Setup

This note records the dataset-only integration for the CelebA-HQ 256 external benchmark.

## Scope

This change only prepares the dataset path. It does not change the paper method, training objective, sampler, replay semantics, or pipeline semantics.

Current target:

- benchmark name: `celebahq256`
- conditioning: unconditional
- image shape: `3 x 256 x 256`
- tensor range: `[-1, 1]`
- dummy label: `0`, returned only for dataloader compatibility

The current YAML configs are dataset-ready placeholders. Before long training, confirm that the selected EBM backbone is intended for `256 x 256` inputs.

## NVAE-Compatible Layout

NVAE prepares CelebA-HQ 256 by downloading GLOW TFRecords and converting them to LMDB:

```text
$DATA_DIR/celeba/celeba-tfr/
$DATA_DIR/celeba/celeba-lmdb/train.lmdb
$DATA_DIR/celeba/celeba-lmdb/validation.lmdb
```

Recommended project-local data root on Polaris:

```bash
export EBM_DATA_ROOT=/eagle/lc-mpi/Zhiqing/polaris_ebm/data
export CELEBA_ROOT=$EBM_DATA_ROOT/celeba
export CELEBA_LMDB=$CELEBA_ROOT/celeba-lmdb
```

The default configs currently point to the prepared Polaris LMDB:

```text
/eagle/lc-mpi/Zhiqing/polaris_ebm/data/celeba/celeba-lmdb
```

Override that path in the YAML or launch environment if the dataset is mirrored elsewhere.

## Preparing LMDB From GLOW/NVAE TFRecords

The original NVAE README points to the GLOW Google Storage URL. That URL may be unavailable now. The dataset prepared here used the OpenAI public Azure mirror referenced in the NVAE issue tracker:

```bash
mkdir -p "$CELEBA_ROOT"
cd "$CELEBA_ROOT"
curl -L -C - --retry 10 --retry-delay 10 --retry-all-errors \
  -o celeba-tfr.tar \
  https://openaipublic.azureedge.net/glow-demo/data/celeba-tfr.tar
tar -xf celeba-tfr.tar
```

If `celeba-tfr` already exists, convert it with:

```bash
PYTHONPATH=/lus/eagle/projects/lc-mpi/Zhiqing \
python scripts/current/celebahq256_data.py \
  --convert \
  --tfr_path "$CELEBA_ROOT/celeba-tfr" \
  --lmdb_path "$CELEBA_LMDB" \
  --split both
```

Required optional packages for conversion:

```bash
pip install lmdb tfrecord
```

The converter writes restart markers:

```text
$CELEBA_LMDB/train._SUCCESS.json
$CELEBA_LMDB/validation._SUCCESS.json
```

If a partial LMDB exists without a success marker, rebuild explicitly:

```bash
python scripts/current/celebahq256_data.py \
  --convert \
  --tfr_path "$CELEBA_ROOT/celeba-tfr" \
  --lmdb_path "$CELEBA_LMDB" \
  --split train \
  --force
```

## Image Folder Fallback

For local smoke checks or non-NVAE exports, the loader also accepts image folders:

```text
data/celeba_hq_256/train/*.png
data/celeba_hq_256/validation/*.png
```

Nested class folders are also accepted, but labels are ignored because this benchmark path is unconditional.

## Verification

Inspect a dataset root without training:

```bash
PYTHONPATH=/lus/eagle/projects/lc-mpi/Zhiqing \
python scripts/current/celebahq256_data.py \
  --verify_only \
  --data_root "$CELEBA_LMDB"
```

Run the unit test with low thread count on login nodes:

```bash
PYTHONPATH=/lus/eagle/projects/lc-mpi/Zhiqing \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/kevienzzq/.conda/envs/llm-env/bin/python \
  -m pytest tests/test_celebahq256_data.py -q
```

Expected behavior:

- `resolve_benchmark_runtime()` recognizes `celebahq256`
- `build_dataset()` loads either NVAE LMDB or image-folder data
- samples return `(x, 0)`
- `x.shape == (3, 256, 256)`
- `x` is normalized to `[-1, 1]`

## Prepared Polaris Instance

Prepared path:

```text
/eagle/lc-mpi/Zhiqing/polaris_ebm/data/celeba/celeba-lmdb
```

Observed contents:

```text
train.lmdb: 27000 entries
validation.lmdb: 3000 entries
source tar bytes: 4535848007
source tar md5_base64: vlrkkB+PtXc3oTgeHdYKTg==
```

## Config Entrypoints

Dataset-ready configs:

```text
configs/celebahq256_single_strict.yaml
configs/celebahq256_ddp_strict.yaml
configs/celebahq256_pipeline_strict.yaml
```

These configs intentionally do not imply that the existing CIFAR-scale backbone is the right final `256 x 256` model. Treat them as clean dataset selection configs until the model scale decision is made.
