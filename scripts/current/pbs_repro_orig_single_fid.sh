#!/bin/bash
#PBS -A lc-mpi
#PBS -l select=1:system=polaris
#PBS -l walltime=03:00:00
#PBS -l filesystems=home:eagle
#PBS -N ebm_orig_fid1g
#PBS -j oe

set -euo pipefail

module use /soft/modulefiles
module load conda
conda activate /home/kevienzzq/.conda/envs/llm-env

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PROJECT_DIR="/eagle/lc-mpi/Zhiqing/polaris_ebm"
RUN_BASE="/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_baseline"

# Repro defaults from the original single-GPU snippet.
export DATA_DIR="${DATA_DIR:-/eagle/lc-mpi/Zhiqing/ebm/data/cifar10}"
export STEPS="${STEPS:-100000}"
export NUM_EVAL="${NUM_EVAL:-5000}"
export K="${K:-100}"
export M="${M:-64}"
export NF="${NF:-64}"
export SIGMA_PD="${SIGMA_PD:-3e-2}"
export NOISE_STD="${NOISE_STD:-1e-2}"
export STEP_SIZE="${STEP_SIZE:-1.0}"
export LR="${LR:-1e-4}"
export SEED="${SEED:-1}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_TAG="orig_single_semantics_fid_s${STEPS}_K${K}_${TS}"
export RUN_DIR="${RUN_BASE}/${RUN_TAG}"
mkdir -p "${RUN_DIR}"

cd "${PROJECT_DIR}"

echo "[JOB] RUN_DIR=${RUN_DIR}"
echo "[JOB] Host=$(hostname)"
echo "[JOB] Config: STEPS=${STEPS} NUM_EVAL=${NUM_EVAL} K=${K} M=${M} NF=${NF} STEP_SIZE=${STEP_SIZE} SIGMA_PD=${SIGMA_PD}"

/home/kevienzzq/.conda/envs/llm-env/bin/python -u - <<'PY' | tee "${RUN_DIR}/train_and_sample.log"
import json
import os
import time

import torch as t
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tr

seed = int(os.environ["SEED"])
im_sz = 32
sigma = float(os.environ["SIGMA_PD"])
n_ch = 3
m = int(os.environ["M"])
K = int(os.environ["K"])
n_f = int(os.environ["NF"])
n_i = int(os.environ["STEPS"])
num_eval = int(os.environ["NUM_EVAL"])
noise_std = float(os.environ["NOISE_STD"])
step_size = float(os.environ["STEP_SIZE"])
lr = float(os.environ["LR"])

run_dir = os.environ["RUN_DIR"]
data_dir = os.environ["DATA_DIR"]

start = time.time()
t.manual_seed(seed)
if t.cuda.is_available():
    t.cuda.manual_seed_all(seed)

device = t.device("cuda" if t.cuda.is_available() else "cpu")
print(
    f"[CFG] device={device} seed={seed} im_sz={im_sz} sigma_pd={sigma} m={m} K={K} n_f={n_f} "
    f"steps={n_i} step_size={step_size} noise_std={noise_std}",
    flush=True,
)


class F(nn.Module):
    def __init__(self, n_c=n_ch, n_f=n_f, l=0.2):
        super(F, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),
            nn.LeakyReLU(l),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),
            nn.LeakyReLU(l),
            nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1),
            nn.LeakyReLU(l),
            nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1),
            nn.LeakyReLU(l),
            nn.Conv2d(n_f * 8, 1, 4, 1, 0),
        )

    def forward(self, x):
        return self.f(x).squeeze()


f = F().to(device)
transform = tr.Compose([tr.Resize(im_sz), tr.ToTensor(), tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
p_d = t.stack([x[0] for x in tv.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform)]).to(device)

noise = lambda x: x + sigma * t.randn_like(x)


def sample_p_d():
    p_d_i = t.randint(0, p_d.shape[0], (m,), device=device)
    return noise(p_d[p_d_i]).detach()


sample_p_0 = lambda: t.empty(m, n_ch, im_sz, im_sz, device=device).uniform_(-1, 1)


def sample_q(k_steps=K):
    x_k = t.autograd.Variable(sample_p_0(), requires_grad=True)
    for _ in range(k_steps):
        f_prime = t.autograd.grad(f(x_k).sum(), [x_k], retain_graph=True)[0]
        x_k.data += step_size * f_prime + noise_std * t.randn_like(x_k)
    return x_k.detach()


optim = t.optim.Adam(f.parameters(), lr=lr, betas=[0.9, 0.999])

for i in range(n_i):
    x_p_d, x_q = sample_p_d(), sample_q()
    L = f(x_p_d).mean() - f(x_q).mean()
    optim.zero_grad()
    (-L).backward()
    optim.step()

    if i % 100 == 0:
        print(
            "{:>6d} f(x_p_d)={:>14.9f} f(x_q)={:>14.9f}".format(
                i, f(x_p_d).mean().item(), f(x_q).mean().item()
            ),
            flush=True,
        )

train_seconds = time.time() - start
print(f"[TRAIN] done in {train_seconds:.1f}s", flush=True)

chunks = []
made = 0
while made < num_eval:
    xq = sample_q(k_steps=K)
    x01 = ((xq.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)
    chunks.append(x01.cpu())
    made += x01.size(0)
    if made % 512 == 0 or made >= num_eval:
        print(f"[SAMPLE] generated={made}/{num_eval}", flush=True)

samples_01 = t.cat(chunks, dim=0)[:num_eval]
out_pt = os.path.join(run_dir, "baseline_samples.pt")
t.save({"samples_01": samples_01}, out_pt)

meta = {
    "samples_path": out_pt,
    "num_eval": int(samples_01.size(0)),
    "sample_mean": float(samples_01.mean().item()),
    "sample_std": float(samples_01.std(unbiased=False).item()),
    "sample_min": float(samples_01.min().item()),
    "sample_max": float(samples_01.max().item()),
    "train_seconds": float(train_seconds),
}
with open(os.path.join(run_dir, "train_meta.json"), "w") as fp:
    json.dump(meta, fp, indent=2)

print(f"[SAMPLE] wrote {out_pt}", flush=True)
PY

/home/kevienzzq/.conda/envs/llm-env/bin/python -u scripts/current/eval_metrics.py \
  --baseline_samples "${RUN_DIR}/baseline_samples.pt" \
  --pipeline_samples "${RUN_DIR}/baseline_samples.pt" \
  --data_dir "${DATA_DIR}" \
  --num_real 5000 \
  --batch 256 \
  --out_json "${RUN_DIR}/metrics_compare.json" \
  --device cuda | tee "${RUN_DIR}/eval_metrics.log"

/home/kevienzzq/.conda/envs/llm-env/bin/python - <<'PY'
import json
import os

run_dir = os.environ["RUN_DIR"]
p = os.path.join(run_dir, "metrics_compare.json")
with open(p) as f:
    m = json.load(f)
print("[RESULT] fid_inception_baseline_vs_real =", m.get("fid_inception_baseline_vs_real"))
print("[RESULT] fid_feature_baseline_vs_real =", m.get("fid_feature_baseline_vs_real"))
print("[RESULT] metrics json =", p)
PY

echo "[JOB] done"
