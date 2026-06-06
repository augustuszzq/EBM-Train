# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import os, time, csv, argparse
# import torch as t
# import torch.nn as nn
# import torchvision as tv
# import torchvision.transforms as tr

# def parse_args():
#     ap = argparse.ArgumentParser("Single-GPU EBM baseline (original semantics)")
#     # data / io
#     ap.add_argument("--data_dir", type=str, default="./data/cifar10",
#                     help="CIFAR-10 directory (must exist on compute node; no download).")
#     ap.add_argument("--output_dir", type=str, default="./runs_baseline",
#                     help="Run output dir (images + metrics.csv).")
#     # model / train
#     ap.add_argument("--n_f", type=int, default=64, help="base channel width (matches 'nf').")
#     ap.add_argument("--im_sz", type=int, default=32)
#     ap.add_argument("--n_ch", type=int, default=3)
#     ap.add_argument("--batch", type=int, default=64, help="m in the original code.")
#     ap.add_argument("--K", type=int, default=100, help="Langevin steps for sampling x_q.")
#     ap.add_argument("--steps", type=int, default=10000, help="training iterations (n_i).")
#     ap.add_argument("--lr", type=float, default=1e-4)
#     ap.add_argument("--sigma_pd", type=float, default=3e-2,
#                     help="noise added to positive samples p_d (same as 'sigma' in original).")
#     ap.add_argument("--noise_std", type=float, default=1e-2,
#                     help="noise scale in the Langevin sampler (original fixed 1e-2).")
#     ap.add_argument("--step_size", type=float, default=1.0,
#                     help="multiplicative factor on grad_x f; set to 1.0 to match original.")
#     ap.add_argument("--seed", type=int, default=1)
#     ap.add_argument("--save_every", type=int, default=100,
#                     help="save image & log every N steps.")
#     return ap.parse_args()

# # ---- model identical to your original 'F' ----
# class F(nn.Module):
#     def __init__(self, n_c=3, n_f=64, l=0.2):
#         super().__init__()
#         self.f = nn.Sequential(
#             nn.Conv2d(n_c, n_f, 3, 1, 1), nn.LeakyReLU(l),
#             nn.Conv2d(n_f, n_f*2, 4, 2, 1), nn.LeakyReLU(l),
#             nn.Conv2d(n_f*2, n_f*4, 4, 2, 1), nn.LeakyReLU(l),
#             nn.Conv2d(n_f*4, n_f*8, 4, 2, 1), nn.LeakyReLU(l),
#             nn.Conv2d(n_f*8, 1, 4, 1, 0)
#         )
#     def forward(self, x):
#         return self.f(x).squeeze()  # [B,1,1,1] -> [B]

# def main():
#     args = parse_args()
#     os.makedirs(args.output_dir, exist_ok=True)
#     img_dir = os.path.join(args.output_dir, "images")
#     os.makedirs(img_dir, exist_ok=True)
#     csv_path = os.path.join(args.output_dir, "metrics.csv")

#     # device & seed
#     device = t.device("cuda" if t.cuda.is_available() else "cpu")
#     t.manual_seed(args.seed)
#     if t.cuda.is_available():
#         t.cuda.manual_seed_all(args.seed)

#     # data (same as original: load entire CIFAR10 to device)
#     tfm = tr.Compose([tr.Resize(args.im_sz), tr.ToTensor(),
#                       tr.Normalize((.5,.5,.5),(.5,.5,.5))])
#     try:
#         ds = tv.datasets.CIFAR10(root=args.data_dir, download=False, transform=tfm)
#     except Exception as e:
#         raise RuntimeError(
#             f"Failed to load CIFAR-10 at {args.data_dir} (no download on compute node). "
#             f"Error: {e}"
#         )
#     # stack all images to a big tensor on GPU (original style)
#     p_d = t.stack([x[0] for x in ds]).to(device, non_blocking=True)

#     # helpers (names & formulas align with your original)
#     noise_pd = lambda x: x + args.sigma_pd * t.randn_like(x)

#     def sample_p_d(m: int):
#         idx = t.randint(0, p_d.shape[0], (m,), device=device)
#         return noise_pd(p_d[idx]).detach()

#     def sample_p_0(m: int):
#         return t.empty(m, args.n_ch, args.im_sz, args.im_sz, device=device).uniform_(-1, 1)

#     def sample_q(K: int, m: int):
#         # original: each step uses grad_x f and adds fixed noise (no clamp, retain_graph=True)
#         x = sample_p_0(m).requires_grad_(True)
#         for _ in range(K):
#             f_sum = model(x).sum()
#             # retain_graph=True matches the original code; not strictly needed, but kept for fidelity.
#             g = t.autograd.grad(f_sum, [x], retain_graph=True)[0]
#             # original update: x.data += grad + 1e-2 * noise
#             x.data.add_(args.step_size * g).add_(args.noise_std * t.randn_like(x))
#         return x.detach()

#     # model & optimizer
#     model = F(n_c=args.n_ch, n_f=args.n_f).to(device)
#     optim = t.optim.Adam(model.parameters(), lr=args.lr, betas=(.9, .999))

#     # logging header
#     with open(csv_path, "w", newline="") as f:
#         csv.writer(f).writerow(["step", "f_pos", "f_neg", "metric(f_pos-f_neg)", "loss(-metric)", "iter_time_sec",
#                                 "n_f", "K", "batch", "sigma_pd", "noise_std", "step_size"])

#     # utility
#     sqrt = lambda x: int(t.sqrt(t.tensor([x]))[0].item())
#     save_grid = lambda path, x: tv.utils.save_image(t.clamp(x, -1., 1.), path, normalize=True, nrow=sqrt(args.batch))

#     start = time.time()
#     last = start

#     for i in range(args.steps):
#         iter_t0 = time.time()

#         x_p_d = sample_p_d(args.batch)
#         x_q   = sample_q(args.K, args.batch)

#         f_pos = model(x_p_d).mean()
#         f_neg = model(x_q).mean()
#         metric = f_pos - f_neg
#         loss = -metric

#         optim.zero_grad(set_to_none=True)
#         loss.backward()
#         optim.step()

#         iter_t1 = time.time()
#         it = iter_t1 - iter_t0

#         if (i % args.save_every) == 0:
#             print(f"{i:6d}  f(x_p_d)={f_pos.item():>12.6f}  f(x_q)={f_neg.item():>12.6f}  "
#                   f"metric={metric.item():>12.6f}  loss={loss.item():>12.6f}  iter={it:.3f}s",
#                   flush=True)
#             save_grid(os.path.join(img_dir, f"x_q_{i:06d}.png"), x_q)
#             with open(csv_path, "a", newline="") as f:
#                 csv.writer(f).writerow([i, f"{f_pos.item():.6f}", f"{f_neg.item():.6f}",
#                                         f"{metric.item():.6f}", f"{loss.item():.6f}",
#                                         f"{it:.6f}", args.n_f, args.K, args.batch,
#                                         args.sigma_pd, args.noise_std, args.step_size])

#     print(f"Done. Total time: {time.time()-start:.2f}s  |  outputs -> {args.output_dir}", flush=True)

# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-GPU EBM baseline (compatible with existing PBS args).

Why this file:
- Your PBS script passes legacy flags: --batch --sigma_pd --save_every
- Previous "new baseline" used different names (pos_noise_std, save_samples_every, csv_every)
- This script supports BOTH styles so you can keep PBS unchanged.

Key features:
- Keeps original baseline semantics (fresh init, no clamp by default, sigma_pd noise on positives)
- Optional timing breakdown: sampler_total_ms / train_cuda_ms (CUDA events)
- Allows decoupling: log_every / csv_every / save_images_every
  (Default: all follow save_every for backward compatibility)
- Optional save .pt tensors for offline metrics

Environment overrides (optional, no PBS edit required):
- BASELINE_PROFILE_TIMING=1
- BASELINE_LOG_EVERY=50
- BASELINE_CSV_EVERY=1
- BASELINE_SAVE_IMAGES_EVERY=200
- BASELINE_SAVE_SAMPLES_PT=1
"""

import os
import time
import csv
import argparse

import torch as t
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tr


# ---- model identical to original baseline ----
class F(nn.Module):
    def __init__(self, n_c=3, n_f=64, leak=0.2):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1), nn.LeakyReLU(leak),
            nn.Conv2d(n_f, n_f*2, 4, 2, 1), nn.LeakyReLU(leak),
            nn.Conv2d(n_f*2, n_f*4, 4, 2, 1), nn.LeakyReLU(leak),
            nn.Conv2d(n_f*4, n_f*8, 4, 2, 1), nn.LeakyReLU(leak),
            nn.Conv2d(n_f*8, 1, 4, 1, 0)
        )

    def forward(self, x):
        return self.f(x).squeeze()  # [B,1,1,1] -> [B]


def env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "")
    if v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def env_int(name: str, default=None):
    v = os.environ.get(name, "")
    if v == "":
        return default
    try:
        return int(v.strip())
    except Exception:
        return default


def parse_args():
    ap = argparse.ArgumentParser("Single-GPU EBM baseline (PBS-compatible)", allow_abbrev=False)

    # data / io
    ap.add_argument("--data_dir", type=str, default="./data/cifar10",
                    help="CIFAR-10 directory (must exist on compute node; no download).")
    ap.add_argument("--output_dir", type=str, default="./runs_baseline",
                    help="Output dir (images + metrics.csv).")

    # model / train
    ap.add_argument("--n_f", type=int, default=64)
    ap.add_argument("--im_sz", type=int, default=32)
    ap.add_argument("--n_ch", type=int, default=3)

    # ---- batch args (support both) ----
    ap.add_argument("--batch", type=int, default=64, help="(legacy) batch size")
    ap.add_argument("--batch_size", type=int, default=None, help="(alias) batch size")

    ap.add_argument("--K", type=int, default=100)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=1e-4)

    # ---- pos noise args (support both) ----
    ap.add_argument("--sigma_pd", type=float, default=None, help="(legacy) positive noise std")
    ap.add_argument("--pos_noise_std", type=float, default=None, help="(alias) positive noise std")

    # sampler
    ap.add_argument("--noise_std", type=float, default=1e-2)
    ap.add_argument("--step_size", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)

    # ---- legacy save_every (PBS passes this) ----
    ap.add_argument("--save_every", type=int, default=100,
                    help="(legacy) print + save image + write csv every N steps")

    # new optional knobs (can decouple frequencies)
    ap.add_argument("--log_every", type=int, default=None, help="print every N steps")
    ap.add_argument("--csv_every", type=int, default=None, help="append CSV every N steps")
    ap.add_argument("--save_images_every", type=int, default=None, help="save x_q image grid every N steps")

    # timing / saving extras
    ap.add_argument("--profile_timing", action="store_true",
                    help="record CUDA event timing: sampler_total_ms, train_cuda_ms")
    ap.add_argument("--save_samples_pt", action="store_true",
                    help="also save raw x_q tensor .pt when saving images")

    return ap.parse_args()


def main():
    args = parse_args()

    # ---- reconcile aliases ----
    if args.batch_size is not None:
        args.batch = int(args.batch_size)

    if args.pos_noise_std is not None and args.sigma_pd is None:
        args.sigma_pd = float(args.pos_noise_std)
    if args.sigma_pd is None:
        args.sigma_pd = 3e-2  # baseline default

    # ---- env overrides (no PBS edit required) ----
    if env_flag("BASELINE_PROFILE_TIMING", False):
        args.profile_timing = True
    if env_flag("BASELINE_SAVE_SAMPLES_PT", False):
        args.save_samples_pt = True

    log_env = env_int("BASELINE_LOG_EVERY", None)
    csv_env = env_int("BASELINE_CSV_EVERY", None)
    img_env = env_int("BASELINE_SAVE_IMAGES_EVERY", None)

    # ---- default frequencies: follow save_every (backward compatible) ----
    if args.log_every is None:
        args.log_every = args.save_every
    if args.csv_every is None:
        args.csv_every = args.save_every
    if args.save_images_every is None:
        args.save_images_every = args.save_every

    # apply env overrides last
    if log_env is not None:
        args.log_every = log_env
    if csv_env is not None:
        args.csv_every = csv_env
    if img_env is not None:
        args.save_images_every = img_env

    # ---- dirs ----
    os.makedirs(args.output_dir, exist_ok=True)
    img_dir = os.path.join(args.output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "metrics.csv")

    # ---- device & seed ----
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    t.manual_seed(args.seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(args.seed)
        t.backends.cudnn.benchmark = True

    # ---- data: load CIFAR10 and stack to device (original style) ----
    tfm = tr.Compose([tr.Resize(args.im_sz), tr.ToTensor(),
                      tr.Normalize((.5, .5, .5), (.5, .5, .5))])
    try:
        ds = tv.datasets.CIFAR10(root=args.data_dir, download=False, transform=tfm)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load CIFAR-10 at {args.data_dir} (no download on compute node). Error: {e}"
        )

    p_d = t.stack([x[0] for x in ds]).to(device, non_blocking=True)

    # ---- helpers ----
    def sample_p_d(m: int):
        idx = t.randint(0, p_d.shape[0], (m,), device=device)
        x = p_d[idx]
        x = x + args.sigma_pd * t.randn_like(x)  # baseline: no clamp
        return x.detach()

    def sample_p_0(m: int):
        return t.empty(m, args.n_ch, args.im_sz, args.im_sz, device=device).uniform_(-1, 1)

    # sampler (baseline fidelity; retain_graph True is legacy, but does not change results)
    def sample_q(K: int, m: int):
        x = sample_p_0(m).requires_grad_(True)
        for _ in range(K):
            f_sum = model(x).sum()
            g = t.autograd.grad(f_sum, [x], retain_graph=True)[0]
            x.data.add_(args.step_size * g).add_(args.noise_std * t.randn_like(x))
        return x.detach()

    # ---- model & optimizer ----
    model = F(n_c=args.n_ch, n_f=args.n_f).to(device)
    optim = t.optim.Adam(model.parameters(), lr=args.lr, betas=(.9, .999))

    # ---- CSV header ----
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "step",
            "f_pos", "f_neg",
            "metric", "metric(f_pos-f_neg)",  # write both for compatibility
            "loss(-metric)",
            "iter_time_sec",
            "sampler_total_ms", "train_cuda_ms",  # NaN if not profile_timing
            # config snapshot
            "n_f", "K", "batch", "sigma_pd", "noise_std", "step_size",
            "log_every", "csv_every", "save_images_every", "profile_timing",
        ])

    # ---- timing events ----
    samp_s = samp_e = train_s = train_e = None
    if args.profile_timing and device.type == "cuda":
        samp_s = t.cuda.Event(enable_timing=True)
        samp_e = t.cuda.Event(enable_timing=True)
        train_s = t.cuda.Event(enable_timing=True)
        train_e = t.cuda.Event(enable_timing=True)

    def sqrt_int(n: int) -> int:
        return int(t.sqrt(t.tensor([n], dtype=t.float32)).item())

    def save_grid(path: str, x: t.Tensor, nrow: int):
        tv.utils.save_image(t.clamp(x, -1., 1.), path, normalize=True, nrow=nrow)

    start = time.time()

    for i in range(args.steps):
        iter_t0 = time.time()

        x_p_d = sample_p_d(args.batch)

        sampler_total_ms = float("nan")
        train_cuda_ms = float("nan")

        # ---- sampling timing ----
        if samp_s is not None:
            samp_s.record()
        x_q = sample_q(args.K, args.batch)
        if samp_e is not None:
            samp_e.record()

        # ---- training timing ----
        if train_s is not None:
            train_s.record()

        f_pos = model(x_p_d).mean()
        f_neg = model(x_q).mean()
        metric = f_pos - f_neg
        loss = -metric

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if train_e is not None:
            train_e.record()
            train_e.synchronize()  # one sync per step for both sampler & train timings
            sampler_total_ms = float(samp_s.elapsed_time(samp_e))
            train_cuda_ms = float(train_s.elapsed_time(train_e))

        it = time.time() - iter_t0

        # ---- logging ----
        if (i % args.log_every) == 0:
            print(
                f"{i:6d}  f(x_p_d)={f_pos.item():>12.6f}  f(x_q)={f_neg.item():>12.6f}  "
                f"metric={metric.item():>12.6f}  loss={loss.item():>12.6f}  iter={it:.3f}s",
                flush=True
            )

        # ---- save images (optional) ----
        if args.save_images_every > 0 and (i % args.save_images_every) == 0:
            save_grid(os.path.join(img_dir, f"x_q_{i:06d}.png"), x_q, nrow=sqrt_int(args.batch))
            if args.save_samples_pt:
                t.save(x_q.detach().cpu(), os.path.join(img_dir, f"x_q_{i:06d}.pt"))

        # ---- write CSV ----
        if args.csv_every > 0 and (i % args.csv_every) == 0:
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    i,
                    f"{f_pos.item():.6f}", f"{f_neg.item():.6f}",
                    f"{metric.item():.6f}", f"{metric.item():.6f}",
                    f"{loss.item():.6f}",
                    f"{it:.6f}",
                    f"{sampler_total_ms:.3f}", f"{train_cuda_ms:.3f}",
                    args.n_f, args.K, args.batch, args.sigma_pd, args.noise_std, args.step_size,
                    args.log_every, args.csv_every, args.save_images_every, int(args.profile_timing),
                ])

    print(f"Done. Total time: {time.time()-start:.2f}s  |  outputs -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
