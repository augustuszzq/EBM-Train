#!/usr/bin/env python3
"""Mode-A synchronized EBM runtime with per-node stage pipeline ring."""
import argparse
import csv
import math
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch as t
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from torchvision import datasets, transforms
except Exception:
    datasets = None
    transforms = None

try:
    from mode_a_contract import (
        MSG_CHAIN,
        dtype_to_id,
        id_to_dtype,
        iter_state_tensors_sorted,
        make_sample_header,
        parse_sample_header,
    )
except Exception:
    from polaris_ebm.scripts.current.mode_a_contract import (
        MSG_CHAIN,
        dtype_to_id,
        id_to_dtype,
        iter_state_tensors_sorted,
        make_sample_header,
        parse_sample_header,
    )

try:
    from ablation_common import (
        completion_ratios,
        compute_completion_aware_alpha,
        compute_k_slices,
    )
except Exception:
    from polaris_ebm.scripts.current.ablation_common import (
        completion_ratios,
        compute_completion_aware_alpha,
        compute_k_slices,
    )

try:
    from benchmark_runtime import (
        build_dataset,
        build_energy_model,
        energy_call,
        resolve_benchmark_runtime,
        unpack_batch,
    )
    from conditional_chain import ChainState
    from conditional_replay import LabelReplayBuffer
    from conditional_sampler import langevin_sample_conditional
except Exception:
    from polaris_ebm.scripts.current.benchmark_runtime import (
        build_dataset,
        build_energy_model,
        energy_call,
        resolve_benchmark_runtime,
        unpack_batch,
    )
    from polaris_ebm.scripts.current.conditional_chain import ChainState
    from polaris_ebm.scripts.current.conditional_replay import LabelReplayBuffer
    from polaris_ebm.scripts.current.conditional_sampler import langevin_sample_conditional


CHAIN_TAG_BASE = 200000
CHAIN_LABEL_TAG_BASE = 300000
CHAIN_META_TAG_BASE = 400000
CHAIN_SIG_TAG_BASE = 500000


def str_to_dtype(name: str) -> t.dtype:
    mapping = {
        "fp32": t.float32,
        "fp16": t.float16,
        "bf16": t.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype string: {name}")
    return mapping[name]


def env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value != "":
            return value
    return default


def setup_rank_env() -> Tuple[int, int, int]:
    rank = int(
        env_first(
            "RANK",
            "OMPI_COMM_WORLD_RANK",
            "PMIX_RANK",
            "PMI_RANK",
            "PALS_RANKID",
            default="0",
        )
    )
    world_size = int(
        env_first(
            "WORLD_SIZE",
            "OMPI_COMM_WORLD_SIZE",
            "PMIX_SIZE",
            "PMI_SIZE",
            "PALS_NTASKS",
            default="1",
        )
    )
    local_rank = int(
        env_first(
            "LOCAL_RANK",
            "OMPI_COMM_WORLD_LOCAL_RANK",
            "PMIX_LOCAL_RANK",
            "MPI_LOCALRANKID",
            "PMI_LOCAL_RANK",
            "PALS_LOCAL_RANKID",
            default="0",
        )
    )
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    return rank, world_size, local_rank


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Mode-A synchronized node-pipeline runtime")
    ap.add_argument(
        "--mode",
        type=str,
        default=os.environ.get("MODE", "baseline"),
        choices=["baseline", "pipeline", "ddp_fullk"],
    )

    ap.add_argument("--model_scale", type=str, default="small", choices=["small", "custom"])
    ap.add_argument("--n_f", type=int, default=64)

    ap.add_argument("--output_dir", type=str, default="./runs_dual")
    ap.add_argument("--run_tag", type=str, default=os.environ.get("RUN_TAG", ""))
    ap.add_argument("--run_dir", type=str, default=os.environ.get("RUN_DIR", ""))
    ap.add_argument("--config", type=str, default=os.environ.get("CONFIG", ""))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data_dir", type=str, default="./data/cifar10")

    ap.add_argument("--batch_size", type=int, default=64, help="global batch across all ranks")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="checkpoint interval in steps (<=0 disables periodic save)",
    )
    ap.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="optional checkpoint directory (default: <run_dir>/checkpoints)",
    )
    ap.add_argument(
        "--resume_ckpt",
        type=str,
        default=os.environ.get("RESUME_CKPT", ""),
        help="optional checkpoint path to resume model/optimizer and step from",
    )
    ap.add_argument(
        "--resume_reset_optimizer",
        action="store_true",
        help="when resuming, load model/chain state but keep a fresh optimizer state",
    )
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="linearly ramp optimizer LR from 0 to --lr over this many global steps",
    )
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument(
        "--grad_clip_norm",
        type=float,
        default=None,
        help="optional explicit global gradient clip norm; overrides --max_grad_norm when set",
    )
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--K", type=int, default=100)
    ap.add_argument(
        "--langevin_sign",
        type=float,
        default=-1.0,
        help="Langevin drift sign: x += sign*step_size*grad + noise; use -1 for energy descent",
    )
    ap.add_argument("--step_size", type=float, default=0.2)
    ap.add_argument("--noise_std", type=float, default=1e-2)
    ap.add_argument(
        "--sampler_prime",
        type=float,
        default=float(os.environ.get("SAMPLER_PRIME", "0.0")),
        help=(
            "optional Langevin prime coefficient; <=0 derives the legacy value "
            "2*step_size/noise_std^2 so existing runs are unchanged"
        ),
    )
    ap.add_argument(
        "--sampler_temperature",
        type=float,
        default=float(os.environ.get("SAMPLER_TEMPERATURE", "1.0")),
        help="positive temperature divisor for sampler_prime drift; default 1 preserves legacy drift",
    )
    ap.add_argument(
        "--energy_loss_scale",
        type=float,
        default=float(os.environ.get("ENERGY_LOSS_SCALE", "1.0")),
        help="multiply the scalar EBM loss for scale-calibration diagnostics; default 1 preserves legacy loss",
    )
    ap.add_argument(
        "--replay_capacity",
        type=int,
        default=int(os.environ.get("REPLAY_CAPACITY", "0")),
        help="conditional replay capacity per local shard; <=0 uses an automatic size, ignored for unconditional runs",
    )
    ap.add_argument(
        "--sample_dtype",
        type=str,
        default=os.environ.get("SAMPLE_DTYPE", "fp16"),
        choices=["fp16", "fp32", "bf16"],
    )
    ap.add_argument("--fresh_init", action="store_true")
    ap.add_argument(
        "--sync_fresh_init",
        action="store_true",
        help="broadcast fresh-init chain values from rank0 so all ranks start from identical tensors",
    )
    ap.add_argument("--no_clamp_x", action="store_true")
    ap.add_argument(
        "--clamp_last_only",
        action="store_true",
        help="if set, clamp x to [-1,1] only after final Langevin step",
    )

    ap.add_argument("--pos_noise_std", type=float, default=0.0)
    ap.add_argument("--no_clamp_pos", action="store_true")

    # Kept for launcher compatibility; no longer used by the runtime.
    ap.add_argument("--n_trainers", type=int, default=-1)

    ap.add_argument(
        "--stages_per_node",
        type=int,
        default=int(os.environ.get("STAGES_PER_NODE", "4")),
        help="pipeline stage count per node; default 4",
    )
    ap.add_argument(
        "--pipe_stages",
        type=int,
        default=int(os.environ.get("PIPE_STAGES", "-1")),
        help="pipeline stage count for the active pipeline group; defaults to stages_per_node or world_size",
    )
    ap.add_argument(
        "--pipe_group_scope",
        type=str,
        default=os.environ.get("PIPE_GROUP_SCOPE", "node"),
        choices=["node", "world"],
        help="pipeline group scope: node preserves canonical behavior; world enables multi-node stage rings",
    )
    ap.add_argument(
        "--weight_mode",
        type=str,
        default=os.environ.get("WEIGHT_MODE", "legacy"),
        choices=["legacy", "uniform", "deep_only", "last2_beta"],
        help="completion-aware weight mode used by ablation runs; legacy preserves existing stage_weight behavior",
    )
    ap.add_argument(
        "--last2_beta",
        type=float,
        default=float(os.environ.get("LAST2_BETA", "0.01")),
        help="beta for weight_mode=last2_beta; deepest stage gets 1-beta",
    )
    ap.add_argument(
        "--stage_weights",
        type=str,
        default=os.environ.get("STAGE_WEIGHTS", ""),
        help="comma-separated per-stage weights; default is all ones",
    )
    ap.add_argument(
        "--stage_weight_mode",
        type=str,
        default=os.environ.get("STAGE_WEIGHT_MODE", "manual"),
        choices=["uniform", "linear", "power", "manual"],
        help="stage weight mode before per-step warmup normalization",
    )
    ap.add_argument(
        "--stage_weight_gamma",
        type=float,
        default=float(os.environ.get("STAGE_WEIGHT_GAMMA", "2.0")),
        help="gamma used when stage_weight_mode=power",
    )
    ap.add_argument(
        "--warmup_gate",
        action="store_true",
        help="set stage weight to 0 when (step-stage) < 0",
    )
    ap.add_argument(
        "--stage2_beta_final",
        type=float,
        default=float(os.environ.get("STAGE2_BETA_FINAL", "0.01")),
        help="final stage2 weight used by optional stage2 beta warmup schedule",
    )
    ap.add_argument(
        "--stage2_beta_warmup",
        action="store_true",
        default=os.environ.get("STAGE2_BETA_WARMUP", "0") == "1",
        help="enable beta schedule: stage2 beta ramps from 0 to stage2_beta_final",
    )
    ap.add_argument(
        "--stage2_beta_warmup_t0",
        type=int,
        default=int(os.environ.get("STAGE2_BETA_WARMUP_T0", "1000")),
        help="warmup beta stays 0 before this step",
    )
    ap.add_argument(
        "--stage2_beta_warmup_t1",
        type=int,
        default=int(os.environ.get("STAGE2_BETA_WARMUP_T1", "2000")),
        help="warmup beta reaches stage2_beta_final at this step",
    )

    ap.add_argument("--dist_timeout_sec", type=int, default=3600)

    ap.add_argument("--debug_level", type=int, default=1, choices=[0, 1, 2])
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument(
        "--diagnostic_first_steps",
        type=int,
        default=0,
        help="force detailed diagnostic logging for every step below this count; <=0 disables",
    )
    ap.add_argument(
        "--diagnostic_log_every",
        type=int,
        default=0,
        help="diagnostic logging interval after diagnostic_first_steps; <=0 uses --log_every",
    )
    ap.add_argument(
        "--crash_abs_energy",
        type=float,
        default=0.0,
        help="early terminate if abs(f_pos) or abs(f_neg_total) exceeds this threshold; <=0 disables",
    )
    ap.add_argument(
        "--crash_max_abs_chain_unclamped",
        type=float,
        default=0.0,
        help="early terminate if unclamped chains exceed this max absolute value; <=0 disables",
    )
    ap.add_argument(
        "--crash_grad_norm",
        type=float,
        default=0.0,
        help="early terminate if global parameter gradient norm exceeds this threshold; <=0 disables",
    )
    ap.add_argument(
        "--soft_abs_energy",
        type=float,
        default=10.0,
        help="mark run unhealthy if abs(f_pos) exceeds this value; <=0 disables",
    )
    ap.add_argument(
        "--soft_grad_norm",
        type=float,
        default=100.0,
        help="mark run unhealthy if pre-clip grad norm exceeds this value; <=0 disables",
    )
    ap.add_argument(
        "--hard_grad_norm",
        type=float,
        default=500.0,
        help="mark run hard-unstable if pre-clip grad norm exceeds this value; <=0 disables",
    )
    ap.add_argument(
        "--clip_active_fraction",
        type=float,
        default=0.25,
        help="mark run unhealthy if clipping is active on more than this cumulative fraction of optimizer updates",
    )
    ap.add_argument(
        "--soft_guard_stop",
        action="store_true",
        help="terminate when the soft stability guard reports any reason",
    )
    ap.add_argument(
        "--skip_optimizer_until_full_diagonal",
        action="store_true",
        help="pipeline only: fill the diagonal chain cache before the first optimizer update",
    )
    ap.add_argument(
        "--vis_every",
        type=int,
        default=200,
        help="save training visualization every N steps (<=0 disables)",
    )
    ap.add_argument("--vis_num", type=int, default=64)
    ap.add_argument("--vis_nrow", type=int, default=8)
    ap.add_argument(
        "--vis_stage",
        type=int,
        default=-1,
        help="pipeline stage to visualize; -1 means last stage",
    )
    ap.add_argument(
        "--vis_dir",
        type=str,
        default="",
        help="optional visualization output directory (default: <run_dir>/vis)",
    )
    return ap.parse_args()


class EnergyModel(nn.Module):
    def __init__(self, n_c: int = 3, n_f: int = 64, leak: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),
            nn.LeakyReLU(leak, inplace=True),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),
            nn.LeakyReLU(leak, inplace=True),
            nn.Conv2d(n_f * 2, n_f * 4, 4, 2, 1),
            nn.LeakyReLU(leak, inplace=True),
            nn.Conv2d(n_f * 4, n_f * 8, 4, 2, 1),
            nn.LeakyReLU(leak, inplace=True),
            nn.Conv2d(n_f * 8, 1, 4, 1, 0),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.net(x).view(x.size(0))


def build_cifar10(data_dir: str):
    if datasets is None or transforms is None:
        raise ImportError("torchvision is required for CIFAR10")
    tfm = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return datasets.CIFAR10(root=data_dir, train=True, download=False, transform=tfm)


def iter_dataloader(dl: t.utils.data.DataLoader):
    while True:
        for batch in dl:
            yield batch


def broadcast_model_state(module: nn.Module, src_rank: int, group) -> None:
    for _, tensor in iter_state_tensors_sorted(module):
        dist.broadcast(tensor, src=src_rank, group=group)


def fill_shared_chain_init(
    dst: t.Tensor,
    rank: int,
    src_rank: int,
    group,
    scratch: t.Tensor = None,
) -> None:
    buf = dst if scratch is None else scratch
    if rank == src_rank:
        buf.uniform_(-1.0, 1.0)
    dist.broadcast(buf, src=src_rank, group=group)
    if buf.data_ptr() != dst.data_ptr():
        dst.copy_(buf)


def langevin_sample(
    model: nn.Module,
    chain: t.Tensor,
    k_steps: int,
    langevin_sign: float,
    step_size: float,
    noise_std: float,
    clamp_x: bool,
    clamp_last_only: bool = False,
    drift_coeff: float | None = None,
) -> t.Tensor:
    x = chain.detach()
    coeff = float(step_size) if drift_coeff is None else float(drift_coeff)
    for t_step in range(k_steps):
        # Force-enable grads for dE/dx even if caller wraps outer scope in no_grad.
        with t.enable_grad():
            x = x.detach().requires_grad_(True)
            score = model(x).sum()
            grad = t.autograd.grad(score, x, retain_graph=False, create_graph=False)[0]
        x = x.detach()
        x.add_(langevin_sign * coeff * grad).add_(noise_std * t.randn_like(x))
        do_clamp = clamp_x and ((not clamp_last_only) or (t_step == (k_steps - 1)))
        if do_clamp:
            x.clamp_(-1, 1)
    chain.copy_(x)
    return x


def should_log(step: int, debug_level: int, log_every: int) -> bool:
    if debug_level <= 0:
        return False
    if debug_level >= 2:
        return True
    return (step % max(1, log_every)) == 0


def should_diagnostic_log(
    step: int,
    debug_level: int,
    log_every: int,
    diagnostic_first_steps: int = 0,
    diagnostic_log_every: int = 0,
) -> bool:
    if int(diagnostic_first_steps) > 0 and int(step) < int(diagnostic_first_steps):
        return True
    interval = int(diagnostic_log_every) if int(diagnostic_log_every) > 0 else int(log_every)
    return should_log(step=step, debug_level=debug_level, log_every=interval)


def active_stage_count_from_alpha(alpha_vec: Sequence[float]) -> int:
    return sum(1 for value in alpha_vec if abs(float(value)) > 1e-12)


def resolve_langevin_calibration(
    *,
    step_size: float,
    noise_std: float,
    sampler_prime: float,
    sampler_temperature: float,
) -> Dict[str, float]:
    """Resolve classic prime/noise coupling without changing legacy defaults.

    Current historical runs used `step_size` directly as the drift coefficient.
    The equivalent prime under `drift = 0.5 * prime * noise_std^2` is
    `2 * step_size / noise_std^2`. Supplying `sampler_prime` makes that
    coefficient explicit for DRL-backbone scale diagnostics.
    """
    step = float(step_size)
    noise = float(noise_std)
    temp = float(sampler_temperature)
    explicit_prime = float(sampler_prime)
    if not math.isfinite(step) or step < 0.0:
        raise ValueError("step_size must be finite and >= 0")
    if not math.isfinite(noise) or noise < 0.0:
        raise ValueError("noise_std must be finite and >= 0")
    if not math.isfinite(temp) or temp <= 0.0:
        raise ValueError("sampler_temperature must be finite and > 0")
    if not math.isfinite(explicit_prime):
        raise ValueError("sampler_prime must be finite")
    if explicit_prime > 0.0:
        prime = explicit_prime
    elif noise > 0.0:
        prime = 2.0 * step / (noise * noise)
    else:
        prime = 0.0
    effective_prime = prime / temp
    drift_coeff = 0.5 * effective_prime * noise * noise
    if noise == 0.0 and explicit_prime <= 0.0:
        drift_coeff = step / temp
    return {
        "sampler_prime": float(prime),
        "sampler_temperature": float(temp),
        "effective_sampler_prime": float(effective_prime),
        "langevin_drift_coeff": float(drift_coeff),
    }


def active_stage_set_from_alpha(alpha_vec: Sequence[float]) -> str:
    active = [str(idx) for idx, value in enumerate(alpha_vec) if abs(float(value)) > 1e-12]
    return ",".join(active) if active else "none"


def should_skip_optimizer_for_pipeline_warmup(
    step: int,
    pipe_stages: int,
    enabled: bool,
) -> bool:
    return bool(enabled) and int(step) < (int(pipe_stages) - 1)


def lr_warmup_factor(step: int, lr_warmup_steps: int) -> float:
    warmup = int(lr_warmup_steps)
    if warmup <= 0:
        return 1.0
    return min(1.0, float(int(step) + 1) / float(warmup))


def apply_lr_warmup(
    optimizer: t.optim.Optimizer,
    *,
    base_lr: float,
    step: int,
    lr_warmup_steps: int,
) -> float:
    lr = float(base_lr) * lr_warmup_factor(step=step, lr_warmup_steps=lr_warmup_steps)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def parameter_grad_norm(parameters: Iterable[t.nn.Parameter]) -> float:
    total_sq = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if not t.isfinite(grad).all().item():
            return float("inf")
        norm = float(grad.float().norm(2).item())
        total_sq += norm * norm
    return float(math.sqrt(total_sq))


def parameter_norm(parameters: Iterable[t.nn.Parameter]) -> float:
    total_sq = 0.0
    saw_param = False
    for param in parameters:
        saw_param = True
        value = param.detach()
        if not t.isfinite(value).all().item():
            return float("inf")
        norm = float(value.float().norm(2).item())
        total_sq += norm * norm
    if not saw_param:
        return float("nan")
    return float(math.sqrt(total_sq))


def is_energy_head_parameter(name: str) -> bool:
    parts = str(name).split(".")
    return "head" in parts


def named_parameter_norm(
    named_parameters: Iterable[Tuple[str, t.nn.Parameter]],
    *,
    include_head: bool,
) -> float:
    params = [
        param
        for name, param in named_parameters
        if is_energy_head_parameter(name) == bool(include_head)
    ]
    return parameter_norm(params)


def named_parameter_grad_norm(
    named_parameters: Iterable[Tuple[str, t.nn.Parameter]],
    *,
    include_head: bool,
) -> float:
    params = [
        param
        for name, param in named_parameters
        if is_energy_head_parameter(name) == bool(include_head)
    ]
    return parameter_grad_norm(params)


def clone_trainable_parameters(parameters: Iterable[t.nn.Parameter]) -> List[t.Tensor]:
    return [param.detach().clone() for param in parameters if param.requires_grad]


def parameter_update_norm(
    before: Iterable[t.Tensor],
    parameters: Iterable[t.nn.Parameter],
) -> float:
    total_sq = 0.0
    saw_param = False
    for before_param, after_param in zip(before, (p for p in parameters if p.requires_grad)):
        saw_param = True
        delta = after_param.detach() - before_param
        if not t.isfinite(delta).all().item():
            return float("inf")
        norm = float(delta.float().norm(2).item())
        total_sq += norm * norm
    if not saw_param:
        return float("nan")
    return float(math.sqrt(total_sq))


def evaluate_stability_crash_guard(
    *,
    f_pos: float,
    f_neg_total: float,
    max_abs_chain: float,
    clamp_x: bool,
    grad_norm_theta: float,
    crash_abs_energy: float,
    crash_max_abs_chain_unclamped: float,
    crash_grad_norm: float,
) -> List[str]:
    reasons: List[str] = []
    if not math.isfinite(float(f_pos)):
        reasons.append("f_pos is non-finite")
    if not math.isfinite(float(f_neg_total)):
        reasons.append("f_neg_total is non-finite")
    if not math.isfinite(float(max_abs_chain)):
        reasons.append("max_abs_chain is non-finite")
    if not math.isfinite(float(grad_norm_theta)):
        reasons.append("grad_norm_theta is non-finite")

    if float(crash_abs_energy) > 0.0:
        threshold = float(crash_abs_energy)
        if abs(float(f_pos)) > threshold:
            reasons.append("abs(f_pos)>%.1f" % threshold)
        if abs(float(f_neg_total)) > threshold:
            reasons.append("abs(f_neg_total)>%.1f" % threshold)
    if (not bool(clamp_x)) and float(crash_max_abs_chain_unclamped) > 0.0:
        threshold = float(crash_max_abs_chain_unclamped)
        if abs(float(max_abs_chain)) > threshold:
            reasons.append("max_abs_chain>%.1f with clamp_x=false" % threshold)
    if float(crash_grad_norm) > 0.0:
        threshold = float(crash_grad_norm)
        if abs(float(grad_norm_theta)) > threshold:
            reasons.append("grad_norm_theta>%.1f" % threshold)
    return reasons


def evaluate_soft_stability_guard(
    *,
    f_pos: float,
    grad_norm_pre_clip: float,
    grad_clip_active_fraction: float,
    soft_abs_energy: float,
    soft_grad_norm: float,
    hard_grad_norm: float,
    clip_active_fraction: float,
) -> List[str]:
    reasons: List[str] = []
    if float(soft_abs_energy) > 0.0 and abs(float(f_pos)) > float(soft_abs_energy):
        reasons.append("soft_abs(f_pos)>%.1f" % float(soft_abs_energy))
    if float(soft_grad_norm) > 0.0 and float(grad_norm_pre_clip) > float(soft_grad_norm):
        reasons.append("soft_grad_norm_pre_clip>%.1f" % float(soft_grad_norm))
    if float(hard_grad_norm) > 0.0 and float(grad_norm_pre_clip) > float(hard_grad_norm):
        reasons.append("hard_grad_norm_pre_clip>%.1f" % float(hard_grad_norm))
    if (
        0.0 <= float(clip_active_fraction) <= 1.0
        and float(grad_clip_active_fraction) > float(clip_active_fraction)
    ):
        reasons.append("clip_active_fraction>%.2f" % float(clip_active_fraction))
    return reasons


def chain_tags(step: int, stage: int, num_stages: int) -> Tuple[int, int]:
    header_tag = CHAIN_TAG_BASE + (step * (2 * num_stages)) + (2 * stage)
    payload_tag = header_tag + 1
    return header_tag, payload_tag


def chain_label_tag(step: int, stage: int, num_stages: int) -> int:
    return CHAIN_LABEL_TAG_BASE + (step * num_stages) + stage


def chain_meta_tag(step: int, stage: int, num_stages: int, field_offset: int) -> int:
    return CHAIN_META_TAG_BASE + (step * num_stages * 3) + (stage * 3) + field_offset


def chain_sig_tag(step: int, stage: int, num_stages: int) -> int:
    return CHAIN_SIG_TAG_BASE + (step * num_stages) + stage


def _clone_cpu_or_none(x: Optional[t.Tensor]) -> Optional[t.Tensor]:
    if x is None:
        return None
    return x.detach().cpu().clone()


def rank_state_ckpt_path(main_ckpt_path: str, rank: int) -> str:
    ckpt_path = os.path.abspath(main_ckpt_path)
    ckpt_dir = os.path.dirname(ckpt_path)
    rank_dir = os.path.join(ckpt_dir, "rank_state")
    base = os.path.basename(ckpt_path)
    stem, ext = os.path.splitext(base)
    return os.path.join(rank_dir, f"{stem}_rank{int(rank)}{ext}")


def state_to_cpu_payload(state: Optional[ChainState]) -> Optional[Dict]:
    if state is None:
        return None
    state = state.validate()
    return {
        "x": state.x.detach().cpu().clone(),
        "y": state.y.detach().cpu().clone(),
        "chain_id": _clone_cpu_or_none(state.chain_id),
        "steps_done": _clone_cpu_or_none(state.steps_done),
        "stage_id": state.stage_id,
        "valid": _clone_cpu_or_none(state.valid),
    }


def chain_state_from_buffers(
    *,
    x: Optional[t.Tensor],
    y: Optional[t.Tensor],
    chain_id: Optional[t.Tensor],
    steps_done: Optional[t.Tensor],
    valid: Optional[t.Tensor],
    stage_id: Optional[int],
) -> Optional[ChainState]:
    if x is None or y is None:
        return None
    return ChainState(
        x=x.detach(),
        y=y.detach().long(),
        chain_id=None if chain_id is None else chain_id.detach().long(),
        steps_done=None if steps_done is None else steps_done.detach().long(),
        stage_id=stage_id,
        valid=None if valid is None else valid.detach().bool(),
    ).validate()


def restore_chain_state_into_buffers(
    state_payload: Optional[Dict],
    *,
    device: t.device,
    chain_buf: Optional[t.Tensor],
    label_buf: Optional[t.Tensor],
    chain_id_buf: Optional[t.Tensor],
    steps_done_buf: Optional[t.Tensor],
    valid_buf: Optional[t.Tensor],
) -> bool:
    if not state_payload:
        return False
    state = ChainState.from_payload(state_payload).to(device)
    if chain_buf is not None:
        chain_buf.copy_(state.x)
    if label_buf is not None:
        label_buf.copy_(state.y)
    if chain_id_buf is not None and state.chain_id is not None:
        chain_id_buf.copy_(state.chain_id)
    if steps_done_buf is not None and state.steps_done is not None:
        steps_done_buf.copy_(state.steps_done)
    if valid_buf is not None and state.valid is not None:
        valid_buf.copy_(state.valid)
    return True


def conditional_state_signature(
    labels: t.Tensor,
    chain_id: Optional[t.Tensor],
    steps_done: Optional[t.Tensor],
    valid: Optional[t.Tensor],
) -> t.Tensor:
    sig = ((labels.long() + 1) * 1_000_003).sum(dtype=t.int64)
    if chain_id is not None:
        sig = sig + ((chain_id.long() + 2) * 1_000_033).sum(dtype=t.int64)
    if steps_done is not None:
        sig = sig + ((steps_done.long() + 3) * 1_000_211).sum(dtype=t.int64)
    if valid is not None:
        sig = sig + ((valid.long() + 5) * 1_000_433).sum(dtype=t.int64)
    return sig.view(1)


def assert_pipeline_received_state(
    *,
    x: t.Tensor,
    y: t.Tensor,
    chain_id: Optional[t.Tensor],
    steps_done: Optional[t.Tensor],
    valid: Optional[t.Tensor],
    expected_signature: Optional[t.Tensor],
    num_classes: int,
    min_steps_done: int,
    rank: int,
    stage: int,
    step: int,
) -> None:
    state = chain_state_from_buffers(
        x=x,
        y=y,
        chain_id=chain_id,
        steps_done=steps_done,
        valid=valid,
        stage_id=stage,
    )
    if state is None:
        raise RuntimeError("rank %d stage %d step %d missing conditional state" % (rank, stage, step))
    if int(num_classes) > 0:
        if bool((state.y < 0).any().item()) or bool((state.y >= int(num_classes)).any().item()):
            raise RuntimeError(
                "rank %d stage %d step %d received label out of range [0,%d)"
                % (rank, stage, step, int(num_classes))
            )
    if state.steps_done is not None and state.valid is not None and int(min_steps_done) > 0:
        active_mask = state.valid
        if bool(active_mask.any().item()):
            min_seen = int(state.steps_done[active_mask].min().item())
            if min_seen < int(min_steps_done):
                raise RuntimeError(
                    "rank %d stage %d step %d received steps_done=%d smaller than expected minimum %d"
                    % (rank, stage, step, min_seen, int(min_steps_done))
                )
    if expected_signature is not None:
        actual_sig = conditional_state_signature(state.y, state.chain_id, state.steps_done, state.valid)
        if int(actual_sig.item()) != int(expected_signature.view(-1)[0].item()):
            raise RuntimeError(
                "rank %d stage %d step %d conditional state signature mismatch expected=%d actual=%d"
                % (
                    rank,
                    stage,
                    step,
                    int(expected_signature.view(-1)[0].item()),
                    int(actual_sig.item()),
                )
            )


def normalize_local_rank(local_rank: int) -> Tuple[int, str]:
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not vis:
        return local_rank, ""
    visible = [x.strip() for x in vis.split(",") if x.strip()]
    if len(visible) == 1 and local_rank != 0:
        return 0, "single_visible_device"
    if len(visible) > 0 and local_rank >= len(visible):
        return local_rank % len(visible), "local_rank_out_of_visible_range"
    return local_rank, ""


def print_env_snapshot(rank: int) -> None:
    keys = [
        "MASTER_ADDR",
        "MASTER_PORT",
        "NCCL_NET",
        "NCCL_DEBUG",
        "NCCL_DEBUG_SUBSYS",
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_DISABLE",
        "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES",
    ]
    for key in keys:
        value = os.environ.get(key, "<unset>")
        if key == "LD_LIBRARY_PATH" and len(value) > 240:
            value = value[:240] + "...(truncated)"
        print("[ENV rank%d] %s=%s" % (rank, key, value), flush=True)


def resolve_local_world_size(default_ppn: int) -> int:
    local_world_size = int(
        env_first(
            "LOCAL_WORLD_SIZE",
            "OMPI_COMM_WORLD_LOCAL_SIZE",
            "PMIX_LOCAL_SIZE",
            "MPI_LOCALNRANKS",
            "PMI_LOCAL_SIZE",
            "MPICH_LOCAL_SIZE",
            "PALS_LOCAL_SIZE",
            default=str(default_ppn),
        )
    )
    if local_world_size <= 0:
        raise RuntimeError(f"LOCAL_WORLD_SIZE must be > 0, got {local_world_size}")
    return local_world_size


def create_node_pipeline_group(world_size: int, local_world_size: int, rank: int):
    if world_size % local_world_size != 0:
        raise RuntimeError(
            "world_size %d must be divisible by local_world_size %d" % (world_size, local_world_size)
        )
    num_nodes = world_size // local_world_size
    node_groups = []
    for n in range(num_nodes):
        ranks_n = list(range(n * local_world_size, (n + 1) * local_world_size))
        node_groups.append(dist.new_group(ranks=ranks_n, backend="nccl"))
    node_id = rank // local_world_size
    return node_groups[node_id], node_id, num_nodes


def parse_stage_weights(spec: str, num_stages: int) -> List[float]:
    text = spec.strip()
    if not text:
        return [1.0 for _ in range(num_stages)]
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    tokens = [tok.strip() for tok in text.split(",") if tok.strip()]
    if not tokens:
        return [1.0 for _ in range(num_stages)]
    values = [float(tok) for tok in tokens]
    if len(values) == 1:
        return values * num_stages
    if len(values) != num_stages:
        raise RuntimeError(
            "stage_weights must provide 1 or %d values, got %d" % (num_stages, len(values))
        )
    return values


def format_stage_weights(weights: Sequence[float]) -> str:
    return ",".join("%.6f" % x for x in weights)


def resolve_raw_stage_weights(
    stage_weight_mode: str,
    manual_weights: Sequence[float],
    num_stages: int,
    gamma: float,
) -> List[float]:
    if num_stages <= 0:
        raise RuntimeError("num_stages must be > 0")
    if stage_weight_mode == "manual":
        if len(manual_weights) != num_stages:
            raise RuntimeError(
                "manual stage_weights length mismatch: expected %d got %d"
                % (num_stages, len(manual_weights))
            )
        return [float(x) for x in manual_weights]
    if stage_weight_mode == "uniform":
        return [1.0 for _ in range(num_stages)]
    if stage_weight_mode == "linear":
        return [float(s + 1) / float(num_stages) for s in range(num_stages)]
    if stage_weight_mode == "power":
        return [(float(s + 1) / float(num_stages)) ** float(gamma) for s in range(num_stages)]
    raise RuntimeError(f"unsupported stage_weight_mode={stage_weight_mode}")


def compute_stage_alpha_vector(
    step: int,
    raw_stage_weights: Sequence[float],
    warmup_gate: bool,
) -> List[float]:
    num_stages = len(raw_stage_weights)
    gated = []
    for s in range(num_stages):
        valid = (step - s) >= 0
        gated.append(float(raw_stage_weights[s]) if (not warmup_gate or valid) else 0.0)
    total = sum(gated)
    if total <= 0.0:
        return [0.0 for _ in range(num_stages)]
    return [w / total for w in gated]


def stage2_beta_schedule(step: int, beta_final: float, t0: int, t1: int) -> float:
    if t1 <= t0:
        raise RuntimeError("stage2_beta_warmup_t1 must be > stage2_beta_warmup_t0")
    if step < t0:
        return 0.0
    if step < t1:
        return float(beta_final) * float(step - t0) / float(t1 - t0)
    return float(beta_final)


def tensor_to_01(x_m11: t.Tensor) -> t.Tensor:
    return (x_m11 * 0.5 + 0.5).clamp(0.0, 1.0)


def reduce_mean_scalar(value: float, world_size: int, group, device: t.device) -> float:
    x = t.tensor([float(value)], dtype=t.float32, device=device)
    dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
    return float((x / float(world_size)).item())


def reduce_max_scalar(value: float, group, device: t.device) -> float:
    x = t.tensor([float(value)], dtype=t.float32, device=device)
    dist.all_reduce(x, op=dist.ReduceOp.MAX, group=group)
    return float(x.item())


def reduce_min_scalar(value: float, group, device: t.device) -> float:
    x = t.tensor([float(value)], dtype=t.float32, device=device)
    dist.all_reduce(x, op=dist.ReduceOp.MIN, group=group)
    return float(x.item())


def reduce_chain_summary(x: t.Tensor, world_size: int, group) -> Tuple[float, float, float, float]:
    x_det = x.detach()
    mean_t = x_det.mean().float()
    std_t = x_det.std(unbiased=False).float()
    min_t = x_det.min().float()
    max_t = x_det.max().float()

    dist.all_reduce(mean_t, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(std_t, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(min_t, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(max_t, op=dist.ReduceOp.MAX, group=group)

    mean_val = float((mean_t / float(world_size)).item())
    std_val = float((std_t / float(world_size)).item())
    min_val = float(min_t.item())
    max_val = float(max_t.item())
    return min_val, max_val, mean_val, std_val


def collect_stage_diagnostic_vectors(
    *,
    stage_index: int,
    num_stage_slots: int,
    f_neg: float,
    max_f_neg: float,
    max_abs_chain: float,
    mean_abs_chain: float,
    clamp_sat_frac: float,
    alpha_bar: float,
    grad_norm_x: float,
    group,
    world_size: int,
    device: t.device,
) -> Dict[str, List[float]]:
    slots = max(1, int(num_stage_slots))
    local = t.tensor(
        [
            float(stage_index),
            float(f_neg),
            float(max_f_neg),
            float(max_abs_chain),
            float(mean_abs_chain),
            float(clamp_sat_frac),
            float(alpha_bar),
            float(grad_norm_x),
        ],
        dtype=t.float32,
        device=device,
    )
    gathered = [t.empty_like(local) for _ in range(int(world_size))]
    dist.all_gather(gathered, local, group=group)

    f_neg_sum = [0.0 for _ in range(slots)]
    mean_abs_sum = [0.0 for _ in range(slots)]
    count = [0 for _ in range(slots)]
    max_f_neg_vals = [float("nan") for _ in range(slots)]
    max_abs_vals = [float("nan") for _ in range(slots)]
    clamp_sat_sum = [0.0 for _ in range(slots)]
    alpha_vals = [float("nan") for _ in range(slots)]
    grad_x_vals = [float("nan") for _ in range(slots)]
    for row in gathered:
        idx = int(row[0].item())
        if idx < 0 or idx >= slots:
            continue
        f_neg_sum[idx] += float(row[1].item())
        mean_abs_sum[idx] += float(row[4].item())
        clamp_sat_sum[idx] += float(row[5].item())
        count[idx] += 1
        max_f_neg_vals[idx] = max(float(row[2].item()), max_f_neg_vals[idx]) if math.isfinite(max_f_neg_vals[idx]) else float(row[2].item())
        max_abs_vals[idx] = max(float(row[3].item()), max_abs_vals[idx]) if math.isfinite(max_abs_vals[idx]) else float(row[3].item())
        alpha_vals[idx] = max(float(row[6].item()), alpha_vals[idx]) if math.isfinite(alpha_vals[idx]) else float(row[6].item())
        grad_x = float(row[7].item())
        if math.isfinite(grad_x):
            grad_x_vals[idx] = max(grad_x, grad_x_vals[idx]) if math.isfinite(grad_x_vals[idx]) else grad_x

    f_neg_vals = [float("nan") for _ in range(slots)]
    mean_abs_vals = [float("nan") for _ in range(slots)]
    clamp_sat_vals = [float("nan") for _ in range(slots)]
    for idx in range(slots):
        if count[idx] > 0:
            f_neg_vals[idx] = f_neg_sum[idx] / float(count[idx])
            mean_abs_vals[idx] = mean_abs_sum[idx] / float(count[idx])
            clamp_sat_vals[idx] = clamp_sat_sum[idx] / float(count[idx])
    return {
        "f_neg_stage": f_neg_vals,
        "E_neg_stage": f_neg_vals,
        "max_f_neg_stage": max_f_neg_vals,
        "max_abs_chain_stage": max_abs_vals,
        "mean_abs_chain_stage": mean_abs_vals,
        "clamp_sat_frac_stage": clamp_sat_vals,
        "alpha_bar_stage": alpha_vals,
        "grad_norm_x_stage": grad_x_vals,
    }


def clamp_saturation_fraction(x: t.Tensor, eps: float = 1e-6) -> float:
    x_det = x.detach().float()
    return float((x_det.abs() >= (1.0 - float(eps))).float().mean().item())


def local_chain_summary(
    x: t.Tensor,
    unique_sample_cap: int = 65536,
) -> Tuple[float, float, float, float, float, float, int, str]:
    x_det = x.detach().float()
    flat = x_det.reshape(-1)
    numel = int(flat.numel())
    if numel > int(unique_sample_cap):
        stride = max(1, numel // int(unique_sample_cap))
        probe = flat[::stride]
        unique_mode = "sampled"
    else:
        probe = flat
        unique_mode = "exact"
    unique_count = int(t.unique(probe).numel())
    frac_zero = float((flat == 0).float().mean().item())
    frac_one = float((flat == 1).float().mean().item())
    return (
        float(x_det.min().item()),
        float(x_det.max().item()),
        float(x_det.mean().item()),
        float(x_det.std(unbiased=False).item()),
        frac_zero,
        frac_one,
        unique_count,
        unique_mode,
    )


def build_stage_diag_columns(num_stage_slots: int) -> List[str]:
    columns: List[str] = []
    for diag_s in range(max(1, int(num_stage_slots))):
        columns.extend(
            [
                "f_neg_stage%d" % diag_s,
                "E_neg_stage%d" % diag_s,
                "max_f_neg_stage%d" % diag_s,
                "max_abs_chain_stage%d" % diag_s,
                "mean_abs_chain_stage%d" % diag_s,
                "clamp_sat_frac_stage%d" % diag_s,
                "alpha_bar_stage%d" % diag_s,
                "grad_norm_x_stage%d" % diag_s,
            ]
        )
    return columns


def build_metrics_columns(num_stage_slots: int) -> List[str]:
    return [
        "step",
        "wait_ms",
        "train_ms",
        "iter_ms",
        "comm_ms",
        "comm_pct",
        "wallclock_sec",
        "gpu_hours",
        "metric",
        "f_pos",
        "f_neg",
        "objective",
        "loss",
        "lr",
        "E_pos",
        "E_neg",
        "E_neg_total",
        "E_neg_minus_E_pos_gap",
        "min_energy",
        "max_energy",
        "is_finite",
        "max_f_neg_across_ranks",
        "max_abs_chain_across_ranks",
        "clamp_sat_frac",
        "stage",
        "w_stage",
        "w_effective",
        "alpha_stage",
        "c_stage",
        "mean_c_stage",
        "sum_alpha_active",
        "beta_t",
        "alpha_stage2",
        "alpha_stage3",
        "chain_min",
        "chain_max",
        "chain_mean",
        "chain_std",
        "chain_frac_zero",
        "chain_frac_one",
        "chain_unique_count",
        "token_i_minus_stage",
        "f_neg_total",
        "loss_pos",
        "loss_neg",
        "loss_total",
        "grad_norm_theta",
        "grad_norm_pre_clip",
        "grad_norm_post_clip",
        "param_norm",
        "update_norm",
        "update_to_param_ratio",
        "energy_head_weight_norm",
        "energy_head_grad_norm",
        "backbone_grad_norm",
        "clip_active",
        "clip_active_fraction",
        "sampler_prime",
        "sampler_temperature",
        "effective_sampler_prime",
        "langevin_drift_coeff",
        "energy_loss_scale",
        "soft_guard_reason",
        "optimizer_update",
        "active_stage_count",
        "active_stage_set",
        "crash_guard_reason",
    ] + build_stage_diag_columns(num_stage_slots)


def save_grid_png(x_m11: t.Tensor, path: str, nrow: int) -> None:
    # Use lazy import to keep startup cost low for non-visualization runs.
    from torchvision.utils import save_image

    save_image(tensor_to_01(x_m11.detach().float().cpu()), path, nrow=nrow)


def maybe_save_checkpoint(
    rank: int,
    step: int,
    args: argparse.Namespace,
    run_dir: str,
    ddp_model: DDP,
    optimizer: t.optim.Optimizer,
    extra_state: Optional[Dict] = None,
) -> None:
    if int(args.save_every) <= 0:
        return
    should_save = ((step + 1) % int(args.save_every)) == 0 or (step == int(args.steps) - 1)
    if not should_save:
        return
    ckpt_dir = args.save_dir.strip() if args.save_dir.strip() else os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "ckpt_step%d.pt" % step)
    payload = {
        "model_state_dict": ddp_model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": int(step),
        "args": vars(args),
    }
    if extra_state is not None:
        payload["extra_state"] = extra_state
    if rank == 0:
        t.save(payload, ckpt_path)
        print("[CKPT] saved %s" % ckpt_path, flush=True)
    if extra_state is not None:
        local_path = rank_state_ckpt_path(ckpt_path, rank=rank)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        t.save(payload, local_path)
        if rank == 0:
            print("[CKPT] saved %s" % local_path, flush=True)


def maybe_resume_from_checkpoint(
    args: argparse.Namespace,
    model: nn.Module,
    optimizer: t.optim.Optimizer,
    device: t.device,
    rank: int,
    extra_state_out: Optional[Dict] = None,
) -> int:
    resume_ckpt = str(getattr(args, "resume_ckpt", "")).strip()
    if not resume_ckpt:
        return 0
    if not os.path.exists(resume_ckpt):
        raise RuntimeError("resume checkpoint not found: %s" % resume_ckpt)
    payload = t.load(resume_ckpt, map_location=device)
    local_rank_ckpt = rank_state_ckpt_path(resume_ckpt, rank=rank)
    if os.path.exists(local_rank_ckpt):
        payload = t.load(local_rank_ckpt, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    if not bool(getattr(args, "resume_reset_optimizer", False)):
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if extra_state_out is not None:
        extra_state_out.clear()
        extra_state_out.update(payload.get("extra_state") or {})
    resume_step = int(payload.get("step", -1)) + 1
    total_steps = int(getattr(args, "steps", 0))
    if resume_step < 0:
        raise RuntimeError("invalid resume step in checkpoint: %s" % resume_ckpt)
    if total_steps > 0 and resume_step >= total_steps:
        raise RuntimeError(
            "resume step %d must be < total steps %d for checkpoint %s"
            % (resume_step, total_steps, resume_ckpt)
        )
    if rank == 0:
        print(
            "[RESUME] loaded %s start_step=%d reset_optimizer=%d"
            % (
                local_rank_ckpt if os.path.exists(local_rank_ckpt) else resume_ckpt,
                resume_step,
                1 if bool(getattr(args, "resume_reset_optimizer", False)) else 0,
            ),
            flush=True,
        )
    return resume_step


def main() -> None:
    args = parse_args()
    runtime = resolve_benchmark_runtime(config_path=str(args.config), data_dir=args.data_dir)
    rank, world_size, local_rank = setup_rank_env()
    host = os.environ.get("HOSTNAME", "")
    if not host:
        host = os.uname().nodename
    local_rank_resolved, adjust_reason = normalize_local_rank(local_rank)
    if local_rank_resolved != local_rank:
        print(
            "[RANKMAP-ADJUST] host=%s rank=%d local_rank=%d -> %d reason=%s cuda_visible_devices=%s"
            % (
                host,
                rank,
                local_rank,
                local_rank_resolved,
                adjust_reason,
                os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
            ),
            flush=True,
        )
        local_rank = local_rank_resolved
        os.environ["LOCAL_RANK"] = str(local_rank)
    if "MASTER_ADDR" not in os.environ or "MASTER_PORT" not in os.environ:
        raise RuntimeError("MASTER_ADDR and MASTER_PORT must be set by launcher")
    if rank == 0 and (args.debug_level >= 2 or int(os.environ.get("DEBUG_LAUNCHER", "0")) >= 1):
        print_env_snapshot(rank=rank)

    if not t.cuda.is_available():
        raise RuntimeError("CUDA is required for mode-A NCCL runtime")
    try:
        t.cuda.set_device(local_rank)
    except RuntimeError as exc:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        dev_count = t.cuda.device_count()
        print(
            "[CUDA-SET-DEVICE-FAIL] host=%s rank=%d local_rank=%d world_size=%d "
            "cuda_visible_devices=%s device_count=%d error=%s"
            % (host, rank, local_rank, world_size, vis, dev_count, str(exc)),
            flush=True,
        )
        raise

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=int(args.dist_timeout_sec)),
    )

    all_group = dist.group.WORLD
    trainer_root = 0
    use_pipeline = args.mode == "pipeline"

    if args.batch_size <= 0:
        raise RuntimeError(f"batch_size must be > 0, got {args.batch_size}")
    if args.batch_size % world_size != 0:
        raise RuntimeError(
            "global batch_size (%d) must be divisible by world_size (%d)"
            % (args.batch_size, world_size)
        )
    local_batch = args.batch_size // world_size

    local_world_size = resolve_local_world_size(default_ppn=max(1, int(args.stages_per_node)))
    if local_rank >= local_world_size:
        raise RuntimeError(
            "local_rank %d must be < local_world_size %d" % (local_rank, local_world_size)
        )

    pipe_group = None
    node_id = rank // max(1, local_world_size)
    num_nodes = max(1, world_size // max(1, local_world_size))
    stage = local_rank
    num_stages = local_world_size
    pipe_scope = str(args.pipe_group_scope)
    next_stage = -1
    prev_stage = -1
    next_rank = -1
    prev_rank = -1
    k_slice = int(args.K)
    k_slices = [int(args.K)]
    stage_completion_ratio = 1.0
    stage_weights = [1.0 for _ in range(max(1, num_stages))]
    raw_stage_weights = [1.0 for _ in range(max(1, num_stages))]
    w_stage = 1.0

    if use_pipeline:
        if pipe_scope == "world":
            pipe_group = all_group
            stage = rank
            num_stages = int(args.pipe_stages) if int(args.pipe_stages) > 0 else int(world_size)
            if num_stages != int(world_size):
                raise RuntimeError(
                    "pipe_group_scope=world requires pipe_stages (%d) == world_size (%d)"
                    % (num_stages, int(world_size))
                )
            next_stage = (stage + 1) % num_stages
            prev_stage = (stage - 1 + num_stages) % num_stages
            next_rank = next_stage
            prev_rank = prev_stage
        else:
            if args.stages_per_node > 0 and local_world_size != int(args.stages_per_node):
                raise RuntimeError(
                    "LOCAL_WORLD_SIZE (%d) must equal --stages_per_node (%d)"
                    % (local_world_size, int(args.stages_per_node))
                )
            pipe_group, node_id, num_nodes = create_node_pipeline_group(
                world_size=world_size,
                local_world_size=local_world_size,
                rank=rank,
            )
            stage = local_rank
            num_stages = int(args.pipe_stages) if int(args.pipe_stages) > 0 else int(local_world_size)
            if num_stages != int(local_world_size):
                raise RuntimeError(
                    "pipe_group_scope=node requires pipe_stages (%d) == LOCAL_WORLD_SIZE (%d)"
                    % (num_stages, int(local_world_size))
                )
            node_base = rank - local_rank
            next_stage = (stage + 1) % num_stages
            prev_stage = (stage - 1 + num_stages) % num_stages
            next_rank = node_base + next_stage
            prev_rank = node_base + prev_stage

        k_slices = compute_k_slices(k=int(args.K), pipe_stages=num_stages)
        k_slice = int(k_slices[stage])
        stage_completion_ratio = float(completion_ratios(k_slices)[stage])

        if str(args.weight_mode) == "legacy":
            if str(args.stage_weight_mode) == "manual":
                stage_weights = parse_stage_weights(args.stage_weights, num_stages)
            else:
                stage_weights = [1.0 for _ in range(num_stages)]
            raw_stage_weights = resolve_raw_stage_weights(
                stage_weight_mode=str(args.stage_weight_mode),
                manual_weights=stage_weights,
                num_stages=num_stages,
                gamma=float(args.stage_weight_gamma),
            )
            w_stage = float(raw_stage_weights[stage])
            if args.stage2_beta_warmup:
                if num_stages != 4:
                    raise RuntimeError(
                        "stage2_beta_warmup currently requires 4 stages, got %d" % num_stages
                    )
                if not (0.0 <= float(args.stage2_beta_final) <= 1.0):
                    raise RuntimeError(
                        "stage2_beta_final must be in [0,1], got %.6f" % float(args.stage2_beta_final)
                    )
                if int(args.stage2_beta_warmup_t1) <= int(args.stage2_beta_warmup_t0):
                    raise RuntimeError(
                        "stage2_beta_warmup_t1 must be > stage2_beta_warmup_t0, got t0=%d t1=%d"
                        % (int(args.stage2_beta_warmup_t0), int(args.stage2_beta_warmup_t1))
                    )
        else:
            if args.stage2_beta_warmup:
                raise RuntimeError("stage2_beta_warmup requires weight_mode=legacy")
            if not (0.0 <= float(args.last2_beta) <= 1.0):
                raise RuntimeError("last2_beta must be in [0,1], got %.6f" % float(args.last2_beta))
            stage_weights = [1.0 for _ in range(num_stages)]
            raw_stage_weights = [1.0 for _ in range(num_stages)]
            w_stage = 1.0

    t.manual_seed(args.seed + rank)
    t.cuda.manual_seed_all(args.seed + rank)

    model = build_energy_model(runtime=runtime, n_f=args.n_f, unconditional_cls=EnergyModel).cuda()
    ddp_model = DDP(
        model,
        process_group=all_group,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
    )
    optimizer = t.optim.Adam(
        ddp_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    base_lr = float(args.lr)
    langevin_calibration = resolve_langevin_calibration(
        step_size=float(args.step_size),
        noise_std=float(args.noise_std),
        sampler_prime=float(args.sampler_prime),
        sampler_temperature=float(args.sampler_temperature),
    )
    energy_loss_scale = float(args.energy_loss_scale)
    if not math.isfinite(energy_loss_scale) or energy_loss_scale <= 0.0:
        raise ValueError("energy_loss_scale must be finite and > 0")
    langevin_drift_coeff = float(langevin_calibration["langevin_drift_coeff"])
    resume_extra_state: Dict = {}
    start_step = maybe_resume_from_checkpoint(
        args=args,
        model=model,
        optimizer=optimizer,
        device=t.device("cuda", local_rank),
        rank=rank,
        extra_state_out=resume_extra_state,
    )

    ds = build_dataset(runtime=runtime, cifar_builder=build_cifar10, train=True)
    dl = t.utils.data.DataLoader(
        ds,
        batch_size=local_batch,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    data_iter = iter_dataloader(dl)

    sample_dtype = str_to_dtype(args.sample_dtype)
    if (
        use_pipeline
        and sample_dtype != t.float32
        and rank == 0
        and should_log(0, args.debug_level, args.log_every)
    ):
        print(
            "[WARN] sample_dtype=%s is ignored in pipeline chain transport; chain payload uses float32"
            % args.sample_dtype,
            flush=True,
        )

    channels, height, width = runtime.image_shape
    elems_per_sample = channels * height * width
    payload_numel = local_batch * elems_per_sample

    chain_buf = t.empty((local_batch, channels, height, width), dtype=t.float32, device="cuda")
    shared_init_buf = t.empty_like(chain_buf) if args.sync_fresh_init else None
    label_buf = t.zeros((local_batch,), dtype=t.long, device="cuda") if runtime.conditional else None
    chain_id_buf = t.full((local_batch,), -1, dtype=t.long, device="cuda") if runtime.conditional else None
    steps_done_buf = t.zeros((local_batch,), dtype=t.long, device="cuda") if runtime.conditional else None
    valid_buf = t.ones((local_batch,), dtype=t.bool, device="cuda") if runtime.conditional else None
    labels_bootstrapped = not runtime.conditional
    use_conditional_replay = bool(runtime.conditional and (not use_pipeline or stage == 0))
    replay_capacity = int(args.replay_capacity)
    if use_conditional_replay:
        if replay_capacity <= 0:
            replay_capacity = max(local_batch * 32, local_batch * max(1, num_stages) * 8)
        replay = LabelReplayBuffer(
            capacity=replay_capacity,
            image_shape=runtime.image_shape,
            num_classes=runtime.num_classes,
        )
        replay_state = resume_extra_state.get("replay_state")
        if replay_state:
            replay.load_state_dict(replay_state)
    else:
        replay = None
    if args.sync_fresh_init:
        fill_shared_chain_init(
            dst=chain_buf,
            rank=rank,
            src_rank=trainer_root,
            group=all_group,
            scratch=shared_init_buf,
        )
    else:
        chain_buf.uniform_(-1, 1)
    chain_next_buf = t.empty_like(chain_buf) if use_pipeline else None
    label_next_buf = t.empty_like(label_buf) if (use_pipeline and runtime.conditional) else None
    chain_id_next_buf = t.empty_like(chain_id_buf) if (use_pipeline and runtime.conditional) else None
    steps_done_next_buf = t.empty_like(steps_done_buf) if (use_pipeline and runtime.conditional) else None
    valid_next_buf = t.empty_like(valid_buf) if (use_pipeline and runtime.conditional) else None
    sig_next_buf = t.empty((1,), dtype=t.int64, device="cuda") if (use_pipeline and runtime.conditional) else None
    restored_local_chain = restore_chain_state_into_buffers(
        resume_extra_state.get("local_chain_state"),
        device=t.device("cuda", local_rank),
        chain_buf=chain_buf if runtime.conditional else None,
        label_buf=label_buf,
        chain_id_buf=chain_id_buf,
        steps_done_buf=steps_done_buf,
        valid_buf=valid_buf,
    )
    if restored_local_chain:
        labels_bootstrapped = True

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_dir.strip():
        run_dir = os.path.abspath(args.run_dir)
        run_tag = os.path.basename(run_dir.rstrip("/"))
    else:
        if args.run_tag.strip():
            run_tag = args.run_tag.strip()
        elif use_pipeline:
            run_tag = "%s_syncA_pipe_ws%d_nn%d_ppn%d_%s" % (
                args.mode,
                world_size,
                num_nodes,
                local_world_size,
                ts,
            )
        else:
            run_tag = "%s_syncA_fullk_ws%d_%s" % (
                args.mode,
                world_size,
                ts,
            )
        run_dir = os.path.join(args.output_dir, run_tag)
    os.makedirs(run_dir, exist_ok=True)
    rank_metrics = os.path.join(run_dir, "metrics_rank%d.csv" % rank)

    if rank == trainer_root:
        if use_pipeline:
            print(
                "[Run] %s out=%s global_batch=%d local_batch=%d steps=%d K=%d K_slice=%d"
                % (run_tag, run_dir, args.batch_size, local_batch, args.steps, args.K, k_slice),
                flush=True,
            )
            print(
                "[SamplerScale] prime=%.6e temperature=%.6e effective_prime=%.6e drift_coeff=%.6e legacy_step_size=%.6e noise_std=%.6e energy_loss_scale=%.6e"
                % (
                    langevin_calibration["sampler_prime"],
                    langevin_calibration["sampler_temperature"],
                    langevin_calibration["effective_sampler_prime"],
                    langevin_drift_coeff,
                    float(args.step_size),
                    float(args.noise_std),
                    energy_loss_scale,
                ),
                flush=True,
            )
            print(
                "[Pipe] scope=%s num_nodes=%d local_world_size=%d num_stages=%d k_slices=[%s] completion=[%s] weight_mode=%s last2_beta=%.6f stage_weight_mode=%s gamma=%.3f raw_stage_weights=[%s] warmup_gate=%s beta_warmup=%s beta_final=%.6f beta_t0=%d beta_t1=%d"
                % (
                    pipe_scope,
                    num_nodes,
                    local_world_size,
                    num_stages,
                    ",".join(str(int(x)) for x in k_slices),
                    ",".join("%.4f" % x for x in completion_ratios(k_slices)),
                    str(args.weight_mode),
                    float(args.last2_beta),
                    str(args.stage_weight_mode),
                    float(args.stage_weight_gamma),
                    format_stage_weights(raw_stage_weights),
                    str(bool(args.warmup_gate)),
                    str(bool(args.stage2_beta_warmup)),
                    float(args.stage2_beta_final),
                    int(args.stage2_beta_warmup_t0),
                    int(args.stage2_beta_warmup_t1),
                ),
                flush=True,
            )
        else:
            print(
                "[Run] %s out=%s global_batch=%d local_batch=%d steps=%d K=%d (full-K DDP)"
                % (run_tag, run_dir, args.batch_size, local_batch, args.steps, args.K),
                flush=True,
            )
            print(
                "[SamplerScale] prime=%.6e temperature=%.6e effective_prime=%.6e drift_coeff=%.6e legacy_step_size=%.6e noise_std=%.6e energy_loss_scale=%.6e"
                % (
                    langevin_calibration["sampler_prime"],
                    langevin_calibration["sampler_temperature"],
                    langevin_calibration["effective_sampler_prime"],
                    langevin_drift_coeff,
                    float(args.step_size),
                    float(args.noise_std),
                    energy_loss_scale,
                ),
                flush=True,
            )

    if should_log(0, args.debug_level, args.log_every):
        if use_pipeline:
            print(
                "[PIPE rank%d] scope=%s node_id=%d local_rank=%d stage=%d prev_rank=%d next_rank=%d k_slice=%d completion=%.4f w_stage=%.6f"
                % (rank, pipe_scope, node_id, local_rank, stage, prev_rank, next_rank, k_slice, stage_completion_ratio, w_stage),
                flush=True,
            )
        else:
            print(
                "[FULLK rank%d] local_rank=%d local_batch=%d"
                % (rank, local_rank, local_batch),
                flush=True,
            )

    diag_stage_slots = int(num_stages if use_pipeline else 1)
    with open(rank_metrics, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(build_metrics_columns(num_stage_slots=diag_stage_slots))

    train_start_wall = time.time()
    optimizer_update_count = 0
    clip_active_count = 0
    for step in range(start_step, args.steps):
        iter_t0 = time.time()
        current_lr = apply_lr_warmup(
            optimizer,
            base_lr=base_lr,
            step=step,
            lr_warmup_steps=int(args.lr_warmup_steps),
        )

        # Step 0: all ranks enter step together.
        dist.barrier(group=all_group, device_ids=[local_rank])

        # Step 1: keep explicit model broadcast to reduce state divergence while debugging.
        broadcast_model_state(ddp_model.module, src_rank=trainer_root, group=all_group)
        dist.barrier(group=all_group, device_ids=[local_rank])

        batch = next(data_iter)
        pos, pos_labels = unpack_batch(batch, device=t.device("cuda", local_rank), conditional=runtime.conditional)

        # Step 2: do exactly one K_slice chain update on this stage.
        sampled_replay_state = None
        if runtime.conditional and not use_pipeline and replay is not None:
            sampled_replay_state, _ = replay.sample(pos_labels.detach().cpu())
            sampled_replay_state = sampled_replay_state.to(chain_buf.device)
            chain_buf.copy_(sampled_replay_state.x)
            label_buf.copy_(sampled_replay_state.y)
            chain_id_buf.copy_(sampled_replay_state.chain_id)
            steps_done_buf.copy_(sampled_replay_state.steps_done)
            valid_buf.copy_(sampled_replay_state.valid)
            labels_bootstrapped = True
        elif args.fresh_init:
            if args.sync_fresh_init:
                if use_pipeline:
                    fill_shared_chain_init(
                        dst=shared_init_buf,
                        rank=rank,
                        src_rank=trainer_root,
                        group=all_group,
                    )
                    if stage == 0:
                        chain_buf.copy_(shared_init_buf)
                else:
                    fill_shared_chain_init(
                        dst=chain_buf,
                        rank=rank,
                        src_rank=trainer_root,
                        group=all_group,
                        scratch=shared_init_buf,
                    )
            elif (use_pipeline and stage == 0) or (not use_pipeline):
                chain_buf.uniform_(-1, 1)
            if runtime.conditional and stage == 0:
                if replay is not None:
                    sampled_replay_state, _ = replay.sample(pos_labels.detach().cpu())
                    sampled_replay_state = sampled_replay_state.to(chain_buf.device)
                    chain_buf.copy_(sampled_replay_state.x)
                    label_buf.copy_(sampled_replay_state.y)
                    chain_id_buf.copy_(sampled_replay_state.chain_id)
                    steps_done_buf.copy_(sampled_replay_state.steps_done)
                    valid_buf.copy_(sampled_replay_state.valid)
                else:
                    label_buf.copy_(pos_labels)
                    chain_id_buf.fill_(-1)
                    steps_done_buf.zero_()
                    valid_buf.fill_(True)
                labels_bootstrapped = True
        if runtime.conditional:
            if not labels_bootstrapped:
                if use_pipeline and stage == 0 and replay is not None:
                    sampled_replay_state, _ = replay.sample(pos_labels.detach().cpu())
                    sampled_replay_state = sampled_replay_state.to(chain_buf.device)
                    chain_buf.copy_(sampled_replay_state.x)
                    label_buf.copy_(sampled_replay_state.y)
                    chain_id_buf.copy_(sampled_replay_state.chain_id)
                    steps_done_buf.copy_(sampled_replay_state.steps_done)
                    valid_buf.copy_(sampled_replay_state.valid)
                else:
                    label_buf.copy_(pos_labels)
                    chain_id_buf.fill_(-1)
                    steps_done_buf.zero_()
                    valid_buf.fill_(True)
                labels_bootstrapped = True
            elif not use_pipeline:
                label_buf.copy_(pos_labels)
            elif args.fresh_init and stage == 0 and replay is None:
                label_buf.copy_(pos_labels)
                chain_id_buf.fill_(-1)
                steps_done_buf.zero_()
                valid_buf.fill_(True)
        sample_t0 = time.time()
        if runtime.conditional:
            chain_out = langevin_sample_conditional(
                model=model,
                chain=chain_buf,
                labels=label_buf,
                k_steps=(k_slice if use_pipeline else int(args.K)),
                langevin_sign=float(args.langevin_sign),
                step_size=args.step_size,
                noise_std=args.noise_std,
                clamp_x=(not args.no_clamp_x),
                clamp_last_only=bool(args.clamp_last_only),
                drift_coeff=langevin_drift_coeff,
            )
        else:
            chain_out = langevin_sample(
                model=model,
                chain=chain_buf,
                k_steps=(k_slice if use_pipeline else int(args.K)),
                langevin_sign=float(args.langevin_sign),
                step_size=args.step_size,
                noise_std=args.noise_std,
                clamp_x=(not args.no_clamp_x),
                clamp_last_only=bool(args.clamp_last_only),
                drift_coeff=langevin_drift_coeff,
            )
        sample_ms = (time.time() - sample_t0) * 1000.0
        if runtime.conditional:
            steps_done_buf.add_(int(k_slice if use_pipeline else int(args.K)))
            valid_buf.fill_(True)

        wait_ms = 0.0
        w_stage_current = w_stage
        w_effective = 1.0
        alpha_stage = 1.0
        c_stage = 1.0
        sum_alpha_active = 1.0
        beta_t = float("nan")
        alpha_stage2 = float("nan")
        alpha_stage3 = float("nan")

        if use_pipeline:
            # Step 3: immediately post ring chain transfer to overlap with training.
            send_payload = chain_out.contiguous().view(-1)
            send_header = make_sample_header(
                step=step,
                numel=send_payload.numel(),
                dtype_id=dtype_to_id(send_payload.dtype),
                device="cuda",
                msg_type=MSG_CHAIN,
            )
            recv_header = t.empty((4,), dtype=t.int64, device="cuda")
            recv_payload = chain_next_buf.view(-1)

            send_header_tag, send_payload_tag = chain_tags(step=step, stage=stage, num_stages=num_stages)
            recv_header_tag, recv_payload_tag = chain_tags(step=step, stage=prev_stage, num_stages=num_stages)

            p2p_ops = [
                dist.P2POp(dist.irecv, recv_header, prev_rank, pipe_group, recv_header_tag),
                dist.P2POp(dist.irecv, recv_payload, prev_rank, pipe_group, recv_payload_tag),
                dist.P2POp(dist.isend, send_header, next_rank, pipe_group, send_header_tag),
                dist.P2POp(dist.isend, send_payload, next_rank, pipe_group, send_payload_tag),
            ]
            if runtime.conditional:
                send_label_tag = chain_label_tag(step=step, stage=stage, num_stages=num_stages)
                recv_label_tag = chain_label_tag(step=step, stage=prev_stage, num_stages=num_stages)
                send_chain_id_tag = chain_meta_tag(step=step, stage=stage, num_stages=num_stages, field_offset=0)
                recv_chain_id_tag = chain_meta_tag(step=step, stage=prev_stage, num_stages=num_stages, field_offset=0)
                send_steps_tag = chain_meta_tag(step=step, stage=stage, num_stages=num_stages, field_offset=1)
                recv_steps_tag = chain_meta_tag(step=step, stage=prev_stage, num_stages=num_stages, field_offset=1)
                send_valid_tag = chain_meta_tag(step=step, stage=stage, num_stages=num_stages, field_offset=2)
                recv_valid_tag = chain_meta_tag(step=step, stage=prev_stage, num_stages=num_stages, field_offset=2)
                send_sig_tag = chain_sig_tag(step=step, stage=stage, num_stages=num_stages)
                recv_sig_tag = chain_sig_tag(step=step, stage=prev_stage, num_stages=num_stages)
                send_sig = conditional_state_signature(label_buf, chain_id_buf, steps_done_buf, valid_buf)
                p2p_ops.extend(
                    [
                        dist.P2POp(dist.irecv, label_next_buf, prev_rank, pipe_group, recv_label_tag),
                        dist.P2POp(dist.isend, label_buf.contiguous(), next_rank, pipe_group, send_label_tag),
                        dist.P2POp(dist.irecv, chain_id_next_buf, prev_rank, pipe_group, recv_chain_id_tag),
                        dist.P2POp(dist.isend, chain_id_buf.contiguous(), next_rank, pipe_group, send_chain_id_tag),
                        dist.P2POp(dist.irecv, steps_done_next_buf, prev_rank, pipe_group, recv_steps_tag),
                        dist.P2POp(dist.isend, steps_done_buf.contiguous(), next_rank, pipe_group, send_steps_tag),
                        dist.P2POp(dist.irecv, valid_next_buf, prev_rank, pipe_group, recv_valid_tag),
                        dist.P2POp(dist.isend, valid_buf.contiguous(), next_rank, pipe_group, send_valid_tag),
                        dist.P2POp(dist.irecv, sig_next_buf, prev_rank, pipe_group, recv_sig_tag),
                        dist.P2POp(dist.isend, send_sig.contiguous(), next_rank, pipe_group, send_sig_tag),
                    ]
                )
            works = dist.batch_isend_irecv(p2p_ops)

        # Step 4: one local train update; DDP does global gradient sync across WORLD.
        train_t0 = time.time()
        if args.pos_noise_std > 0:
            pos = pos + args.pos_noise_std * t.randn_like(pos)
        if not args.no_clamp_pos:
            pos = pos.clamp(-1, 1)

        if use_pipeline:
            if str(args.weight_mode) == "legacy":
                step_raw_stage_weights = raw_stage_weights
                if args.stage2_beta_warmup:
                    beta_t = stage2_beta_schedule(
                        step=step,
                        beta_final=float(args.stage2_beta_final),
                        t0=int(args.stage2_beta_warmup_t0),
                        t1=int(args.stage2_beta_warmup_t1),
                    )
                    step_raw_stage_weights = [0.0 for _ in range(num_stages)]
                    step_raw_stage_weights[2] = float(beta_t)
                    step_raw_stage_weights[3] = float(1.0 - beta_t)
                w_stage_current = float(step_raw_stage_weights[stage])
                valid = (step - stage) >= 0
                w_effective = w_stage_current if (not args.warmup_gate or valid) else 0.0
                alpha_vec = compute_stage_alpha_vector(
                    step=step,
                    raw_stage_weights=step_raw_stage_weights,
                    warmup_gate=bool(args.warmup_gate),
                )
            else:
                alpha_vec = compute_completion_aware_alpha(
                    step=step,
                    pipe_stages=num_stages,
                    weight_mode=str(args.weight_mode),
                    last2_beta=float(args.last2_beta),
                )
                w_stage_current = float(alpha_vec[stage])
                w_effective = float(alpha_vec[stage])
            alpha_stage = float(alpha_vec[stage])
            c_stage = float(num_stages) * alpha_stage
            sum_alpha_active = float(sum(alpha_vec))
            if num_stages > 2:
                alpha_stage2 = float(alpha_vec[2])
            if num_stages > 3:
                alpha_stage3 = float(alpha_vec[3])
        else:
            w_effective = 1.0
            alpha_stage = 1.0
            c_stage = 1.0
            sum_alpha_active = 1.0

        active_stage_count = active_stage_count_from_alpha(alpha_vec if use_pipeline else [1.0])
        active_stage_set = active_stage_set_from_alpha(alpha_vec if use_pipeline else [1.0])
        skip_optimizer_update = use_pipeline and should_skip_optimizer_for_pipeline_warmup(
            step=step,
            pipe_stages=num_stages,
            enabled=bool(args.skip_optimizer_until_full_diagonal),
        )

        optimizer.zero_grad(set_to_none=True)
        pos_energy = energy_call(ddp_model, pos, pos_labels)
        neg_energy = energy_call(ddp_model, chain_out.detach(), label_buf if runtime.conditional else None)
        f_pos = pos_energy.mean()
        f_neg = neg_energy.mean()
        # Completion-aware weighting belongs on the negative scalar loss term for this
        # stage. Stage chains are never averaged in image space; pipeline stages only
        # interact through chain state handoff and updated parameters.
        loss_pos = f_pos
        loss_neg = c_stage * f_neg
        objective = loss_pos - loss_neg
        # Keep objective sign consistent with sampler direction:
        # langevin_sign>0 matches score-ascent (maximize objective), <0 matches energy-descent (minimize objective).
        loss_sign = -1.0 if float(args.langevin_sign) > 0.0 else 1.0
        loss = objective * loss_sign * energy_loss_scale
        f_pos_val_pre = float(f_pos.detach().item())
        f_neg_val_pre = float(f_neg.detach().item())
        loss_neg_val_pre = float(loss_neg.detach().item())
        min_energy_local_pre = float(
            t.minimum(pos_energy.detach().min(), neg_energy.detach().min()).item()
        )
        max_energy_local_pre = float(
            t.maximum(pos_energy.detach().max(), neg_energy.detach().max()).item()
        )
        chain_max_abs_local = float(chain_out.detach().abs().max().item())
        chain_mean_abs_local = float(chain_out.detach().abs().mean().item())
        clamp_sat_frac_local = clamp_saturation_fraction(chain_out)
        f_pos_total_pre = reduce_mean_scalar(f_pos_val_pre, world_size=world_size, group=all_group, device=chain_out.device)
        f_neg_total_pre = reduce_mean_scalar(loss_neg_val_pre, world_size=world_size, group=all_group, device=chain_out.device)
        crash_reasons = evaluate_stability_crash_guard(
            f_pos=f_pos_val_pre,
            f_neg_total=f_neg_total_pre,
            max_abs_chain=chain_max_abs_local,
            clamp_x=(not args.no_clamp_x),
            grad_norm_theta=0.0,
            crash_abs_energy=float(args.crash_abs_energy),
            crash_max_abs_chain_unclamped=float(args.crash_max_abs_chain_unclamped),
            crash_grad_norm=0.0,
        )
        pre_crash_flag = t.tensor([1 if crash_reasons else 0], dtype=t.int32, device=chain_out.device)
        dist.all_reduce(pre_crash_flag, op=dist.ReduceOp.MAX, group=all_group)
        if int(pre_crash_flag.item()) > 0 and not crash_reasons:
            crash_reasons = ["peer_crash_guard_before_backward"]
        optimizer_update = (not skip_optimizer_update) and (int(pre_crash_flag.item()) == 0)
        grad_norm_theta = 0.0
        grad_norm_pre_clip = 0.0
        grad_norm_post_clip = 0.0
        energy_head_grad_norm = 0.0
        backbone_grad_norm = 0.0
        param_norm_val = parameter_norm(ddp_model.parameters())
        energy_head_weight_norm = named_parameter_norm(
            ddp_model.named_parameters(),
            include_head=True,
        )
        update_norm = 0.0
        update_to_param_ratio = 0.0
        clip_active = False
        if optimizer_update:
            loss.backward()
            grad_norm_pre_clip = parameter_grad_norm(ddp_model.parameters())
            grad_norm_theta = grad_norm_pre_clip
            energy_head_grad_norm = named_parameter_grad_norm(
                ddp_model.named_parameters(),
                include_head=True,
            )
            backbone_grad_norm = named_parameter_grad_norm(
                ddp_model.named_parameters(),
                include_head=False,
            )
            crash_reasons = evaluate_stability_crash_guard(
                f_pos=f_pos_val_pre,
                f_neg_total=f_neg_total_pre,
                max_abs_chain=chain_max_abs_local,
                clamp_x=(not args.no_clamp_x),
                grad_norm_theta=grad_norm_theta,
                crash_abs_energy=float(args.crash_abs_energy),
                crash_max_abs_chain_unclamped=float(args.crash_max_abs_chain_unclamped),
                crash_grad_norm=float(args.crash_grad_norm),
            )
            post_crash_flag = t.tensor([1 if crash_reasons else 0], dtype=t.int32, device=chain_out.device)
            dist.all_reduce(post_crash_flag, op=dist.ReduceOp.MAX, group=all_group)
            if int(post_crash_flag.item()) > 0 and not crash_reasons:
                crash_reasons = ["peer_crash_guard_after_backward"]
            if not crash_reasons:
                params_before_step = clone_trainable_parameters(ddp_model.parameters())
                clip_norm = args.grad_clip_norm if args.grad_clip_norm is not None else args.max_grad_norm
                if clip_norm and float(clip_norm) > 0:
                    nn.utils.clip_grad_norm_(ddp_model.parameters(), float(clip_norm))
                    grad_norm_post_clip = parameter_grad_norm(ddp_model.parameters())
                    clip_active = bool(grad_norm_pre_clip > float(clip_norm))
                else:
                    grad_norm_post_clip = grad_norm_pre_clip
                optimizer.step()
                optimizer_update_count += 1
                if clip_active:
                    clip_active_count += 1
                update_norm = parameter_update_norm(params_before_step, ddp_model.parameters())
                param_norm_val = parameter_norm(ddp_model.parameters())
                energy_head_weight_norm = named_parameter_norm(
                    ddp_model.named_parameters(),
                    include_head=True,
                )
                update_to_param_ratio = (
                    update_norm / param_norm_val
                    if math.isfinite(param_norm_val) and abs(param_norm_val) > 0.0
                    else float("nan")
                )
            else:
                optimizer_update = False
        crash_flag = t.tensor([1 if crash_reasons else 0], dtype=t.int32, device=chain_out.device)
        dist.all_reduce(crash_flag, op=dist.ReduceOp.MAX, group=all_group)
        crash_guard_triggered = bool(int(crash_flag.item()) > 0)
        crash_guard_reason = ";".join(crash_reasons)
        clip_active_fraction_value = (
            float(clip_active_count) / float(optimizer_update_count)
            if optimizer_update_count > 0
            else 0.0
        )
        soft_guard_reasons = evaluate_soft_stability_guard(
            f_pos=f_pos_val_pre,
            grad_norm_pre_clip=grad_norm_pre_clip,
            grad_clip_active_fraction=clip_active_fraction_value,
            soft_abs_energy=float(args.soft_abs_energy),
            soft_grad_norm=float(args.soft_grad_norm),
            hard_grad_norm=float(args.hard_grad_norm),
            clip_active_fraction=float(args.clip_active_fraction),
        )
        soft_guard_reason = ";".join(soft_guard_reasons)
        if bool(args.soft_guard_stop) and soft_guard_reasons and not crash_guard_triggered:
            crash_reasons = list(soft_guard_reasons)
            crash_guard_triggered = True
            crash_guard_reason = soft_guard_reason

        if runtime.conditional and replay is not None and chain_id_buf is not None:
            if bool((chain_id_buf >= 0).all().item()):
                replay_state = chain_state_from_buffers(
                    x=chain_out.detach(),
                    y=label_buf,
                    chain_id=chain_id_buf,
                    steps_done=steps_done_buf,
                    valid=valid_buf,
                    stage_id=(stage if use_pipeline else 0),
                )
                replay.update_slots(replay_state.to("cpu"))

        train_ms = (time.time() - train_t0) * 1000.0
        metric = float((f_pos - f_neg).item())
        f_pos_val = float(f_pos.detach().item())
        f_neg_val = float(f_neg.detach().item())
        loss_pos_val = float(loss_pos.detach().item())
        loss_neg_val = float(loss_neg.detach().item())
        f_pos_total_val = f_pos_total_pre
        f_neg_total_val = reduce_mean_scalar(loss_neg_val, world_size=world_size, group=all_group, device=chain_out.device)
        e_neg_minus_e_pos_gap = float(f_neg_total_val - f_pos_total_val)
        max_f_neg_local = float(neg_energy.detach().max().item())
        min_energy_across_ranks = reduce_min_scalar(
            min_energy_local_pre,
            group=all_group,
            device=chain_out.device,
        )
        max_energy_across_ranks = reduce_max_scalar(
            max_energy_local_pre,
            group=all_group,
            device=chain_out.device,
        )
        objective_val = float(objective.detach().item())
        loss_val = float(loss.detach().item())
        is_finite_local = bool(
            t.isfinite(f_pos.detach()).item()
            and t.isfinite(f_neg.detach()).item()
            and t.isfinite(loss.detach()).item()
            and t.isfinite(chain_out).all().item()
        )
        nonfinite_flag = t.tensor([0 if is_finite_local else 1], dtype=t.int32, device=chain_out.device)
        dist.all_reduce(nonfinite_flag, op=dist.ReduceOp.MAX, group=all_group)
        force_log_nonfinite = bool(int(nonfinite_flag.item()) > 0)
        max_f_neg_across_ranks = reduce_max_scalar(max_f_neg_local, group=all_group, device=chain_out.device)
        max_abs_chain_across_ranks = reduce_max_scalar(
            chain_max_abs_local,
            group=all_group,
            device=chain_out.device,
        )
        stage_diag = collect_stage_diagnostic_vectors(
            stage_index=(stage if use_pipeline else 0),
            num_stage_slots=diag_stage_slots,
            f_neg=f_neg_val,
            max_f_neg=max_f_neg_local,
            max_abs_chain=chain_max_abs_local,
            mean_abs_chain=chain_mean_abs_local,
            clamp_sat_frac=clamp_sat_frac_local,
            alpha_bar=alpha_stage,
            grad_norm_x=float("nan"),
            group=all_group,
            world_size=world_size,
            device=chain_out.device,
        )

        if args.vis_every > 0 and (step % int(args.vis_every) == 0):
            vis_stage = int(args.vis_stage)
            if use_pipeline:
                if vis_stage < 0:
                    vis_stage = num_stages - 1
                if vis_stage < 0 or vis_stage >= num_stages:
                    raise RuntimeError(
                        "vis_stage must be in [0,%d] or -1, got %d" % (num_stages - 1, int(args.vis_stage))
                    )
                if pipe_scope == "world":
                    do_write = rank == vis_stage
                else:
                    do_write = (node_id == 0 and stage == vis_stage)
            else:
                if vis_stage not in (-1, 0):
                    raise RuntimeError("vis_stage for non-pipeline mode must be -1 or 0")
                do_write = rank == 0
            if do_write:
                vis_dir = args.vis_dir.strip() if args.vis_dir.strip() else os.path.join(run_dir, "vis")
                os.makedirs(vis_dir, exist_ok=True)
                vis_num = max(1, int(args.vis_num))
                vis_nrow = max(1, int(args.vis_nrow))
                token_idx_vis = (step - stage) if use_pipeline else step
                neg_path = os.path.join(
                    vis_dir,
                    "step%06d_neg_stage%d_token%d.png" % (step, stage, token_idx_vis),
                )
                pos_path = os.path.join(
                    vis_dir,
                    "step%06d_pos_rank%d.png" % (step, rank),
                )
                save_grid_png(chain_out[:vis_num], neg_path, nrow=vis_nrow)
                save_grid_png(pos[:vis_num], pos_path, nrow=vis_nrow)
            dist.barrier(group=all_group, device_ids=[local_rank])

        # Step 5: complete chain transfer then swap to next chain input.
        if use_pipeline:
            wait_t0 = time.time()
            for work in works:
                work.wait()
            wait_ms = (time.time() - wait_t0) * 1000.0

            msg_type, msg_step, numel, dtype_id = parse_sample_header(recv_header)
            if msg_type != MSG_CHAIN:
                raise RuntimeError(f"rank {rank} stage {stage} got unexpected msg_type={msg_type}")
            if msg_step != step:
                raise RuntimeError(
                    "rank %d stage %d got step=%d from prev stage %d, expected step=%d"
                    % (rank, stage, msg_step, prev_stage, step)
                )
            if numel != payload_numel:
                raise RuntimeError(
                    "rank %d stage %d got numel=%d from prev stage %d, expected=%d"
                    % (rank, stage, numel, prev_stage, payload_numel)
                )
            recv_dtype = id_to_dtype(dtype_id)
            if recv_dtype != t.float32:
                raise RuntimeError(
                    "rank %d stage %d got recv_dtype=%s from prev stage %d, expected float32"
                    % (rank, stage, str(recv_dtype), prev_stage)
                )
            if runtime.conditional:
                assert_pipeline_received_state(
                    x=chain_next_buf,
                    y=label_next_buf,
                    chain_id=chain_id_next_buf,
                    steps_done=steps_done_next_buf,
                    valid=valid_next_buf,
                    expected_signature=sig_next_buf,
                    num_classes=runtime.num_classes,
                    min_steps_done=int(k_slices[prev_stage]),
                    rank=rank,
                    stage=stage,
                    step=step,
                )

            chain_buf, chain_next_buf = chain_next_buf, chain_buf
            if runtime.conditional:
                label_buf, label_next_buf = label_next_buf, label_buf
                chain_id_buf, chain_id_next_buf = chain_id_next_buf, chain_id_buf
                steps_done_buf, steps_done_next_buf = steps_done_next_buf, steps_done_buf
                valid_buf, valid_next_buf = valid_next_buf, valid_buf

        # Step 6: global end-of-step alignment.
        dist.barrier(group=all_group, device_ids=[local_rank])

        iter_ms = (time.time() - iter_t0) * 1000.0
        comm_ms = max(0.0, iter_ms - train_ms)
        comm_pct = 100.0 * comm_ms / iter_ms if iter_ms > 0 else 0.0
        wallclock_sec = max(0.0, time.time() - train_start_wall)
        gpu_hours = wallclock_sec * float(world_size) / 3600.0
        token_idx = (step - stage) if use_pipeline else step
        mean_c_stage = float("nan")
        chain_min_global = float("nan")
        chain_max_global = float("nan")
        chain_mean_global = float("nan")
        chain_std_global = float("nan")
        chain_min_local = float("nan")
        chain_max_local = float("nan")
        chain_mean_local = float("nan")
        chain_std_local = float("nan")
        chain_frac_zero_local = float("nan")
        chain_frac_one_local = float("nan")
        chain_unique_count_local = -1
        chain_unique_mode_local = "na"
        do_log = (
            should_diagnostic_log(
                step=step,
                debug_level=args.debug_level,
                log_every=args.log_every,
                diagnostic_first_steps=args.diagnostic_first_steps,
                diagnostic_log_every=args.diagnostic_log_every,
            )
            or force_log_nonfinite
            or crash_guard_triggered
        )
        per_rank_fneg_msg = ""

        if do_log:
            mean_c_stage = reduce_mean_scalar(c_stage, world_size=world_size, group=all_group, device=chain_out.device)
            chain_min_global, chain_max_global, chain_mean_global, chain_std_global = reduce_chain_summary(
                chain_out, world_size=world_size, group=all_group
            )
            (
                chain_min_local,
                chain_max_local,
                chain_mean_local,
                chain_std_local,
                chain_frac_zero_local,
                chain_frac_one_local,
                chain_unique_count_local,
                chain_unique_mode_local,
            ) = local_chain_summary(chain_out)
            gathered_local = t.tensor([float(rank), float(stage), f_neg_val], dtype=t.float32, device=chain_out.device)
            gathered = [t.empty_like(gathered_local) for _ in range(world_size)]
            dist.all_gather(gathered, gathered_local, group=all_group)
            if rank == trainer_root:
                per_rank_fneg_msg = " ".join(
                    "r%d:s%d=%.4e" % (int(x[0].item()), int(x[1].item()), float(x[2].item()))
                    for x in gathered
                )

        if do_log:
            if use_pipeline:
                print(
                    "[STEP rank%d stage%d] step=%d token=%d w_stage=%.6f w_eff=%.6f alpha=%.6f c_stage=%.6f mean_c_stage=%.6f sum_alpha=%.6f beta_t=%.6f alpha2=%.6f alpha3=%.6f f_pos=%.4e f_neg=%.4e objective=%.4e loss=%.4e finite=%d max_f_neg=%.4e max_abs_chain=%.4e chain_local[min=%.4e max=%.4e mean=%.4e std=%.4e z0=%.4e o1=%.4e uniq=%d(%s)] chain_global[min=%.4e max=%.4e mean=%.4e std=%.4e] sample=%.2fms train=%.2fms wait=%.2fms"
                    % (
                        rank,
                        stage,
                        step,
                        token_idx,
                        w_stage_current,
                        w_effective,
                        alpha_stage,
                        c_stage,
                        mean_c_stage,
                        sum_alpha_active,
                        beta_t,
                        alpha_stage2,
                        alpha_stage3,
                        f_pos_val,
                        f_neg_val,
                        objective_val,
                        loss_val,
                        1 if is_finite_local else 0,
                        max_f_neg_across_ranks,
                        max_abs_chain_across_ranks,
                        chain_min_local,
                        chain_max_local,
                        chain_mean_local,
                        chain_std_local,
                        chain_frac_zero_local,
                        chain_frac_one_local,
                        chain_unique_count_local,
                        chain_unique_mode_local,
                        chain_min_global,
                        chain_max_global,
                        chain_mean_global,
                        chain_std_global,
                        sample_ms,
                        train_ms,
                        wait_ms,
                    ),
                    flush=True,
                )
                if rank == trainer_root:
                    print(
                        "[GLOBAL step%d] per_rank_fneg: %s" % (step, per_rank_fneg_msg),
                        flush=True,
                    )
                    print(
                        "[DIAG step%d] lr=%.4e E_pos=%.4e E_neg=%.4e E_neg_total=%.4e gap=%.4e min_energy=%.4e max_energy=%.4e f_neg_total=%.4e loss_pos=%.4e loss_neg=%.4e loss_total=%.4e grad_norm_theta=%.4e grad_pre=%.4e grad_post=%.4e param_norm=%.4e update_norm=%.4e update_ratio=%.4e head_w=%.4e head_grad=%.4e backbone_grad=%.4e clip_active=%d clip_frac=%.4f sampler_prime=%.4e sampler_temp=%.4e eff_prime=%.4e drift_coeff=%.4e loss_scale=%.4e clamp_sat_frac=%.4e wallclock_sec=%.2f gpu_hours=%.6f optimizer_update=%d active_stages=%s alpha_bar=%s max_abs_chain_stage=%s mean_abs_chain_stage=%s clamp_sat_frac_stage=%s soft=%s crash=%s"
                        % (
                            step,
                            current_lr,
                            f_pos_val,
                            f_neg_val,
                            f_neg_total_val,
                            e_neg_minus_e_pos_gap,
                            min_energy_across_ranks,
                            max_energy_across_ranks,
                            f_neg_total_val,
                            loss_pos_val,
                            loss_neg_val,
                            loss_val,
                            grad_norm_theta,
                            grad_norm_pre_clip,
                            grad_norm_post_clip,
                            param_norm_val,
                            update_norm,
                            update_to_param_ratio,
                            energy_head_weight_norm,
                            energy_head_grad_norm,
                            backbone_grad_norm,
                            1 if clip_active else 0,
                            clip_active_fraction_value,
                            langevin_calibration["sampler_prime"],
                            langevin_calibration["sampler_temperature"],
                            langevin_calibration["effective_sampler_prime"],
                            langevin_drift_coeff,
                            energy_loss_scale,
                            clamp_sat_frac_local,
                            wallclock_sec,
                            gpu_hours,
                            1 if optimizer_update else 0,
                            active_stage_set,
                            ",".join("%.4e" % x for x in stage_diag["alpha_bar_stage"]),
                            ",".join("%.4e" % x for x in stage_diag["max_abs_chain_stage"]),
                            ",".join("%.4e" % x for x in stage_diag["mean_abs_chain_stage"]),
                            ",".join("%.4e" % x for x in stage_diag["clamp_sat_frac_stage"]),
                            soft_guard_reason or "none",
                            crash_guard_reason or "none",
                        ),
                        flush=True,
                    )
            else:
                print(
                    "[STEP rank%d fullk] step=%d f_pos=%.4e f_neg=%.4e objective=%.4e loss=%.4e finite=%d max_f_neg=%.4e max_abs_chain=%.4e chain[min=%.4e max=%.4e mean=%.4e std=%.4e z0=%.4e o1=%.4e uniq=%d(%s)] sample=%.2fms train=%.2fms"
                    % (
                        rank,
                        step,
                        f_pos_val,
                        f_neg_val,
                        objective_val,
                        loss_val,
                        1 if is_finite_local else 0,
                        max_f_neg_across_ranks,
                        max_abs_chain_across_ranks,
                        chain_min_local,
                        chain_max_local,
                        chain_mean_local,
                        chain_std_local,
                        chain_frac_zero_local,
                        chain_frac_one_local,
                        chain_unique_count_local,
                        chain_unique_mode_local,
                        sample_ms,
                        train_ms,
                    ),
                    flush=True,
                )
                if rank == trainer_root:
                    print(
                        "[DIAG step%d] lr=%.4e E_pos=%.4e E_neg=%.4e E_neg_total=%.4e gap=%.4e min_energy=%.4e max_energy=%.4e f_neg_total=%.4e loss_pos=%.4e loss_neg=%.4e loss_total=%.4e grad_norm_theta=%.4e grad_pre=%.4e grad_post=%.4e param_norm=%.4e update_norm=%.4e update_ratio=%.4e head_w=%.4e head_grad=%.4e backbone_grad=%.4e clip_active=%d clip_frac=%.4f sampler_prime=%.4e sampler_temp=%.4e eff_prime=%.4e drift_coeff=%.4e loss_scale=%.4e clamp_sat_frac=%.4e wallclock_sec=%.2f gpu_hours=%.6f optimizer_update=%d active_stages=%s alpha_bar=%s max_abs_chain_stage=%s mean_abs_chain_stage=%s clamp_sat_frac_stage=%s soft=%s crash=%s"
                        % (
                            step,
                            current_lr,
                            f_pos_val,
                            f_neg_val,
                            f_neg_total_val,
                            e_neg_minus_e_pos_gap,
                            min_energy_across_ranks,
                            max_energy_across_ranks,
                            f_neg_total_val,
                            loss_pos_val,
                            loss_neg_val,
                            loss_val,
                            grad_norm_theta,
                            grad_norm_pre_clip,
                            grad_norm_post_clip,
                            param_norm_val,
                            update_norm,
                            update_to_param_ratio,
                            energy_head_weight_norm,
                            energy_head_grad_norm,
                            backbone_grad_norm,
                            1 if clip_active else 0,
                            clip_active_fraction_value,
                            langevin_calibration["sampler_prime"],
                            langevin_calibration["sampler_temperature"],
                            langevin_calibration["effective_sampler_prime"],
                            langevin_drift_coeff,
                            energy_loss_scale,
                            clamp_sat_frac_local,
                            wallclock_sec,
                            gpu_hours,
                            1 if optimizer_update else 0,
                            active_stage_set,
                            ",".join("%.4e" % x for x in stage_diag["alpha_bar_stage"]),
                            ",".join("%.4e" % x for x in stage_diag["max_abs_chain_stage"]),
                            ",".join("%.4e" % x for x in stage_diag["mean_abs_chain_stage"]),
                            ",".join("%.4e" % x for x in stage_diag["clamp_sat_frac_stage"]),
                            soft_guard_reason or "none",
                            crash_guard_reason or "none",
                        ),
                        flush=True,
                    )

        stage_diag_values: List[object] = []
        for diag_s in range(diag_stage_slots):
            stage_diag_values.extend(
                [
                    "%.6e" % stage_diag["f_neg_stage"][diag_s],
                    "%.6e" % stage_diag["E_neg_stage"][diag_s],
                    "%.6e" % stage_diag["max_f_neg_stage"][diag_s],
                    "%.6e" % stage_diag["max_abs_chain_stage"][diag_s],
                    "%.6e" % stage_diag["mean_abs_chain_stage"][diag_s],
                    "%.6e" % stage_diag["clamp_sat_frac_stage"][diag_s],
                    "%.6e" % stage_diag["alpha_bar_stage"][diag_s],
                    "%.6e" % stage_diag["grad_norm_x_stage"][diag_s],
                ]
            )

        with open(rank_metrics, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    step,
                    "%.3f" % wait_ms,
                    "%.3f" % train_ms,
                    "%.3f" % iter_ms,
                    "%.3f" % comm_ms,
                    "%.2f" % comm_pct,
                    "%.6f" % wallclock_sec,
                    "%.9f" % gpu_hours,
                    "%.6f" % metric,
                    "%.6e" % f_pos_val,
                    "%.6e" % f_neg_val,
                    "%.6e" % objective_val,
                    "%.6e" % loss_val,
                    "%.6e" % current_lr,
                    "%.6e" % f_pos_val,
                    "%.6e" % f_neg_val,
                    "%.6e" % f_neg_total_val,
                    "%.6e" % e_neg_minus_e_pos_gap,
                    "%.6e" % min_energy_across_ranks,
                    "%.6e" % max_energy_across_ranks,
                    1 if is_finite_local else 0,
                    "%.6e" % max_f_neg_across_ranks,
                    "%.6e" % max_abs_chain_across_ranks,
                    "%.6e" % clamp_sat_frac_local,
                    stage,
                    "%.6f" % w_stage,
                    "%.6f" % w_effective,
                    "%.6f" % alpha_stage,
                    "%.6f" % c_stage,
                    "%.6f" % mean_c_stage,
                    "%.6f" % sum_alpha_active,
                    "%.6f" % beta_t,
                    "%.6f" % alpha_stage2,
                    "%.6f" % alpha_stage3,
                    "%.6e" % chain_min_global,
                    "%.6e" % chain_max_global,
                    "%.6e" % chain_mean_global,
                    "%.6e" % chain_std_global,
                    "%.6e" % chain_frac_zero_local,
                    "%.6e" % chain_frac_one_local,
                    chain_unique_count_local,
                    token_idx,
                    "%.6e" % f_neg_total_val,
                    "%.6e" % loss_pos_val,
                    "%.6e" % loss_neg_val,
                    "%.6e" % loss_val,
                    "%.6e" % grad_norm_theta,
                    "%.6e" % grad_norm_pre_clip,
                    "%.6e" % grad_norm_post_clip,
                    "%.6e" % param_norm_val,
                    "%.6e" % update_norm,
                    "%.6e" % update_to_param_ratio,
                    "%.6e" % energy_head_weight_norm,
                    "%.6e" % energy_head_grad_norm,
                    "%.6e" % backbone_grad_norm,
                    1 if clip_active else 0,
                    "%.6f" % clip_active_fraction_value,
                    "%.6e" % langevin_calibration["sampler_prime"],
                    "%.6e" % langevin_calibration["sampler_temperature"],
                    "%.6e" % langevin_calibration["effective_sampler_prime"],
                    "%.6e" % langevin_drift_coeff,
                    "%.6e" % energy_loss_scale,
                    soft_guard_reason,
                    1 if optimizer_update else 0,
                    active_stage_count,
                    active_stage_set,
                    crash_guard_reason,
                ] + stage_diag_values
            )

        if crash_guard_triggered:
            if rank == trainer_root:
                crash_path = os.path.join(run_dir, "crash_guard_step%d.txt" % step)
                with open(crash_path, "w") as crash_f:
                    crash_f.write("step=%d\n" % step)
                    crash_f.write("reason=%s\n" % (crash_guard_reason or "peer_crash_guard"))
                    crash_f.write("f_pos=%.8e\n" % f_pos_val)
                    crash_f.write("f_neg_total=%.8e\n" % f_neg_total_val)
                    crash_f.write("max_abs_chain=%.8e\n" % max_abs_chain_across_ranks)
                    crash_f.write("grad_norm_theta=%.8e\n" % grad_norm_theta)
                    crash_f.write("grad_norm_pre_clip=%.8e\n" % grad_norm_pre_clip)
                    crash_f.write("grad_norm_post_clip=%.8e\n" % grad_norm_post_clip)
                    crash_f.write("param_norm=%.8e\n" % param_norm_val)
                    crash_f.write("update_norm=%.8e\n" % update_norm)
                    crash_f.write("update_to_param_ratio=%.8e\n" % update_to_param_ratio)
                    crash_f.write("energy_head_weight_norm=%.8e\n" % energy_head_weight_norm)
                    crash_f.write("energy_head_grad_norm=%.8e\n" % energy_head_grad_norm)
                    crash_f.write("backbone_grad_norm=%.8e\n" % backbone_grad_norm)
                    crash_f.write("clip_active=%d\n" % (1 if clip_active else 0))
                    crash_f.write("clip_active_fraction=%.8e\n" % clip_active_fraction_value)
                    crash_f.write("soft_guard_reason=%s\n" % (soft_guard_reason or ""))
                print(
                    "[CRASH_GUARD] step=%d reason=%s wrote=%s"
                    % (step, crash_guard_reason or "peer_crash_guard", crash_path),
                    flush=True,
                )
            dist.barrier(group=all_group, device_ids=[local_rank])
            raise RuntimeError(
                "stability crash guard triggered at step %d: %s"
                % (step, crash_guard_reason or "peer_crash_guard")
            )

        maybe_save_checkpoint(
            rank=rank,
            step=step,
            args=args,
            run_dir=run_dir,
            ddp_model=ddp_model,
            optimizer=optimizer,
            extra_state=(
                {
                    "local_chain_state": state_to_cpu_payload(
                        chain_state_from_buffers(
                            x=chain_buf if runtime.conditional else None,
                            y=label_buf,
                            chain_id=chain_id_buf,
                            steps_done=steps_done_buf,
                            valid=valid_buf,
                            stage_id=(stage if use_pipeline else 0),
                        )
                    ),
                    "replay_state": (replay.state_dict() if replay is not None else None),
                }
                if runtime.conditional
                else None
            ),
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    main()
