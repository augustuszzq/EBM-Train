from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch


MSG_SAMPLE = 1
MSG_CHAIN = 2

DTYPE_TO_ID = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.int64: 3,
}
ID_TO_DTYPE = {v: k for k, v in DTYPE_TO_ID.items()}


@dataclass(frozen=True)
class RolePlan:
    trainer_ranks: List[int]
    sampler_ranks: List[int]
    trainer_for_sampler: Dict[int, int]
    samplers_by_trainer: Dict[int, List[int]]


@dataclass(frozen=True)
class BatchPlan:
    trainer_local_batch: int
    sampler_batch_by_rank: Dict[int, int]


def _validate_world(world_size: int, n_trainers: int) -> None:
    if world_size < 2:
        raise ValueError("world_size must be >= 2 for trainer/sampler split")
    if n_trainers <= 0:
        raise ValueError("n_trainers must be >= 1")
    if n_trainers >= world_size:
        raise ValueError("n_trainers must be < world_size")


def build_role_plan(world_size: int, n_trainers: int) -> RolePlan:
    _validate_world(world_size=world_size, n_trainers=n_trainers)
    trainer_ranks = list(range(n_trainers))
    sampler_ranks = list(range(n_trainers, world_size))
    if n_trainers + len(sampler_ranks) != world_size:
        raise ValueError("n_trainers + n_samplers must equal world_size")

    trainer_for_sampler: Dict[int, int] = {}
    samplers_by_trainer: Dict[int, List[int]] = {trank: [] for trank in trainer_ranks}
    for srank in sampler_ranks:
        trank = trainer_ranks[(srank - n_trainers) % n_trainers]
        trainer_for_sampler[srank] = trank
        samplers_by_trainer[trank].append(srank)

    for trank in trainer_ranks:
        samplers_by_trainer[trank].sort()
        if not samplers_by_trainer[trank]:
            raise ValueError(
                f"trainer rank {trank} has no samplers; reduce n_trainers or add ranks"
            )

    return RolePlan(
        trainer_ranks=trainer_ranks,
        sampler_ranks=sampler_ranks,
        trainer_for_sampler=trainer_for_sampler,
        samplers_by_trainer=samplers_by_trainer,
    )


def build_batch_plan(global_batch: int, role_plan: RolePlan) -> BatchPlan:
    n_trainers = len(role_plan.trainer_ranks)
    if global_batch <= 0:
        raise ValueError("global_batch must be > 0")
    if global_batch % n_trainers != 0:
        raise ValueError(
            f"global_batch ({global_batch}) must be divisible by n_trainers ({n_trainers})"
        )

    trainer_local_batch = global_batch // n_trainers
    sampler_batch_by_rank: Dict[int, int] = {}
    for trank in role_plan.trainer_ranks:
        srcs = role_plan.samplers_by_trainer[trank]
        base, rem = divmod(trainer_local_batch, len(srcs))
        if base == 0:
            raise ValueError(
                "trainer local batch too small for sampler fan-in; "
                f"trainer_local_batch={trainer_local_batch}, sources={len(srcs)}"
            )
        for idx, srank in enumerate(srcs):
            sampler_batch_by_rank[srank] = base + (1 if idx < rem else 0)

    return BatchPlan(
        trainer_local_batch=trainer_local_batch,
        sampler_batch_by_rank=sampler_batch_by_rank,
    )


def dtype_to_id(dtype: torch.dtype) -> int:
    if dtype not in DTYPE_TO_ID:
        raise ValueError(f"unsupported dtype for protocol: {dtype}")
    return DTYPE_TO_ID[dtype]


def id_to_dtype(dtype_id: int) -> torch.dtype:
    if int(dtype_id) not in ID_TO_DTYPE:
        raise ValueError(f"unsupported dtype id for protocol: {dtype_id}")
    return ID_TO_DTYPE[int(dtype_id)]


def make_sample_header(
    step: int, numel: int, dtype_id: int, device: str, msg_type: int = MSG_SAMPLE
) -> torch.Tensor:
    return torch.tensor(
        [int(msg_type), int(step), int(numel), int(dtype_id)],
        dtype=torch.int64,
        device=device,
    )


def parse_sample_header(header: torch.Tensor) -> Tuple[int, int, int, int]:
    if header.numel() != 4:
        raise ValueError(f"header must have 4 values, got {header.numel()}")
    return (
        int(header[0].item()),
        int(header[1].item()),
        int(header[2].item()),
        int(header[3].item()),
    )


def iter_state_tensors_sorted(module: torch.nn.Module) -> Iterable[Tuple[str, torch.Tensor]]:
    state = module.state_dict()
    for key in sorted(state.keys()):
        tensor = state[key]
        if torch.is_tensor(tensor):
            yield key, tensor
