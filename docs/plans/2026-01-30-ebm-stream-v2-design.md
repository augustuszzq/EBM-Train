# EBM Stream V2 Design (Full-batch Streaming + Bounded Staleness)

**Date:** 2026-01-30

## Goals
- Implement a clean, single-script EBM trainer + streaming sampler pipeline for Polaris.
- Full-batch streaming (M=1) with bounded staleness and stable throughput.
- Node-local sampler pipeline (stage0/1/2) + cross-node DDP trainer only.
- Strong observability and fail-fast behavior (no silent hangs).

## Non-goals
- No microbatching or complex dynamic role mapping.
- No multi-model or dynamic pipeline reconfiguration.
- No heavy unit tests for NCCL/GPUs.

## Runtime Topology
- Per node: 4 GPUs
  - local_rank 0/1/2: sampler stages (stage0/1/2)
  - local_rank 3: trainer
- Groups:
  - `lane_pg`: node-local 4 ranks for p2p stage communications
  - `trainer_pg`: all trainer ranks (global ranks 3,7,11,...) for DDP

## Message Protocol
- Header: `int32[4] = (wver_used, seq, msg_type, bs)`
  - msg_type: 0=DATA, -1=TERMINATE
- Send/recv order: header then tensor, strictly consistent for all hops.
- Hard asserts:
  - bs == local_bs
  - trainer expects seq == step
  - stage forward preserves header fields

## Weight Versioning
- Trainer produces `wver=t+1` after update step t.
- Stage0/1/2 keep a `WeightRx` with perpetual `irecv`.
- Stash is a ring buffer sized `>= max_staleness + pipe_stages + 8`.
- Stage waits only when stash missing `wver_used` (records wait time).

## Staleness Policy
- `max_staleness` bounds `stale = seq - wver_used`.
- `align_every` optionally tightens staleness at periodic steps.
- Stage0 chooses the newest version not exceeding `seq` and within bound.

## Stage Behavior
- Stage0 picks `wver_used`, applies weights, runs K_chunk Langevin steps, sends to stage1.
- Stage1/2 receive header + x, ensure required weight present, run K_chunk, forward.
- Terminate: stage0 sends TERM header at end; stages forward and exit.

## Trainer Behavior
- Receives neg batch and header from stage2.
- Asserts `seq == step`, computes loss and updates DDP model.
- Sends new weights to all stages.
- Records `f_pos/f_neg/metric`, timing, staleness.

## Observability & Fail-fast
- CSV metrics per rank; rank0 aggregates key metrics.
- `--debug_level {0,1,2}` for logging verbosity.
- `--watchdog_sec` triggers traceback + abort if stalled.
- NCCL monitoring env vars documented in PBS template.

## PBS Run Template (V2)
- Single-node: `torchrun --standalone --nproc_per_node=4 ...`
- Multi-node: `mpiexec -n ${NNODES} -ppn 1 ... torchrun --nnodes=...`
- NCCL Profile A/B switch for Polaris networking.

