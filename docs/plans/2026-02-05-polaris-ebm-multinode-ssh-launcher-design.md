# Polaris EBM Multi-node SSH Launcher Design (2026-02-05)

## Context
Multi-node runs in `polaris_ebm` intermittently fail with NCCL socket connection errors (e.g., attempts to connect to `10.201.*`). This points to the wrong network interface being used for TCPStore/GLOO/NCCL bootstrap. The working reference in `/eagle/lc-mpi/Zhiqing/ebm/polaris_scaling_20250816_024916` explicitly binds to `hsn0` and resolves the master node’s HSN IP.

## Goals
- Keep training code unchanged (`scripts/ebm_train_stream_v2.py`).
- Provide reliable multi-node launchers that bind to HSN (`hsn0`).
- Expose baseline and pipeline variants similar to the single-node scripts.

## Decision
Use an SSH-per-node launcher that:
- Resolves master HSN IP.
- Exports `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` on every node.
- Uses `torchrun` with `--nnodes/--node_rank/--master_addr/--master_port`.

Add two wrappers:
- `run_ssh_baseline.sh` (sync_fullbatch, no pipeline stages).
- `run_ssh_pipeline.sh` (stream_fullbatch, 3 stages).

## Implementation Plan
1. Fix `run_ssh.sh` to correctly propagate `MASTER_IFACE` into the per-node launcher so `GLOO_SOCKET_IFNAME` is always set.
2. Add `run_ssh_baseline.sh` and `run_ssh_pipeline.sh` wrappers that set schedule-specific env and exec `run_ssh.sh`.
3. Update `README.md` exports to include HSN-related env vars and point to the new SSH scripts for multi-node.

## Error Handling
- Validate `PBS_NODEFILE` exists (must be in `qsub -I`).
- Ensure `MASTER_ADDR` is non-empty.
- Validate `GLOBAL_BATCH % NNODES == 0` early.

## Testing
- 1-node sanity: run `run_ssh_baseline.sh` to confirm exports and log output.
- 2-node debug: run `run_ssh_baseline.sh` and `run_ssh_pipeline.sh`, check logs for HSN IP (`MASTER_ADDR`) and absence of `10.201.*` connections.

## Out of Scope
- Changes to training code.
- Alternative MPI-based launchers.
