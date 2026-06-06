from pathlib import Path


PBS = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/current/pbs_run_ablation_case.pbs").read_text()


def test_ablation_pbs_uses_torchrun_for_single_node_distributed():
    assert '--standalone --nproc_per_node="${WORLD_SIZE}"' in PBS


def test_ablation_pbs_uses_node_level_mpi_launcher_for_multi_node():
    assert 'mpiexec -n "${nnodes}" -ppn 1 --hostfile "${PBS_NODEFILE}"' in PBS
    assert 'NODE_RANK=${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-${PALS_RANKID:-0}}}}' in PBS
    assert 'launch_cmd="${TORCHRUN} --nnodes=${nnodes} --node_rank=\\${NODE_RANK} --nproc_per_node=${PPN}' in PBS
    assert 'eval "${launch_cmd}"' in PBS
    assert "printf -v launch_cmd_str '%q '" not in PBS


def test_ablation_pbs_does_not_launch_one_mpi_process_per_training_rank():
    assert 'mpiexec -n "${WORLD_SIZE}" -ppn "${PPN}" --hostfile "${PBS_NODEFILE}" bash -lc' not in PBS
    assert 'export RANK=${OMPI_COMM_WORLD_RANK}' not in PBS


def test_ablation_pbs_propagates_resume_checkpoint_when_present():
    assert 'RESUME_CKPT="${RESUME_CKPT:-}"' in PBS
    assert 'export DATA_DIR SYNC_FRESH_INIT DEBUG_LEVEL LOG_EVERY DRY_RUN RESUME_CKPT CONFIG CONFIG_JSON JOB_INFO_JSON' in PBS
    assert 'if [[ -n "${RESUME_CKPT}" ]]; then' in PBS
    assert 'train_args+=(--resume_ckpt "${RESUME_CKPT}")' in PBS


def test_ablation_pbs_propagates_benchmark_config_when_present():
    assert 'CONFIG="${CONFIG:-}"' in PBS
    assert 'export DATA_DIR SYNC_FRESH_INIT DEBUG_LEVEL LOG_EVERY DRY_RUN RESUME_CKPT CONFIG CONFIG_JSON JOB_INFO_JSON' in PBS
    assert 'if [[ -n "${CONFIG}" ]]; then' in PBS
    assert 'cmd+=(--config "${CONFIG}")' in PBS
    assert 'train_args+=(--config "${CONFIG}")' in PBS
