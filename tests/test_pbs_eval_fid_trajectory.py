from pathlib import Path


PBS = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/current/pbs_eval_fid_trajectory.pbs").read_text()


def test_fid_trajectory_pbs_uses_preemptable_queue():
    assert "#PBS -q preemptable" in PBS


def test_fid_trajectory_pbs_exports_llm_env_python():
    assert 'export EVAL_PYTHON="/home/kevienzzq/.conda/envs/llm-env/bin/python"' in PBS


def test_fid_trajectory_pbs_invokes_eval_fid_trajectory_script():
    assert '"${PROJECT_DIR}/scripts/current/eval_fid_trajectory.py"' in PBS
    assert 'exec "${cmd[@]}"' in PBS
