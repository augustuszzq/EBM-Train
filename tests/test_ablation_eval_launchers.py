from pathlib import Path


PBS_EVAL = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/current/pbs_eval_ablation_case.pbs").read_text()
EVAL_RUNTIME = Path("/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/current/run_eval_ablation_case.sh").read_text()


def test_pbs_eval_wrapper_delegates_to_reusable_runtime_script():
    assert "run_eval_ablation_case.sh" in PBS_EVAL
    assert 'exec "${PROJECT_DIR}/scripts/current/run_eval_ablation_case.sh"' in PBS_EVAL


def test_eval_runtime_exports_project_pythonpath_for_summary_step():
    assert 'PROJECT_PARENT="$(cd "${PROJECT_DIR}/.." && pwd)"' in EVAL_RUNTIME
    assert 'export PYTHONPATH="${PROJECT_PARENT}${PYTHONPATH:+:$PYTHONPATH}"' in EVAL_RUNTIME
