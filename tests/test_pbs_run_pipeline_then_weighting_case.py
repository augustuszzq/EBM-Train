from pathlib import Path


PBS = Path(
    "/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/scripts/current/pbs_run_pipeline_then_weighting_case.pbs"
).read_text()


def test_pipeline_then_weighting_wrapper_invokes_ablation_launcher_via_bash():
    assert 'exec /usr/bin/bash "${TARGET}"' in PBS
    assert 'exec "${TARGET}"' not in PBS
