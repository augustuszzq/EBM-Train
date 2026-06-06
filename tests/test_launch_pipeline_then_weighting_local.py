from pathlib import Path


def test_local_launcher_script_exists():
    path = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm/scripts/current/launch_pipeline_then_weighting_local.py")
    assert path.exists()
