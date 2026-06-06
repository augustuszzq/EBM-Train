from pathlib import Path


BASE = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm")
RUN_HSN = (BASE / "launchers" / "current" / "run_hsn.sh").read_text()


def test_run_hsn_has_mode_switch_and_sync_entry():
    assert 'MODE="${MODE:-baseline}"' in RUN_HSN
    assert "if [[ \"${MODE}\" == \"baseline\" ]]" in RUN_HSN
    assert "elif [[ \"${MODE}\" == \"pipeline\" ]]" in RUN_HSN
    assert "scripts/current/ebm_train_sync_mode_a.py" in RUN_HSN
    assert "--n_trainers" in RUN_HSN


def test_run_hsn_uses_single_mpi_launcher_path():
    assert "mpiexec -n \"${WORLD_SIZE}\" -ppn \"${PPN}\"" in RUN_HSN
    assert "export RANK=${OMPI_COMM_WORLD_RANK" in RUN_HSN
    assert "export WORLD_SIZE=${OMPI_COMM_WORLD_SIZE" in RUN_HSN
    assert "export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK" in RUN_HSN


def test_run_hsn_wrappers_exist_and_set_modes():
    baseline = (BASE / "launchers" / "current" / "run_hsn_baseline.sh")
    pipeline = (BASE / "launchers" / "current" / "run_hsn_pipeline.sh")
    assert baseline.exists()
    assert pipeline.exists()
    btxt = baseline.read_text()
    ptxt = pipeline.read_text()
    assert 'MODE="${MODE:-baseline}"' in btxt
    assert "run_hsn.sh" in btxt
    assert 'MODE="${MODE:-pipeline}"' in ptxt
    assert "run_hsn.sh" in ptxt
