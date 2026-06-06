# Polaris EBM Multi-node SSH Launcher Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix multi-node SSH launcher exports (HSN binding) and add baseline/pipeline wrappers while keeping training code unchanged.

**Architecture:** Use a single robust `run_ssh.sh` that always binds to HSN and consumes env-configurable flags. Provide two small wrapper scripts that set baseline vs pipeline env vars and then exec `run_ssh.sh`.

**Tech Stack:** Bash, Python (pytest), PBS/Polaris environment.

---

### Task 1: Add failing tests for SSH launcher env and wrappers

**Files:**
- Create: `/eagle/lc-mpi/Zhiqing/polaris_ebm/tests/test_run_ssh_launcher.py`

**Step 1: Write the failing test**

```python
from pathlib import Path

BASE = Path("/eagle/lc-mpi/Zhiqing/polaris_ebm")
RUN_SSH = (BASE / "run_ssh.sh").read_text()


def test_run_ssh_exports_master_iface():
    # Ensure MASTER_IFACE is exported into the launcher env
    assert "export MASTER_IFACE" in RUN_SSH


def test_run_ssh_parameterizes_core_flags():
    # Ensure core flags are configurable via env vars (not hard-coded)
    assert "--batch_size ${GLOBAL_BATCH}" in RUN_SSH
    assert "--schedule ${SCHEDULE}" in RUN_SSH
    assert "--pipe_stages ${PIPE_STAGES}" in RUN_SSH
    assert "--max_staleness ${MAX_STALENESS}" in RUN_SSH
    assert "--log_every ${LOG_EVERY}" in RUN_SSH
    assert "--watchdog_sec ${WATCHDOG_SEC}" in RUN_SSH


def test_run_ssh_wrappers_exist_and_set_modes():
    baseline = (BASE / "run_ssh_baseline.sh")
    pipeline = (BASE / "run_ssh_pipeline.sh")
    assert baseline.exists()
    assert pipeline.exists()
    btxt = baseline.read_text()
    ptxt = pipeline.read_text()
    assert "SCHEDULE=sync_fullbatch" in btxt
    assert "PIPE_STAGES=-1" in btxt
    assert "MAX_STALENESS=0" in btxt
    assert "run_ssh.sh" in btxt
    assert "SCHEDULE=stream_fullbatch" in ptxt
    assert "PIPE_STAGES=3" in ptxt
    assert "MAX_STALENESS=2" in ptxt
    assert "run_ssh.sh" in ptxt
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest /eagle/lc-mpi/Zhiqing/polaris_ebm/tests/test_run_ssh_launcher.py -v`

Expected: FAIL because `run_ssh.sh` does not export `MASTER_IFACE`, does not use env vars for flags, and wrapper scripts do not exist yet.

**Step 3: Commit**

```bash
git add /eagle/lc-mpi/Zhiqing/polaris_ebm/tests/test_run_ssh_launcher.py
git commit -m "test: add ssh launcher env and wrapper checks"
```

(If this directory is not a git repo, skip the commit.)

---

### Task 2: Fix `run_ssh.sh` env propagation and parameterize core flags

**Files:**
- Modify: `/eagle/lc-mpi/Zhiqing/polaris_ebm/run_ssh.sh`

**Step 1: Write the failing test**

Already covered in Task 1.

**Step 2: Run test to verify it fails**

Already covered in Task 1.

**Step 3: Write minimal implementation**

Add exported variables and replace hard-coded flags with env variables + defaults (subset shown):

```bash
# defaults near top
MODEL_SCALE=${MODEL_SCALE:-small}
NF=${NF:-64}
K=${K:-100}
GLOBAL_BATCH=${GLOBAL_BATCH:-64}
STEPS=${STEPS:-2000}
LR=${LR:-1e-4}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
STEP_SIZE=${STEP_SIZE:-0.2}
NOISE_STD=${NOISE_STD:-1e-2}
PIPE_STAGES=${PIPE_STAGES:--1}
PIPE_K_SPLITS=${PIPE_K_SPLITS:-auto}
WEIGHT_STASH_LEN=${WEIGHT_STASH_LEN:--1}
X_TRANSPORT_DTYPE=${X_TRANSPORT_DTYPE:-fp16}
NEG_DTYPE=${NEG_DTYPE:-fp16}
SCHEDULE=${SCHEDULE:-sync_fullbatch}
MAX_STALENESS=${MAX_STALENESS:-0}
ALIGN_EVERY=${ALIGN_EVERY:-0}
WEIGHT_DTYPE=${WEIGHT_DTYPE:-fp32}
WEIGHT_PREFETCH=${WEIGHT_PREFETCH:-0}
DEBUG_LEVEL=${DEBUG_LEVEL:-1}
LOG_EVERY=${LOG_EVERY:-50}
WATCHDOG_SEC=${WATCHDOG_SEC:-120}
```

Ensure `MASTER_IFACE` is exported to the launcher env (either inside the here-doc or by env passing). Example inside the here-doc:

```bash
export MASTER_IFACE="${MASTER_IFACE}"
```

And update CLI flags in the launcher to use these env vars:

```bash
--batch_size ${GLOBAL_BATCH}
--schedule ${SCHEDULE}
--pipe_stages ${PIPE_STAGES}
--max_staleness ${MAX_STALENESS}
--log_every ${LOG_EVERY}
--watchdog_sec ${WATCHDOG_SEC}
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest /eagle/lc-mpi/Zhiqing/polaris_ebm/tests/test_run_ssh_launcher.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add /eagle/lc-mpi/Zhiqing/polaris_ebm/run_ssh.sh
git commit -m "fix: export master iface and parametrize ssh launcher"
```

(If this directory is not a git repo, skip the commit.)

---

### Task 3: Add baseline and pipeline SSH wrappers

**Files:**
- Create: `/eagle/lc-mpi/Zhiqing/polaris_ebm/run_ssh_baseline.sh`
- Create: `/eagle/lc-mpi/Zhiqing/polaris_ebm/run_ssh_pipeline.sh`

**Step 1: Write the failing test**

Already covered in Task 1.

**Step 2: Run test to verify it fails**

Already covered in Task 1.

**Step 3: Write minimal implementation**

`run_ssh_baseline.sh`:

```bash
#!/bin/bash
set -euo pipefail

export SCHEDULE="${SCHEDULE:-sync_fullbatch}"
export PIPE_STAGES="${PIPE_STAGES:--1}"
export MAX_STALENESS="${MAX_STALENESS:-0}"
export ALIGN_EVERY="${ALIGN_EVERY:-0}"

exec bash "$(dirname "$0")/run_ssh.sh"
```

`run_ssh_pipeline.sh`:

```bash
#!/bin/bash
set -euo pipefail

export SCHEDULE="${SCHEDULE:-stream_fullbatch}"
export PIPE_STAGES="${PIPE_STAGES:-3}"
export MAX_STALENESS="${MAX_STALENESS:-2}"
export ALIGN_EVERY="${ALIGN_EVERY:-0}"

exec bash "$(dirname "$0")/run_ssh.sh"
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest /eagle/lc-mpi/Zhiqing/polaris_ebm/tests/test_run_ssh_launcher.py -v`

Expected: PASS

**Step 5: Commit**

```bash
git add /eagle/lc-mpi/Zhiqing/polaris_ebm/run_ssh_baseline.sh \
  /eagle/lc-mpi/Zhiqing/polaris_ebm/run_ssh_pipeline.sh
git commit -m "feat: add baseline and pipeline ssh wrappers"
```

(If this directory is not a git repo, skip the commit.)

---

### Task 4: Update README exports and multi-node instructions

**Files:**
- Modify: `/eagle/lc-mpi/Zhiqing/polaris_ebm/README.md`

**Step 1: Write the failing test**

No automated test required for README updates.

**Step 2: Write minimal implementation**

Update the multi-node setup to include HSN exports and the new SSH scripts, e.g.:

```bash
export HSN_IFNAME=hsn0
export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
```

And add usage:

```bash
bash ./run_ssh_baseline.sh
bash ./run_ssh_pipeline.sh
```

**Step 3: Commit**

```bash
git add /eagle/lc-mpi/Zhiqing/polaris_ebm/README.md
git commit -m "docs: update multinode exports and ssh launchers"
```

(If this directory is not a git repo, skip the commit.)

---

## Notes
- This workspace does not appear to be a git repository, so worktree creation and commits may not be possible. If a git root exists elsewhere, switch to it before execution.
