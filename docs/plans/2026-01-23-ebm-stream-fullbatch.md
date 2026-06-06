# EBM Stream Fullbatch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a new `ebm_train_stream.py` that supports full‑batch streaming with bounded staleness (default), plus a `sync_fullbatch` mode for correctness baseline.

**Architecture:** New script owns the streaming loop (trainer + stage0/1/2), with fixed per‑node lane mapping. It uses a 4‑int header `(wver_used, seq, mb_id, bs)` and a `WeightRx + stash` pipeline to avoid per‑step blocking. Termination uses `seq = -1` as a sentinel propagated through the pipeline. Metrics include staleness and weight‑send timing.

**Tech Stack:** Python 3.10, PyTorch (torch.distributed), torchrun, PBS scripts.

> **Note:** This repo is not a git repository, so commit steps can be skipped unless git is initialized.

---

### Task 1: Add pure scheduling helpers (staleness logic)

**Files:**
- Create: `polaris_ebm/scripts/stream_schedule.py`
- Test: `polaris_ebm/tests/test_stream_schedule.py`

**Step 1: Write the failing test**

```python
# polaris_ebm/tests/test_stream_schedule.py
import pytest
from polaris_ebm.scripts.stream_schedule import effective_max_staleness, choose_wver_used

def test_effective_max_staleness_align_every():
    assert effective_max_staleness(seq=0, max_staleness=2, align_every=0) == 2
    assert effective_max_staleness(seq=5, max_staleness=2, align_every=0) == 2
    assert effective_max_staleness(seq=4, max_staleness=2, align_every=4) == 1
    assert effective_max_staleness(seq=5, max_staleness=2, align_every=4) == 2


def test_choose_wver_used_bounds():
    # stash_max_wver below lower bound -> wait sentinel
    assert choose_wver_used(seq=10, stash_max_wver=6, max_staleness_eff=2) is None
    # normal case: clamp to seq, but not older than bound
    assert choose_wver_used(seq=10, stash_max_wver=9, max_staleness_eff=2) == 9
    assert choose_wver_used(seq=10, stash_max_wver=20, max_staleness_eff=2) == 10
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest polaris_ebm/tests/test_stream_schedule.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing function errors.

**Step 3: Write minimal implementation**

```python
# polaris_ebm/scripts/stream_schedule.py
from __future__ import annotations
from typing import Optional

def effective_max_staleness(seq: int, max_staleness: int, align_every: int) -> int:
    if align_every and seq > 0 and (seq % align_every == 0):
        return min(max_staleness, 1)
    return max_staleness


def choose_wver_used(seq: int, stash_max_wver: int, max_staleness_eff: int) -> Optional[int]:
    wver_min = max(0, seq - max_staleness_eff)
    if stash_max_wver < wver_min:
        return None
    return min(stash_max_wver, seq)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest polaris_ebm/tests/test_stream_schedule.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add polaris_ebm/tests/test_stream_schedule.py polaris_ebm/scripts/stream_schedule.py
# git commit -m "feat: add stream schedule helpers"
```

---

### Task 2: Add header pack/unpack helpers

**Files:**
- Create: `polaris_ebm/scripts/stream_protocol.py`
- Test: `polaris_ebm/tests/test_stream_protocol.py`

**Step 1: Write the failing test**

```python
# polaris_ebm/tests/test_stream_protocol.py
import torch
from polaris_ebm.scripts.stream_protocol import pack_header, unpack_header, header_tensor


def test_pack_unpack_roundtrip():
    hdr = pack_header(wver_used=7, seq=12, mb_id=0, bs=64)
    assert hdr == (7, 12, 0, 64)
    assert unpack_header(hdr) == (7, 12, 0, 64)


def test_header_tensor_shape():
    t = header_tensor((1, 2, 0, 64), device="cpu")
    assert t.shape == (4,)
    assert t.dtype == torch.int32
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest polaris_ebm/tests/test_stream_protocol.py -v`
Expected: FAIL with missing module/functions.

**Step 3: Write minimal implementation**

```python
# polaris_ebm/scripts/stream_protocol.py
from __future__ import annotations
from typing import Tuple
import torch

def pack_header(wver_used: int, seq: int, mb_id: int, bs: int) -> Tuple[int, int, int, int]:
    return (int(wver_used), int(seq), int(mb_id), int(bs))


def unpack_header(hdr: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    return (int(hdr[0]), int(hdr[1]), int(hdr[2]), int(hdr[3]))


def header_tensor(hdr: Tuple[int, int, int, int], device: str) -> torch.Tensor:
    return torch.tensor(hdr, dtype=torch.int32, device=device)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest polaris_ebm/tests/test_stream_protocol.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add polaris_ebm/tests/test_stream_protocol.py polaris_ebm/scripts/stream_protocol.py
# git commit -m "feat: add stream header helpers"
```

---

### Task 3: Add WeightStash (unit‑testable) + WeightRx skeleton

**Files:**
- Create: `polaris_ebm/scripts/stream_weights.py`
- Test: `polaris_ebm/tests/test_stream_weights.py`

**Step 1: Write the failing test**

```python
# polaris_ebm/tests/test_stream_weights.py
import torch
from polaris_ebm.scripts.stream_weights import WeightStash

def test_stash_put_get():
    stash = WeightStash(flat_len=4, stash_len=4, device="cpu")
    w0 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    stash.put(0, w0)
    got = stash.get(0)
    assert torch.allclose(got, w0)


def test_stash_missing():
    stash = WeightStash(flat_len=4, stash_len=2, device="cpu")
    assert stash.get(5) is None
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest polaris_ebm/tests/test_stream_weights.py -v`
Expected: FAIL with missing module/functions.

**Step 3: Write minimal implementation**

```python
# polaris_ebm/scripts/stream_weights.py
from __future__ import annotations
from typing import Optional
import torch

class WeightStash:
    def __init__(self, flat_len: int, stash_len: int, device: str):
        self.flat_len = int(flat_len)
        self.stash_len = int(stash_len)
        self.device = device
        self._bufs = [torch.empty((self.flat_len,), device=device, dtype=torch.float32)
                      for _ in range(self.stash_len)]
        self._ids = [-1] * self.stash_len

    def put(self, wver: int, wflat: torch.Tensor) -> None:
        # overwrite oldest (simple FIFO)
        idx = 0
        if wver in self._ids:
            idx = self._ids.index(wver)
        else:
            idx = self._ids.index(min(self._ids))
        self._bufs[idx].copy_(wflat)
        self._ids[idx] = int(wver)

    def get(self, wver: int) -> Optional[torch.Tensor]:
        if wver in self._ids:
            return self._bufs[self._ids.index(wver)]
        return None

    def max_wver(self) -> int:
        return max(self._ids)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest polaris_ebm/tests/test_stream_weights.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add polaris_ebm/tests/test_stream_weights.py polaris_ebm/scripts/stream_weights.py
# git commit -m "feat: add weight stash"
```

---

### Task 4: Implement `ebm_train_stream.py` (new streaming script)

**Files:**
- Create: `polaris_ebm/scripts/ebm_train_stream.py`
- Modify: `polaris_ebm/scripts/pbs_train_ebm.sh`

**Step 1: Write a failing integration smoke test stub**

```python
# polaris_ebm/tests/test_stream_smoke.py
import subprocess, sys

def test_stream_script_imports():
    # Basic import sanity; skips full torchrun
    cmd = [sys.executable, "-c", "import polaris_ebm.scripts.ebm_train_stream"]
    assert subprocess.call(cmd) == 0
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest polaris_ebm/tests/test_stream_smoke.py -v`
Expected: FAIL with import error.

**Step 3: Write minimal implementation**

Implement the new script with:
- CLI args: `--schedule` (default `stream_fullbatch`), `--max_staleness`, `--align_every`, `--force_fullbatch_stream`, `--weight_dtype`, `--weight_send_async` (optional), plus existing args reused from `ebm_train_dual.py`.
- Fixed lane mapping: `local_rank 0/1/2/3 -> stage0/1/2/trainer`.
- Header: `(wver_used, seq, mb_id, bs)` using `stream_protocol` helpers.
- Termination: `seq = -1` header to end stage loops and trainer.
- Trainer loop: receive one full‑batch, compute `stale=seq-wver_used`, update model, send `wver=seq+1`.
- Stage0: choose `wver_used` via `stream_schedule`, wait if stash not ready, generate full‑batch, send header+x.
- Stage1/2: receive header+x, ensure `wver_used` in stash, apply weight, run K_chunk, forward header+x.
- Metrics: add `stale`, `stale_violation`, `weight_send_ms`, `seq_mismatch_count`, `wver_missing_wait_ms`.
- Keep `sync_fullbatch` schedule: force `wver_used=seq` and wait until available.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest polaris_ebm/tests/test_stream_smoke.py -v`
Expected: PASS

**Step 5: Manual smoke run (single node)**

Run (example):
```bash
cd /eagle/lc-mpi/Zhiqing/polaris_ebm
qsub -l select=1:system=polaris -v GLOBAL_BATCH=64,EBM_PROFILE_TIMING=1 scripts/pbs_train_ebm.sh
```
Expected: logs show `schedule=stream_fullbatch`, `seq` monotonic, `stale<=max_staleness`.

**Step 6: Commit**

```bash
git add polaris_ebm/scripts/ebm_train_stream.py polaris_ebm/scripts/pbs_train_ebm.sh polaris_ebm/tests/test_stream_smoke.py
# git commit -m "feat: add streaming trainer and update PBS launcher"
```

---

### Task 5: Update PBS script to use new script by default

**Files:**
- Modify: `polaris_ebm/scripts/pbs_train_ebm.sh`

**Step 1: Write failing test**

```python
# polaris_ebm/tests/test_pbs_script.py
from pathlib import Path

def test_pbs_uses_stream_script():
    text = Path("polaris_ebm/scripts/pbs_train_ebm.sh").read_text()
    assert "ebm_train_stream.py" in text
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest polaris_ebm/tests/test_pbs_script.py -v`
Expected: FAIL until script updated.

**Step 3: Update PBS script**

- Replace entrypoint with `ebm_train_stream.py`.
- Add `--schedule ${SCHEDULE:-stream_fullbatch}` to allow override.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest polaris_ebm/tests/test_pbs_script.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add polaris_ebm/scripts/pbs_train_ebm.sh polaris_ebm/tests/test_pbs_script.py
# git commit -m "chore: default PBS to stream trainer"
```

---

## Execution Handoff

Plan complete and saved to `polaris_ebm/docs/plans/2026-01-23-ebm-stream-fullbatch.md`.

Two execution options:
1) Subagent-Driven (this session)
2) Parallel Session (separate) using superpowers:executing-plans

Which approach?
