# EBM Stream V2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a new single-file `ebm_train_stream_v2.py` implementing full-batch streaming with bounded staleness and a matching PBS launch script.

**Architecture:** One script owns all roles (stage0/1/2 + trainer) with fixed lane mapping, explicit protocol headers, WeightRx+stash for non-blocking weight updates, and strong observability + watchdogs.

**Tech Stack:** Python 3.10, PyTorch (torch.distributed/NCCL), torchrun, PBS on Polaris.

---

### Task 1: Helper functions + protocol roundtrip (TDD)

**Files:**
- Create: `polaris_ebm/tests/test_ebm_stream_v2_helpers.py`
- Create: `polaris_ebm/scripts/ebm_train_stream_v2.py`

**Step 1: Write the failing test**

```python
# polaris_ebm/tests/test_ebm_stream_v2_helpers.py
import torch
import polaris_ebm.scripts.ebm_train_stream_v2 as v2


def test_effective_max_staleness():
    assert v2.effective_max_staleness(seq=0, max_staleness=2, align_every=0) == 2
    assert v2.effective_max_staleness(seq=4, max_staleness=2, align_every=4) in (0, 1)


def test_choose_wver_used():
    assert v2.choose_wver_used(seq=10, max_wver=6, max_staleness=2) is None
    assert v2.choose_wver_used(seq=10, max_wver=9, max_staleness=2) == 9
    assert v2.choose_wver_used(seq=10, max_wver=20, max_staleness=2) == 10


def test_header_roundtrip_cpu():
    h = v2.make_header(wver=3, seq=7, msg_type=0, bs=64, device="cpu")
    assert h.dtype == torch.int32
    assert v2.parse_header(h) == (3, 7, 0, 64)
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest polaris_ebm/tests/test_ebm_stream_v2_helpers.py -v`
Expected: FAIL with import or missing function errors.

**Step 3: Write minimal implementation**

```python
# in polaris_ebm/scripts/ebm_train_stream_v2.py
HEADER_DTYPE = torch.int32
MSG_DATA = 0
MSG_TERM = -1


def make_header(wver, seq, msg_type, bs, device):
    return torch.tensor([int(wver), int(seq), int(msg_type), int(bs)], dtype=HEADER_DTYPE, device=device)


def parse_header(tensor):
    return (int(tensor[0].item()), int(tensor[1].item()), int(tensor[2].item()), int(tensor[3].item()))


def effective_max_staleness(seq, max_staleness, align_every):
    if align_every > 0 and seq > 0 and (seq % align_every == 0):
        return min(max_staleness, 1)
    return max_staleness


def choose_wver_used(seq, max_wver, max_staleness):
    if max_wver < 0:
        return None
    wver_min = max(0, seq - max_staleness)
    if max_wver < wver_min:
        return None
    return min(max_wver, seq)
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest polaris_ebm/tests/test_ebm_stream_v2_helpers.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add polaris_ebm/tests/test_ebm_stream_v2_helpers.py polaris_ebm/scripts/ebm_train_stream_v2.py
# git commit -m "feat: add v2 protocol and schedule helpers"
```

---

### Task 2: WeightStash + WeightRx (stash tested)

**Files:**
- Create: `polaris_ebm/tests/test_ebm_stream_v2_weights.py`
- Modify: `polaris_ebm/scripts/ebm_train_stream_v2.py`

**Step 1: Write the failing test**

```python
# polaris_ebm/tests/test_ebm_stream_v2_weights.py
import torch
import polaris_ebm.scripts.ebm_train_stream_v2 as v2


def test_weight_stash_put_get():
    stash = v2.WeightStash(flat_len=4, stash_len=4, device="cpu")
    w0 = torch.tensor([1.0, 2.0, 3.0, 4.0])
    stash.put(0, w0)
    got = stash.get(0)
    assert torch.allclose(got, w0)


def test_weight_stash_max_wver():
    stash = v2.WeightStash(flat_len=2, stash_len=2, device="cpu")
    stash.put(3, torch.tensor([1.0, 2.0]))
    stash.put(5, torch.tensor([3.0, 4.0]))
    assert stash.max_wver() == 5
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest polaris_ebm/tests/test_ebm_stream_v2_weights.py -v`
Expected: FAIL with missing class errors.

**Step 3: Write minimal implementation**

```python
class WeightStash:
    def __init__(self, flat_len, stash_len, device):
        self.flat_len = int(flat_len)
        self.stash_len = int(stash_len)
        self.device = device
        self.buf = [None for _ in range(self.stash_len)]
        self.ver = [-1 for _ in range(self.stash_len)]
        self._max_wver = -1

    def put(self, wver, flat_tensor):
        idx = int(wver) % self.stash_len
        self.buf[idx] = flat_tensor.detach().clone()
        self.ver[idx] = int(wver)
        if int(wver) > self._max_wver:
            self._max_wver = int(wver)

    def get(self, wver):
        idx = int(wver) % self.stash_len
        if self.ver[idx] != int(wver):
            return None
        return self.buf[idx]

    def max_wver(self):
        return self._max_wver
```

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest polaris_ebm/tests/test_ebm_stream_v2_weights.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add polaris_ebm/tests/test_ebm_stream_v2_weights.py polaris_ebm/scripts/ebm_train_stream_v2.py
# git commit -m "feat: add v2 weight stash"
```

---

### Task 3: Core pipeline + trainer loop

**Files:**
- Modify: `polaris_ebm/scripts/ebm_train_stream_v2.py`

**Step 1: Implement argument parsing and role mapping**
- Add CLI args for schedule, staleness, align_every, debug_level, watchdog, etc.
- Define lane/trainer groups and rank mapping.

**Step 2: Implement WeightRx (irecv + poll) and apply_weight helpers**
- Non-blocking weight pipeline, no per-step blocking.

**Step 3: Implement sampler stages**
- Stage0: choose wver_used, generate neg, send header + x.
- Stage1/2: recv header + x, wait for weight if missing, compute, forward.
- Termination propagation.

**Step 4: Implement trainer loop**
- recv header + neg, assert seq, compute loss, DDP step, send weights.

**Step 5: Manual smoke run (no automated test)**

Run (single node, short):
```
torchrun --standalone --nproc_per_node=4 \
  polaris_ebm/scripts/ebm_train_stream_v2.py \
  --schedule sync_fullbatch --steps 5 --K 3 --max_staleness 0
```
Expected: completes 5 steps, no hangs, seq monotonic.

**Step 6: Commit**

```bash
git add polaris_ebm/scripts/ebm_train_stream_v2.py
# git commit -m "feat: add v2 streaming pipeline"
```

---

### Task 4: Metrics + watchdog + debug levels

**Files:**
- Modify: `polaris_ebm/scripts/ebm_train_stream_v2.py`

**Step 1: Add per-rank and aggregated CSV writers**
- Trainer rank0 writes global metrics.
- Stages write recv/compute/send/weight_wait + stale.

**Step 2: Add watchdog**
- Track last_progress_time and dump traceback on timeout.

**Step 3: Manual smoke run**
- Same as Task 3, verify CSV output created.

**Step 4: Commit**

```bash
git add polaris_ebm/scripts/ebm_train_stream_v2.py
# git commit -m "feat: add v2 metrics and watchdog"
```

---

### Task 5: PBS template for Polaris

**Files:**
- Create: `polaris_ebm/scripts/pbs_train_ebm_stream_v2.sh`

**Step 1: Write PBS script with Profile A/B toggles**
- Include NCCL vars and a fallback profile.
- Single-node and multi-node launch paths.

**Step 2: Verify script syntax**

Run: `bash -n polaris_ebm/scripts/pbs_train_ebm_stream_v2.sh`
Expected: no errors.

**Step 3: Commit**

```bash
git add polaris_ebm/scripts/pbs_train_ebm_stream_v2.sh
# git commit -m "feat: add v2 PBS launcher"
```

