import torch

from polaris_ebm.scripts.current.ebm_train_sync_mode_a import (
    assert_pipeline_received_state,
    chain_tags,
    chain_sig_tag,
    conditional_state_signature,
    compute_stage_alpha_vector,
    parse_stage_weights,
    resolve_raw_stage_weights,
    stage2_beta_schedule,
)
from polaris_ebm.scripts.current.mode_a_contract import (
    MSG_CHAIN,
    MSG_SAMPLE,
    build_batch_plan,
    build_role_plan,
    dtype_to_id,
    id_to_dtype,
    make_sample_header,
    parse_sample_header,
)


def test_role_plan_mapping_one_to_many():
    plan = build_role_plan(world_size=8, n_trainers=2)
    assert plan.trainer_ranks == [0, 1]
    assert plan.sampler_ranks == [2, 3, 4, 5, 6, 7]
    assert plan.samplers_by_trainer[0] == [2, 4, 6]
    assert plan.samplers_by_trainer[1] == [3, 5, 7]
    assert plan.trainer_for_sampler[2] == 0
    assert plan.trainer_for_sampler[7] == 1


def test_batch_plan_even_with_remainder_distribution():
    role = build_role_plan(world_size=8, n_trainers=2)
    batch = build_batch_plan(global_batch=64, role_plan=role)
    assert batch.trainer_local_batch == 32

    assert batch.sampler_batch_by_rank[2] == 11
    assert batch.sampler_batch_by_rank[4] == 11
    assert batch.sampler_batch_by_rank[6] == 10
    assert batch.sampler_batch_by_rank[3] == 11
    assert batch.sampler_batch_by_rank[5] == 11
    assert batch.sampler_batch_by_rank[7] == 10

    trainer0_sum = sum(batch.sampler_batch_by_rank[s] for s in role.samplers_by_trainer[0])
    trainer1_sum = sum(batch.sampler_batch_by_rank[s] for s in role.samplers_by_trainer[1])
    assert trainer0_sum == batch.trainer_local_batch
    assert trainer1_sum == batch.trainer_local_batch


def test_header_roundtrip():
    h = make_sample_header(step=9, numel=3072, dtype_id=1, device="cpu")
    msg_type, step, numel, dtype_id = parse_sample_header(h)
    assert msg_type == MSG_SAMPLE
    assert step == 9
    assert numel == 3072
    assert dtype_id == 1


def test_chain_header_roundtrip():
    h = make_sample_header(step=7, numel=4096, dtype_id=0, device="cpu", msg_type=MSG_CHAIN)
    msg_type, step, numel, dtype_id = parse_sample_header(h)
    assert msg_type == MSG_CHAIN
    assert step == 7
    assert numel == 4096
    assert dtype_id == 0


def test_dtype_id_roundtrip():
    assert id_to_dtype(dtype_to_id(torch.float32)) == torch.float32
    assert id_to_dtype(dtype_to_id(torch.float16)) == torch.float16
    assert id_to_dtype(dtype_to_id(torch.bfloat16)) == torch.bfloat16


def test_chain_tags_are_stage_and_step_stable():
    h0, p0 = chain_tags(step=0, stage=0, num_stages=4)
    h1, p1 = chain_tags(step=0, stage=1, num_stages=4)
    h2, p2 = chain_tags(step=1, stage=0, num_stages=4)
    assert (h0, p0) == (200000, 200001)
    assert (h1, p1) == (200002, 200003)
    assert (h2, p2) == (200008, 200009)


def test_chain_sig_tags_are_stage_and_step_stable():
    assert chain_sig_tag(step=0, stage=0, num_stages=4) == 500000
    assert chain_sig_tag(step=0, stage=1, num_stages=4) == 500001
    assert chain_sig_tag(step=1, stage=0, num_stages=4) == 500004


def test_conditional_state_signature_is_deterministic():
    labels = torch.tensor([1, 2], dtype=torch.long)
    chain_id = torch.tensor([5, -1], dtype=torch.long)
    steps_done = torch.tensor([25, 10], dtype=torch.long)
    valid = torch.tensor([True, True], dtype=torch.bool)
    sig0 = conditional_state_signature(labels, chain_id, steps_done, valid)
    sig1 = conditional_state_signature(labels.clone(), chain_id.clone(), steps_done.clone(), valid.clone())
    assert int(sig0.item()) == int(sig1.item())


def test_assert_pipeline_received_state_rejects_signature_mismatch():
    x = torch.zeros((2, 3, 8, 8), dtype=torch.float32)
    y = torch.tensor([0, 1], dtype=torch.long)
    chain_id = torch.tensor([2, 3], dtype=torch.long)
    steps_done = torch.tensor([25, 25], dtype=torch.long)
    valid = torch.tensor([True, True], dtype=torch.bool)
    bad_sig = torch.tensor([123], dtype=torch.int64)
    try:
        assert_pipeline_received_state(
            x=x,
            y=y,
            chain_id=chain_id,
            steps_done=steps_done,
            valid=valid,
            expected_signature=bad_sig,
            num_classes=10,
            min_steps_done=25,
            rank=1,
            stage=2,
            step=7,
        )
        assert False, "expected signature mismatch to raise"
    except RuntimeError as exc:
        assert "signature mismatch" in str(exc)


def test_parse_stage_weights_default_and_broadcast():
    assert parse_stage_weights("", 4) == [1.0, 1.0, 1.0, 1.0]
    assert parse_stage_weights("0.25", 4) == [0.25, 0.25, 0.25, 0.25]
    assert parse_stage_weights("1,2,3,4", 4) == [1.0, 2.0, 3.0, 4.0]


def test_resolve_raw_stage_weights_modes():
    manual = [1.0, 2.0, 3.0, 4.0]
    assert resolve_raw_stage_weights("manual", manual, 4, gamma=2.0) == manual
    assert resolve_raw_stage_weights("uniform", manual, 4, gamma=2.0) == [1.0, 1.0, 1.0, 1.0]
    assert resolve_raw_stage_weights("linear", manual, 4, gamma=2.0) == [0.25, 0.5, 0.75, 1.0]
    assert resolve_raw_stage_weights("power", manual, 4, gamma=2.0) == [0.0625, 0.25, 0.5625, 1.0]


def test_compute_stage_alpha_vector_with_warmup():
    raw = [1.0, 1.0, 1.0, 1.0]
    assert compute_stage_alpha_vector(step=0, raw_stage_weights=raw, warmup_gate=True) == [1.0, 0.0, 0.0, 0.0]
    assert compute_stage_alpha_vector(step=1, raw_stage_weights=raw, warmup_gate=True) == [0.5, 0.5, 0.0, 0.0]
    assert compute_stage_alpha_vector(step=2, raw_stage_weights=raw, warmup_gate=True) == [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0]
    assert compute_stage_alpha_vector(step=3, raw_stage_weights=raw, warmup_gate=True) == [0.25, 0.25, 0.25, 0.25]


def test_compute_stage_alpha_vector_manual_weights_sums_to_one():
    raw = [1.0, 2.0, 3.0, 4.0]
    a_full = compute_stage_alpha_vector(step=3, raw_stage_weights=raw, warmup_gate=True)
    assert a_full == [0.1, 0.2, 0.3, 0.4]
    assert abs(sum(a_full) - 1.0) < 1e-12

    a_warm = compute_stage_alpha_vector(step=1, raw_stage_weights=raw, warmup_gate=True)
    assert a_warm == [1.0 / 3.0, 2.0 / 3.0, 0.0, 0.0]
    assert abs(sum(a_warm) - 1.0) < 1e-12


def test_stage2_beta_schedule_piecewise_linear():
    assert stage2_beta_schedule(step=0, beta_final=0.01, t0=1000, t1=2000) == 0.0
    assert stage2_beta_schedule(step=999, beta_final=0.01, t0=1000, t1=2000) == 0.0
    assert stage2_beta_schedule(step=1000, beta_final=0.01, t0=1000, t1=2000) == 0.0
    assert stage2_beta_schedule(step=1500, beta_final=0.01, t0=1000, t1=2000) == 0.005
    assert abs(stage2_beta_schedule(step=1999, beta_final=0.01, t0=1000, t1=2000) - 0.00999) < 1e-12
    assert stage2_beta_schedule(step=2000, beta_final=0.01, t0=1000, t1=2000) == 0.01
    assert stage2_beta_schedule(step=5000, beta_final=0.01, t0=1000, t1=2000) == 0.01
