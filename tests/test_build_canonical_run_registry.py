from scripts.current.build_canonical_run_registry import (
    config_hash,
    config_signature,
    execution_semantics,
    include_in_current_story,
    normalize_ptw_story_bucket,
    normalize_status,
    objective_family,
)


def test_execution_semantics_and_roles_are_stable():
    assert execution_semantics("single_fullk") == "single_strict"
    assert execution_semantics("ddp_fullk") == "ddp_strict"
    assert execution_semantics("single_pipe_emul") == "single_emulation"
    assert execution_semantics("pipe_strict") == "strict_pipeline"


def test_objective_family_derivation_matches_current_story():
    assert objective_family("single_fullk", "uniform") == "strict_baseline"
    assert objective_family("single_pipe_emul", "uniform") == "uniform_weighted"
    assert objective_family("pipe_strict", "deep_only") == "deep_only_weighted"
    assert objective_family("pipe_strict", "last2_beta") == "late_stage_weighted"


def test_status_and_include_rules_are_explicit():
    assert normalize_status("done_local") == "completed"
    assert normalize_status("F:0") == "completed"
    assert normalize_status("F:126") == "historical_failed"
    assert include_in_current_story("pipeline_then_weighting", "completed") == "yes"
    assert include_in_current_story("phase2_mainline", "completed") == "context_only"
    assert include_in_current_story("pipeline_then_weighting", "historical_failed") == "no"


def test_config_hash_changes_with_story_identity():
    row_a = {
        "benchmark": "cifar10",
        "conditioned": "no",
        "train_mode": "single_pipe_emul",
        "world_size": "1",
        "pipe_stages": "4",
        "seed": "1",
        "K": "100",
        "steps": "300000",
        "lr": "1e-4",
        "step_size": "1.0",
        "weight_mode": "uniform",
        "last2_beta": "0.01",
        "data_dir": "/tmp/cifar10",
        "story_bucket": "objective_first_weighting_single",
    }
    row_b = dict(row_a, weight_mode="deep_only")
    sig_a = config_signature(row_a)
    sig_b = config_signature(row_b)
    assert sig_a != sig_b
    assert config_hash(sig_a) != config_hash(sig_b)


def test_story_bucket_classifier_is_explicit():
    assert normalize_ptw_story_bucket("E1_single_pipe_emul_P4_deep_K100_s300k") == "objective_first_p_sweep_single"
    assert normalize_ptw_story_bucket("E2_pipe_strict_uniform_P4_K100_s300k") == "objective_first_weighting_pipeline"
