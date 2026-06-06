from scripts.current.collect_objective_first_followup import (
    best_from_points,
    expected_trajectory_points,
    first_leq,
    infer_status,
    late_window_mean,
)


def test_first_leq_returns_first_matching_ordinal_step():
    points = [
        {"ordinal_step": "5000", "fid_inception": "80.0"},
        {"ordinal_step": "10000", "fid_inception": "69.0"},
        {"ordinal_step": "15000", "fid_inception": "60.0"},
    ]
    assert first_leq(points, 70.0) == "10000"
    assert first_leq(points, 60.0) == "15000"
    assert first_leq(points, 50.0) == ""


def test_late_window_mean_uses_selected_window():
    points = [
        {"ordinal_step": "200000", "fid_inception": "70.0"},
        {"ordinal_step": "250000", "fid_inception": "60.0"},
        {"ordinal_step": "300000", "fid_inception": "50.0"},
    ]
    assert late_window_mean(points, 200000) == 60.0
    assert late_window_mean(points, 260000) == 50.0


def test_best_from_points_extracts_summary_fields():
    points = [
        {"ordinal_step": "5000", "step": "4999", "fid_inception": "90.0"},
        {"ordinal_step": "10000", "step": "9999", "fid_inception": "70.0"},
        {"ordinal_step": "15000", "step": "14999", "fid_inception": "80.0"},
    ]
    summary = best_from_points(points)
    assert summary["best_fid"] == 70.0
    assert summary["best_step"] == 10000
    assert summary["final_traj_fid"] == 80.0
    assert summary["trajectory_points"] == 3
    assert summary["first_leq_70"] == "10000"


def test_expected_trajectory_points_and_status_fallback_to_trajectory_completion(tmp_path):
    row = {"submit": "yes", "steps": "300000"}
    metrics_path = tmp_path / "missing_metrics.json"
    assert expected_trajectory_points(row) == 60
    assert infer_status(row, str(metrics_path), {"train_status": "succeeded"}, trajectory_points=59) == "succeeded"
    assert infer_status(row, str(metrics_path), {"train_status": "succeeded"}, trajectory_points=60) == "done"
