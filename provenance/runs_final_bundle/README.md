# Final Experiment Bundle

This directory collects the current experiment summaries and status snapshots.

## Snapshot
- `pipeline_then_weighting`: done=18, running=0, queued=0, failed_history=6
- `imagenet32_phasea`: completed=9/9

## Included Files
- `phase1_summary.csv`
- `phase1_summary.json`
- `phase1_summary.md`
- `phase2_summary.csv`
- `phase2_summary.json`
- `phase2_summary.md`
- `ablation_500k_summary.csv`
- `ablation_500k_summary.json`
- `ablation_500k_summary.md`
- `ablation_replay_points.csv`
- `ablation_final_summary.csv`
- `ablation_final_summary.json`
- `seed_fid_trajectory_points.csv`
- `pipeline_then_weighting_replay_points.csv`
- `pipeline_then_weighting_replay_summary.csv`
- `pipeline_then_weighting_replay_summary.json`
- `pipeline_then_weighting_current.csv`
- `pipeline_then_weighting_current.json`
- `imagenet32_phasea_current.csv`
- `imagenet32_phasea_current.json`
- `pipeline_then_weighting_manifest.csv`
- `pipeline_then_weighting_manifest_submitted.csv`
- `pipeline_then_weighting_local_submitted.csv`
- `pipeline_then_weighting_replay_manifest.csv`
- `imagenet32_phasea_manifest.csv`
- `imagenet32_phasea_submitted.csv`

## Notes
- `pipeline_then_weighting` local single runs are merged with PBS runs so completed local 300k jobs are not mistaken for old failed queue attempts.
- `imagenet32_phasea` currently records train completion status; evaluation JSONs are included if present.