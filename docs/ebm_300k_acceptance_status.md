# 300k Acceptance Status

Last updated: 2026-03-25 UTC

## Single baseline

- Run dir: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_baseline/orig_single_semantics_fid_s300000_K100_20260324_001055`
- Samples: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_baseline/orig_single_semantics_fid_s300000_K100_20260324_001055/baseline_samples.pt`
- Grid image: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_baseline/orig_single_semantics_fid_s300000_K100_20260324_001055/baseline_grid.png`
- Metrics: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_baseline/orig_single_semantics_fid_s300000_K100_20260324_001055/metrics_compare.json`
- Train meta: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_baseline/orig_single_semantics_fid_s300000_K100_20260324_001055/train_meta.json`
- FID (Inception): `61.65055605301666`
- FID (feature): `0.5418381401279113`
- Unique ratio: `1.0`
- Training-only time: `27531.898766756058 s` (`7:38:51.9`)
- End-to-end time: `27582 s` (`7:39:42`)
- Notes: original run produced `baseline_samples.pt` and metrics; `baseline_grid.png` was rendered afterward from the saved samples for review consistency.

## DDP strict baseline

- Job id: `6974361.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`
- Run dir: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_dual/modea_fullk_align128/ddp_fullk_s300k_seed1_syncinit_20260324_165322`
- Log: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_dual/modea_fullk_align128/ddp_fullk_s300k_seed1_syncinit_20260324_165322/logs/train_eval.log`
- Metrics: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_dual/modea_fullk_align128/ddp_fullk_s300k_seed1_syncinit_20260324_165322/eval/metrics_compare_20260325_055841.json`
- Grid image: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_dual/modea_fullk_align128/ddp_fullk_s300k_seed1_syncinit_20260324_165322/eval/baseline_gen_20260325_055841/grid.png`
- Samples: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_dual/modea_fullk_align128/ddp_fullk_s300k_seed1_syncinit_20260324_165322/eval/baseline_gen_20260325_055841/samples.pt`
- Status: `finished`
- Exit status: `0`
- FID (Inception): `66.85674838265982`
- FID (feature): `0.4363502421222769`
- Unique ratio: `1.0`
- Diversity trace: `8.480574274267594`
- End-to-end time: `30394 s` (`8:26:34`)
- Config: `SEED=1`, `EVAL_SEED=1`, `SYNC_FRESH_INIT=1`, `STEPS=300000`, `SAVE_EVERY=5000`
- Notes: this run finished cleanly under PBS and wrote final eval artifacts.

## Pipeline strict

- Job id: `6974362.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov`
- Run dir: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_pipeline_strict/pipeline_strict_s300k_b001_lr1e-4_seed1_syncinit_20260324_165322`
- Metrics: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_pipeline_strict/pipeline_strict_s300k_b001_lr1e-4_seed1_syncinit_20260324_165322/eval/metrics_compare_step299999_20260325_011109.json`
- Convergence summary: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_pipeline_strict/pipeline_strict_s300k_b001_lr1e-4_seed1_syncinit_20260324_165322/eval/convergence_summary.json`
- Last-1k stats: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_pipeline_strict/pipeline_strict_s300k_b001_lr1e-4_seed1_syncinit_20260324_165322/eval/last1k_stats.json`
- Grid image: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_pipeline_strict/pipeline_strict_s300k_b001_lr1e-4_seed1_syncinit_20260324_165322/eval/pipeline_gen_step299999_20260325_011109/grid.png`
- Samples: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_pipeline_strict/pipeline_strict_s300k_b001_lr1e-4_seed1_syncinit_20260324_165322/eval/pipeline_gen_step299999_20260325_011109/samples.pt`
- Status: `finished`
- Exit status: `0`
- FID (Inception): `55.994582500105366`
- FID (feature): `0.45003938602726734`
- Unique ratio: `1.0`
- Diversity trace: `8.857848344407158`
- Last-1k max_f_neg mean: `10.458973`
- Last-1k max_abs_chain mean: `1.4159449999999996`
- Last-1k fneg2/fneg3 mean: `0.4249434432998346`
- End-to-end time: `13139 s` (`3:38:59`)
- Config: `STAGE2_BETA=0.01`, `LR=1e-4`, `SEED=1`, `EVAL_SEED=1`, `SYNC_FRESH_INIT=1`, `STEPS=300000`

## Sync-init note

- Multi-GPU runs above use shared fresh-init tensors via `SYNC_FRESH_INIT=1`.
- Semantics: rank 0 samples the fresh-init tensor, then broadcasts that exact tensor to the other participating ranks.
- This guarantees the per-rank fresh initialization values are copied from the same source tensor rather than independently re-sampled.

## Acceptance checklist

- Single baseline 300k: finished, recorded.
- Pipeline strict 300k: finished, recorded.
- DDP strict 300k: finished, recorded.
