# Polaris EBM 代码地图速查

## 三条训练线

| 线 | 训练入口 | 提交脚本 | eval 路径 |
| -- | -- | -- | -- |
| 单卡 baseline | `scripts/current/ebm_train_baseline_single.py` | `scripts/current/pbs_repro_orig_single_fid.sh` | PBS 内联采样 + `scripts/current/eval_metrics.py` |
| 多卡 baseline | `scripts/current/ebm_train_sync_mode_a.py --mode ddp_fullk` | `scripts/current/pbs_modea_fullk_align_singlefid.pbs` | `scripts/current/eval_generate.py` + `scripts/current/eval_metrics.py` |
| strict pipeline | `scripts/current/ebm_train_sync_mode_a.py --mode pipeline` | `scripts/current/pbs_pipeline_stage2b001_convergence_eval.pbs` | `scripts/current/eval_generate.py` + `scripts/current/eval_metrics.py` |

## 三个最该记住的提交脚本

1. `scripts/current/pbs_repro_orig_single_fid.sh`
2. `scripts/current/pbs_modea_fullk_align_singlefid.pbs`
3. `scripts/current/pbs_pipeline_stage2b001_convergence_eval.pbs`

## 三个最该记住的 eval 脚本

1. `scripts/current/eval_generate.py`
2. `scripts/current/eval_metrics.py`
3. `scripts/pbs_eval_modea_compare.pbs`

说明：

- 真正 canonical 的 eval 还是 `eval_generate.py` + `eval_metrics.py`
- `pbs_eval_modea_compare.pbs` 是现成 compare helper，但它不是最稳妥的主线入口

## 当前最佳 strict pipeline 配方

- 训练脚本：`scripts/current/ebm_train_sync_mode_a.py`
- 提交脚本：`scripts/current/pbs_pipeline_stage2b001_convergence_eval.pbs`
- 关键超参：
  - `STAGE2_BETA=0.01`
  - `lr=1e-4`
  - `steps=20000`
  - `K=100`
  - `stages_per_node=4`
  - `langevin_sign=1.0`
  - sign fix 已在 `scripts/current/ebm_train_sync_mode_a.py:952-955` 落地
- 代表结果：
  - 5k：`fid_inception=196.5274`, `unique_ratio=1.0`
  - 10k：`fid_inception=172.7536`, `unique_ratio=1.0`
  - 20k：`fid_inception=124.2257`, `unique_ratio=1.0`

## 下一步先看哪 10 个文件

1. `scripts/current/ebm_train_sync_mode_a.py`
2. `scripts/current/pbs_pipeline_stage2b001_convergence_eval.pbs`
3. `scripts/pbs_repro_orig_pipeline_strict_fid.pbs`
4. `scripts/current/pbs_modea_fullk_align_singlefid.pbs`
5. `scripts/current/mode_a_contract.py`
6. `scripts/current/eval_generate.py`
7. `scripts/current/eval_metrics.py`
8. `scripts/ebm_train_baseline_ddp_strict.py`
9. `scripts/current/ebm_train_baseline_single.py`
10. `scripts/current/pbs_repro_orig_single_fid.sh`
