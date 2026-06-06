# Seed1 FID Trajectory Recheck

本文件记录对已有 `seed1` 的两个 `300k` run 做离线 checkpoint 重评估的结果。

重评估对象：

- DDP strict baseline
  - run: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_dual/modea_fullk_align128/ddp_fullk_s300k_seed1_syncinit_20260324_165322`
  - trajectory: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_analysis/seed1_fid_trajectory/ddp_seed1_local`
- Pipeline strict P4 beta=0.01
  - run: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_pipeline_strict/pipeline_strict_s300k_b001_lr1e-4_seed1_syncinit_20260324_165322`
  - trajectory: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_analysis/seed1_fid_trajectory/pipeline_seed1_local`

评估口径与原始历史最终评估保持一致：

- `eval_images = 5000`
- `K_eval = 100`
- `step_size = 1.0`
- `noise_std = 0.01`
- `langevin_sign = 1.0`
- `eval_seed = 1`
- `no_clamp_x = True`

## 1. 最终结论

- 历史最终值是可复现的，不存在“旧 run 的最终 FID 记错了”这种问题。
- DDP 最终点重评估：
  - 历史记录：`66.856748`
  - 本次重评估：`66.940922`
  - 差值：`0.084173`
- Pipeline 最终点重评估：
  - 历史记录：`55.994583`
  - 本次重评估：`56.039880`
  - 差值：`0.045298`
- 两条 run 的最终样本统计也几乎完全重合，`sample_01_mean/std` 与 `sample_m11_mean/std` 的差别都在 `1e-5` 量级。

因此，当前需要解释的不是“为什么最终 FID 对不上”，而是“为什么中后段轨迹波动这么大”。

## 2. DDP seed1 轨迹摘要

- summary: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_analysis/seed1_fid_trajectory/ddp_seed1_local/summary.json`
- best step: `280000`
- best FID: `54.805434`
- final step: `300000`
- final FID: `66.940922`
- first `<100`: `30000`
- first `<70`: `70000`
- first `<60`: `270000`

中后段代表点：

| step | FID |
| --- | ---: |
| 190k | 76.5129 |
| 195k | 61.4538 |
| 200k | 65.1155 |
| 210k | 61.6043 |
| 220k | 71.2396 |
| 240k | 82.0949 |
| 255k | 61.6619 |
| 270k | 59.3708 |
| 280k | 54.8054 |
| 290k | 64.3458 |
| 295k | 62.2779 |
| 300k | 66.9409 |

结论：

- DDP 也不是单调下降。
- 但它更像“正常的高波动训练曲线”，有局部坏点，也有更好的局部低谷。
- 真正最好点不是最终 `300k`，而是 `280k`。

## 3. Pipeline seed1 轨迹摘要

- summary: `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_analysis/seed1_fid_trajectory/pipeline_seed1_local/summary.json`
- best step: `300000`
- best FID: `56.039880`
- final step: `300000`
- final FID: `56.039880`
- first `<100`: `40000`
- first `<70`: `145000`
- first `<60`: `300000`

中后段代表点：

| step | FID |
| --- | ---: |
| 190k | 84.4345 |
| 195k | 68.8968 |
| 200k | 73.0937 |
| 205k | 85.3080 |
| 210k | 69.6333 |
| 220k | 79.0431 |
| 230k | 65.7277 |
| 240k | 67.9057 |
| 250k | 65.0169 |
| 265k | 65.0938 |
| 275k | 63.4663 |
| 285k | 76.9268 |
| 290k | 90.6679 |
| 295k | 80.5831 |
| 300k | 56.0399 |

结论：

- Pipeline 的中后段明显比 DDP 更不平滑。
- `200k~295k` 区间出现多次明显恶化，尤其 `290k` 达到 `90.6679`。
- 但最终 `300k` 又回到全程最优 `56.0399`。
- 这意味着当前 pipeline winner 不是“稳定收敛到最好”，而更像“最终落在一个非常好的 checkpoint 上”。

## 4. 当前判断

- 旧 `seed1` 的最终 FID 没问题，离线重评估已经确认这一点。
- 真正值得进一步研究的是 checkpoint 轨迹的形状：
  - DDP 的最佳点早于最终点。
  - Pipeline 的最佳点恰好是最终点，但此前大幅震荡。
- 因此下一步更稳的做法是：
  - 用同样方法复查 `seed2` 和 `seed3`
  - 看它们是否也出现：
    - DDP 最佳点早于最终点
    - Pipeline 中后段大幅震荡但最终突然回升
  - 如果这种模式跨 seed 复现，就说明这是训练动力学特征，不是偶然噪声。
