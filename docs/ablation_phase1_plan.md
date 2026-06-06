# Phase 1 Ablation Plan

## 目标

这次不是直接铺开全部 `300k` 大矩阵，而是先把“实验故事线 + manifest + 自动提交 + 自动收数”固定下来，然后用一轮 `20k` screening 先判断趋势。这样做有三个直接好处：

1. 先回答最关键的 scientific questions，再决定哪一支值得扩到 `300k`。
2. 把训练、评估、收数和汇总表格一次性标准化，后续不再手工补脚本。
3. 当前汇报更需要“结构化证据”，不是更多零散 run。

## Phase 0: 已有 milestone reference

这部分不重跑，只在 collector 中作为 reference 条目插入：

1. `single strict baseline`, `300k`, `K=100`
2. `ddp strict`, `300k`, `K=100`
3. `pipeline strict`, `300k`, `K=100`, `stage2_beta=0.01`, `lr=1e-4`

它们回答的是“当前主线各自已经能到什么位置”，不是本轮 screening 的候选矩阵。

## Phase 1: Screening Batch

本轮 screening 的默认训练长度固定为：

- `steps = 20000`
- `seed = 1`
- `eval_images = 5000`
- `save_every = 5000`
- `vis_every = 1000`

除非组内有特别说明，否则全部沿用这组设置。筛选标准优先看：

- `fid_inception`
- `fid_feature`
- `unique_ratio`
- `diversity_trace`
- end-to-end walltime

## Scientific Questions

### Q0. Baseline alignment

问题：在没有 pipeline、没有 weighted-sum 的情况下，single strict 与 DDP strict 分别处在什么位置。

意义：这给出所有后续 ablation 的坐标原点。A0 reference 负责记录 `300k` 里程碑，A1/A3 里的 control 则负责给出 `20k` screening 里的相对位置。

### Q1. Weighted-sum factor

问题：completion-aware weighted-sum 这个机制本身是否有效。

设计：使用 `single_pipe_emul`，在一张卡上串行模拟 `P=4` 的 strict pipeline 语义，严格保持

`L_t = f_pos - Σ_s α_s(t) * f_neg(x_{t,s})`

这里权重只加在 loss 的负项上，不对图像 tensor 做加权平均，也不把 training 结果混回 sampling。这样可以把“weighted-sum factor”从多卡通信里剥离出来。

### Q2. Pipeline factor

问题：在固定 weighting 规则后，pipeline stage 数从 `2 -> 4 -> 8` 会如何影响质量和时间。

设计：只让 `world_size = pipe_stages`，每个实验只对应一个完整 pipeline group，先避免引入 node-local trainer/sampler 混编的额外复杂性。主比较行设置为 `uniform` 和 `last2_beta=0.01` 两种 mixture。

### Q3. Sampling fidelity factor

问题：当前方法的收益，是否只是因为每张卡每步采样更浅。

设计：做 `K sweep`，并把 `K=400, P=4` 单独标成关键点。因为这时 pipeline 每张卡单次 slice 推进 `100` 步，已经等于旧的 full-K=100 baseline 的单次深度，所以它不是“浅链 shortcut”。

### Q4. Langevin step size factor

问题：方法是否只在某个幸运的 `epsilon` 上有效。

设计：对 baseline 与 ours 同时做 `step_size` sweep，观察两条线对采样步长的敏感性差异。

## 权重规则

所有新的 Phase 1 实验都统一使用 completion-aware active-stage 规则：

### `uniform`

对当前 active stages 平均：

`alpha_s = 1 / (#active stages)`

### `deep_only`

只使用最深 active stage：

`alpha_deepest = 1`

### `last2_beta`

只使用最深两层 active stage。若 active stage 少于两层，则自动退化：

- 只有一个 active stage 时，该 stage 权重为 `1`
- 有两层及以上时，倒数第二深权重为 `beta`，最深权重为 `1-beta`

这条规则同时用于：

- `single_pipe_emul`
- `pipe_strict`

## 为什么这轮只做 screening，不直接跑 full 300k matrix

原因不是算力不够，而是当前最需要先澄清“哪一个 factor 值得扩展”。如果现在直接上 full matrix，会同时承担：

- 长 walltime
- 大量排队成本
- 收数口径不统一
- 最终难以把结果组织成一条清晰故事线

因此更合理的策略是：

1. 用 `20k` 先测方向
2. 用统一 collector 自动汇总
3. 再从每组挑出最值得放大的点去做 `300k`

## 分组说明

### A0. Existing milestones

只登记 reference，不提交。

### A1. Weighted-sum on single GPU

目标是隔离 weighted-sum factor。单卡 control 与单卡 pipeline emulation 使用尽量统一的基础设施，只让 weighting 规则不同。

### A2. Pipeline stage-count

目标是隔离 pipeline factor。比较 `P=2/4/8` 在相同 `K=100` 下的质量和时间变化。

### A3. K sweep

目标是回答“收益是否只是来自更浅采样”。这是最直接的 fidelity 因子实验。

### A4. Step-size sweep

目标是回答“方法是否只在某个幸运 step size 上有效”。如果 ours 在较宽的 `epsilon` 范围内仍保持优势，论证会更稳。

## 产物规范

所有训练 run 统一落在：

`runs_ablation/<group>/<exp_id>_<timestamp>/`

训练产物至少包含：

- `config_resolved.json`
- `manifest_row.json`
- `job_info.json`
- `train.log`
- `checkpoints/`
- `vis/`

评估产物统一落在 `eval/` 下，至少包含：

- `metrics_compare.json`
- `stats.json`
- `grid.png`

collector 最终输出：

- `runs_ablation/phase1_summary.csv`
- `runs_ablation/phase1_summary.json`
- `runs_ablation/phase1_summary.md`
