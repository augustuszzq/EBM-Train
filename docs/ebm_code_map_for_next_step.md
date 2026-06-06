# Polaris EBM 主线代码地图（供下一步实验接手）

本文只做当前仓库的只读审计，聚焦 3 条主线：

1. 单卡 baseline
2. 多卡 baseline
3. 多卡优化版 strict pipeline

我把“当前 canonical”理解为：现在最值得你优先读、且最直接服务下一步实验修改的入口；旧脚本/实验脚本会明确标出来。

## 一、总表

| 版本 | 当前 canonical 入口训练脚本 | 当前 canonical 提交脚本 | 评估脚本 | 是否 pipeline | 是否 full-K | 当前推荐配置 |
| -- | -- | -- | -- | -- | -- | -- |
| 单卡 baseline | `scripts/current/ebm_train_baseline_single.py` | `scripts/current/pbs_repro_orig_single_fid.sh` | 训练后采样是 PBS 内联 Python；指标脚本是 `scripts/current/eval_metrics.py` | 否 | 是 | `steps=20000, K=100, batch=64, lr=1e-4, sigma_pd=3e-2, noise_std=1e-2, step_size=1.0` |
| 多卡 baseline | `scripts/current/ebm_train_sync_mode_a.py --mode ddp_fullk` | `scripts/current/pbs_modea_fullk_align_singlefid.pbs` | `scripts/current/eval_generate.py` + `scripts/current/eval_metrics.py` | 否 | 是 | `steps=5000` 或 `20000`, `K=100, batch_size=64, lr=1e-4, pos_noise_std=3e-2, noise_std=1e-2, step_size=1.0, langevin_sign=1.0` |
| 多卡优化版 | `scripts/current/ebm_train_sync_mode_a.py --mode pipeline` | `scripts/current/pbs_pipeline_stage2b001_convergence_eval.pbs` | `scripts/current/eval_generate.py` + `scripts/current/eval_metrics.py` | 是 | 否，`K` 被切成 `K_slice=K/4` | `STAGE2_BETA=0.01, lr=1e-4, steps=20000, K=100, stages_per_node=4, langevin_sign=1.0` |

补充说明：

- 单卡 baseline 还有一个更轻量的训练专用提交脚本 `scripts/pbs_ebm_baseline_1gpu.sh`，它直接调用 `scripts/current/ebm_train_baseline_single.py`，但不负责 checkpoint 化评估。
- 多卡 baseline 还有一个更“原始语义复刻”的旧版本：`scripts/ebm_train_baseline_ddp_strict.py` + `scripts/pbs_repro_orig_ddp_strict_fid.pbs`。它更像“原始单卡 snippet 的 DDP 翻版”；现在更常用、也更适合和 pipeline 对照的是 `ebm_train_sync_mode_a.py --mode ddp_fullk`。
- 多卡优化版还有一个更通用的旧/泛化 launcher：`scripts/pbs_repro_orig_pipeline_strict_fid.pbs`。现在要跑“主线 0.01 / 20k / 带 checkpoint 评估”，优先看 `scripts/current/pbs_pipeline_stage2b001_convergence_eval.pbs`。
- `scripts/run_baseline.sh`、`scripts/run_pipeline.sh`、`scripts/ebm_train_stream_v2.py`、`scripts/pbs_train_ebm_stream_v2.sh` 属于更早的 stream runtime，不是当前 strict baseline / strict pipeline 主线。

## 二、分别梳理三类版本

## A. 单卡 baseline

### 1. 训练入口

- 当前训练脚本：`scripts/current/ebm_train_baseline_single.py`
- 关键位置：
  - 参数与输出目录：`scripts/current/ebm_train_baseline_single.py:215-306`
  - 正样本采样：`scripts/current/ebm_train_baseline_single.py:327-333`
  - 负样本采样：`scripts/current/ebm_train_baseline_single.py:334-345`
  - loss：`scripts/current/ebm_train_baseline_single.py:399-406`
- 主要依赖模块：
  - 无本仓共享 runtime；模型 `F` 就定义在本文件里：`scripts/current/ebm_train_baseline_single.py:182-196`
  - 不依赖 `mode_a_contract.py`

### 2. 提交入口

- 当前“复现+FID”提交脚本：`scripts/current/pbs_repro_orig_single_fid.sh`
  - 这是最接近“原始 single-GPU snippet”的一体化脚本
  - 它没有调用 `ebm_train_baseline_single.py`，而是在 PBS 内联了一段老式训练/采样代码：`scripts/current/pbs_repro_orig_single_fid.sh:48-176`
  - 训练后调用 `scripts/current/eval_metrics.py`：`scripts/current/pbs_repro_orig_single_fid.sh:178-185`
- 当前“训练专用”bash/PBS 入口：`scripts/pbs_ebm_baseline_1gpu.sh`
  - 直接调用 `scripts/current/ebm_train_baseline_single.py`：`scripts/pbs_ebm_baseline_1gpu.sh:57-67`
- 启动方式：
  - 单 GPU，本地 `python`
  - 不用 `torchrun` / `mpiexec`

### 3. 训练语义

- 单步训练：
  - 从 CIFAR-10 全数据张量里随机抽一个 batch，当作 `x_p_d`
  - 对 `x_p_d` 加正样本噪声 `sigma_pd`
  - 从均匀分布 `U(-1,1)` fresh init 出 `x_q`
  - 对 `x_q` 做完整 `K` 步 Langevin
  - 计算 `f_pos - f_neg`，优化 `loss = -(f_pos - f_neg)`
- 负样本生成：
  - `sample_p_0(m)` fresh init：`scripts/current/ebm_train_baseline_single.py:334-335`
  - `sample_q(K, m)` 做完整 `K` 步：`scripts/current/ebm_train_baseline_single.py:338-344`
- 是否 full-K：是，单卡完整跑完全部 `K`
- 是否 pipeline：否
- stage / token / chain / ring：
  - 这个版本没有这些概念
- training 与 sampling：
  - 同一循环里紧耦合
  - 先 sampling，再立刻用当前 `x_q` 训练

### 4. 核心关键参数

- `steps`：`scripts/current/ebm_train_baseline_single.py:234`
- `K`：`scripts/current/ebm_train_baseline_single.py:233`
- `lr`：`scripts/current/ebm_train_baseline_single.py:235`
- `batch_size` / `batch`：`scripts/current/ebm_train_baseline_single.py:230-231`
- `world_size / nproc_per_node`：固定单卡，无此参数
- `langevin_sign`：没有独立参数；代码写死为 `x += step_size * grad + noise`
- `stage2_beta` / `stage_weights`：无
- `warmup_gate`：无
- `deep_only`：无

### 5. loss 形式

- `f_pos`：`f_pos = model(x_p_d).mean()`，`scripts/current/ebm_train_baseline_single.py:399`
- `f_neg`：`f_neg = model(x_q).mean()`，`scripts/current/ebm_train_baseline_single.py:400`
- `loss`：`loss = -(f_pos - f_neg)`，`scripts/current/ebm_train_baseline_single.py:401-402`
- 结论：
  - 没有 `c_stage` / `alpha_stage` / `stage2_beta`
  - `x_q` 对应代码里的整批负样本张量

### 6. 提交命令示例

```bash
qsub -v STEPS=20000,K=100,M=64,LR=1e-4 scripts/current/pbs_repro_orig_single_fid.sh
```

如果你只是想跑训练脚本本身：

```bash
qsub -v STEPS=20000,K=100,BATCH=64,LR=1e-4 scripts/pbs_ebm_baseline_1gpu.sh
```

### 7. 输出产物

- `scripts/current/ebm_train_baseline_single.py` 训练专用输出：
  - `output_dir/images/x_q_*.png`
  - `output_dir/metrics.csv`
  - 可选 `output_dir/images/x_q_*.pt`
- `scripts/current/pbs_repro_orig_single_fid.sh` 一体化输出：
  - `baseline_samples.pt`
  - `metrics_compare.json`
  - `train_meta.json`
  - `grid_baseline_256.png`
  - `train_and_sample.log`
  - `eval_metrics.log`
- 没有 checkpoint 目录
- 没有 `vis/` 或 `eval/` 子目录

### 8. 当前状态判断

- 当前它更像“单卡原始 snippet / 论文语义对齐基线”。

## B. 多卡 baseline

### 1. 训练入口

- 当前 canonical 训练脚本：`scripts/current/ebm_train_sync_mode_a.py --mode ddp_fullk`
  - 参数入口：`scripts/current/ebm_train_sync_mode_a.py:101-244`
  - full-K / non-pipeline 分支：`scripts/current/ebm_train_sync_mode_a.py:617-618`, `scripts/current/ebm_train_sync_mode_a.py:747-760`, `scripts/current/ebm_train_sync_mode_a.py:942-955`
  - 采样：`scripts/current/ebm_train_sync_mode_a.py:860-869`
  - loss：`scripts/current/ebm_train_sync_mode_a.py:948-959`
- 主要依赖模块：
  - `scripts/current/mode_a_contract.py`，用于状态广播顺序和 message header：`scripts/current/ebm_train_sync_mode_a.py:21-38`
  - 但 `ddp_fullk` 模式本身不走 chain ring 通信
- 旧的“更原始 strict DDP”脚本：
  - `scripts/ebm_train_baseline_ddp_strict.py`
  - 不依赖 `mode_a_contract.py`
  - 它更忠实复刻原始单卡语义：随机抽 full data tensor、每 rank 本地 full-K、DDP 同步梯度

### 2. 提交入口

- 当前 canonical 提交脚本：`scripts/current/pbs_modea_fullk_align_singlefid.pbs`
  - `torchrun --standalone --nproc_per_node=4 scripts/current/ebm_train_sync_mode_a.py --mode ddp_fullk`：`scripts/current/pbs_modea_fullk_align_singlefid.pbs:85-109`
  - 随后显式跑 `eval_generate.py` + `eval_metrics.py`：`scripts/current/pbs_modea_fullk_align_singlefid.pbs:134-156`
- 当前 20k 训练专用脚本：`scripts/pbs_modea_fullk_20k.pbs`
  - 同样调用 `ebm_train_sync_mode_a.py --mode ddp_fullk`：`scripts/pbs_modea_fullk_20k.pbs:63-80`
  - 但这个脚本默认不带 eval
- 旧的 strict DDP 提交脚本：`scripts/pbs_repro_orig_ddp_strict_fid.pbs`
  - 调的是旧训练脚本 `scripts/ebm_train_baseline_ddp_strict.py`：`scripts/pbs_repro_orig_ddp_strict_fid.pbs:75-93`
- 启动方式：
  - 当前 canonical：`torchrun`
  - 旧 strict DDP：也是 `torchrun`
  - 不用 `mpiexec` / `mpirun`

### 3. 训练语义

- 当前 canonical `ddp_fullk`：
  - 每个 rank 都有一个完整模型副本，DDP 跨全 world 同步梯度：`scripts/current/ebm_train_sync_mode_a.py:702-709`
  - 每个 rank 每步都本地 fresh init 一个 `chain_buf`，然后直接做完整 `K` 步 Langevin：`scripts/current/ebm_train_sync_mode_a.py:857-869`
  - 不做 pipeline，不做 ring 传链
  - 训练时 `f_neg` 直接用本 rank 的 `chain_out`：`scripts/current/ebm_train_sync_mode_a.py:949-951`
- 负样本生成：
  - 当前 canonical：每 rank full-K，本地 `chain_out`
  - 旧 strict DDP：`sample_q(K, local_batch)`，每 rank 本地 full-K：`scripts/ebm_train_baseline_ddp_strict.py:188-193`
- 是否 full-K：是
- 是否 pipeline：否
- stage / token / chain / ring：
  - 当前 canonical 脚本里变量仍然存在，但 `ddp_fullk` 不启用 pipeline
  - 可把它看作“和 pipeline 共用 runtime 的 full-K 对照组”
- training 与 sampling：
  - 同一循环里紧耦合
  - 先 sampling，再立刻本地 train；梯度由 DDP all-reduce

额外语义差异：

- 当前 canonical `ddp_fullk` 使用 `DataLoader(..., shuffle=True)`，但没有 `DistributedSampler`：`scripts/current/ebm_train_sync_mode_a.py:716-723`
- 旧 strict DDP 则更像单卡版：先把 CIFAR-10 全量堆到 GPU，然后每 rank 随机取 index：`scripts/ebm_train_baseline_ddp_strict.py:174-183`

### 4. 核心关键参数

- 当前 canonical 参数位置：
  - `steps`：`scripts/current/ebm_train_sync_mode_a.py:117-124`
  - `K`：`scripts/current/ebm_train_sync_mode_a.py:136`
  - `lr`：`scripts/current/ebm_train_sync_mode_a.py:131`
  - `batch_size`：`scripts/current/ebm_train_sync_mode_a.py:117`
  - `world_size / nproc_per_node`：PBS 里 `nproc_per_node=4`，`scripts/current/pbs_modea_fullk_align_singlefid.pbs:85-88`
  - `langevin_sign`：`scripts/current/ebm_train_sync_mode_a.py:137-142`
  - `stage2_beta` / `stage_weights`：full-K 模式无效
  - `warmup_gate`：无效
  - `deep_only`：无效
- 当前推荐对照配置：
  - `K=100`
  - `batch_size=64`
  - `lr=1e-4`
  - `noise_std=1e-2`
  - `step_size=1.0`
  - `pos_noise_std=3e-2`
  - `langevin_sign=1.0`

### 5. loss 形式

当前 canonical `ddp_fullk`：

- `f_pos`：`f_pos = ddp_model(pos).mean()`，`scripts/current/ebm_train_sync_mode_a.py:949`
- `f_neg`：`f_neg = ddp_model(chain_out.detach()).mean()`，`scripts/current/ebm_train_sync_mode_a.py:950`
- `objective = f_pos - f_neg`，因为非 pipeline 时 `c_stage = 1.0`：`scripts/current/ebm_train_sync_mode_a.py:942-951`
- `loss_sign = -1.0 if langevin_sign > 0 else 1.0`：`scripts/current/ebm_train_sync_mode_a.py:952-955`
- 当前脚本默认跑 `langevin_sign=1.0` 时，等价于最小化 `-(f_pos - f_neg)`

旧 strict DDP：

- `f_pos = train_model(x_p_d).mean()`：`scripts/ebm_train_baseline_ddp_strict.py:235`
- `f_neg = train_model(x_q).mean()`：`scripts/ebm_train_baseline_ddp_strict.py:236`
- `loss = -(f_pos - f_neg)`：`scripts/ebm_train_baseline_ddp_strict.py:237-238`

### 6. 提交命令示例

当前 canonical：

```bash
qsub -v STEPS=20000,K=100,BATCH_SIZE=64,LR=1e-4 scripts/pbs_modea_fullk_20k.pbs
```

如果你要带 eval 的 one-shot 对照：

```bash
qsub -v STEPS=5000,K=100,BATCH_SIZE=64,LR=1e-4 scripts/current/pbs_modea_fullk_align_singlefid.pbs
```

如果你要跑旧 strict DDP 复刻版：

```bash
qsub -v STEPS=20000,K=100,BATCH_SIZE=64,LR=1e-4 scripts/pbs_repro_orig_ddp_strict_fid.pbs
```

### 7. 输出产物

- 当前 canonical `pbs_modea_fullk_align_singlefid.pbs`：
  - `RUN_ROOT/logs/train_eval.log`
  - `RUN_ROOT/logs/command.txt`
  - `RUN_ROOT/artifacts/checkpoints/ckpt_step*.pt`
  - `RUN_ROOT/train_outputs/ddp_fullk_syncA_fullk_ws*/metrics_rank*.csv`
  - `RUN_ROOT/eval/baseline_gen_*/{grid.png,samples.pt,stats.json}`
  - `RUN_ROOT/eval/metrics_compare_*.json`
- 当前 canonical `pbs_modea_fullk_20k.pbs`：
  - `RUN_ROOT/logs/train.log`
  - `RUN_ROOT/artifacts/checkpoints/`
  - `RUN_ROOT/train_outputs/ddp_fullk_syncA_fullk_ws*/metrics_rank*.csv`
  - `RUN_ROOT/summary.txt`
- 旧 strict DDP：
  - 直接写到 `RUN_DIR/`
  - `checkpoints/ckpt_step*.pt`
  - `metrics.csv`
  - `baseline_samples.pt`
  - `train_meta.json`
  - `metrics_compare.json`

### 8. 当前状态判断

- 当前它是“和 pipeline strict 共用 runtime 的 full-K 对照组”；如果你要做 apples-to-apples 比较，先看这个。
- 旧 `ebm_train_baseline_ddp_strict.py` 则是“更接近原始单卡语义的 DDP 复刻版”。

## C. 多卡优化版（strict pipeline / 当前主线）

### 1. 训练入口

- 当前 canonical 训练脚本：`scripts/current/ebm_train_sync_mode_a.py --mode pipeline`
- 关键位置：
  - rank / stage / local_world_size：`scripts/current/ebm_train_sync_mode_a.py:628-671`
  - `K_slice = K / num_stages`：`scripts/current/ebm_train_sync_mode_a.py:661-665`
  - stage 权重解析：`scripts/current/ebm_train_sync_mode_a.py:393-454`
  - stage2 beta warmup：`scripts/current/ebm_train_sync_mode_a.py:456-463`
  - chain buffer 与 run dir：`scripts/current/ebm_train_sync_mode_a.py:743-763`
  - pipeline 发送/接收：`scripts/current/ebm_train_sync_mode_a.py:882-905`, `scripts/current/ebm_train_sync_mode_a.py:1015-1043`
  - token 定义：`scripts/current/ebm_train_sync_mode_a.py:1050`
  - loss：`scripts/current/ebm_train_sync_mode_a.py:915-955`
- 主要依赖模块：
  - `scripts/current/mode_a_contract.py`
  - `scripts/current/eval_generate.py`
  - `scripts/current/eval_metrics.py`

### 2. 提交入口

- 当前最应优先看的提交脚本：`scripts/current/pbs_pipeline_stage2b001_convergence_eval.pbs`
  - 它把主线配方写死为 `STAGE2_BETA=0.01, lr=1e-4`
  - 会训练并在 `5k / 10k / 20k` checkpoint 自动评估：`scripts/current/pbs_pipeline_stage2b001_convergence_eval.pbs:54-66`, `scripts/current/pbs_pipeline_stage2b001_convergence_eval.pbs:206-289`
  - 训练命令：`scripts/current/pbs_pipeline_stage2b001_convergence_eval.pbs:120-148`
- 更通用的 pipeline launcher：`scripts/pbs_repro_orig_pipeline_strict_fid.pbs`
  - 支持 `STAGE2_BETA` / `DEEP_ONLY` / `STAGE2_BETA_WARMUP` 切换：`scripts/pbs_repro_orig_pipeline_strict_fid.pbs:61-78`
  - 训练命令：`scripts/pbs_repro_orig_pipeline_strict_fid.pbs:129-176`
  - 训练后显式 eval：`scripts/pbs_repro_orig_pipeline_strict_fid.pbs:185-211`
- 更早的实验脚本：`scripts/pbs_modea_pipeline_20k_deep_s2.pbs`
  - 这是“手动 stage2/stage3 权重”的早期实验 launcher，不是当前 canonical：`scripts/pbs_modea_pipeline_20k_deep_s2.pbs:45-58`, `scripts/pbs_modea_pipeline_20k_deep_s2.pbs:74-100`
- 启动方式：
  - 当前主线单节点：`torchrun --standalone --nproc_per_node=4`
  - 多节点扩展脚本也存在：`scripts/pbs_modea_pipeline_2n_deep.pbs` 用 `mpiexec`

### 3. 训练语义

- 单步训练在做什么：
  1. 所有 rank 对齐 barrier：`scripts/current/ebm_train_sync_mode_a.py:849-850`
  2. 显式广播模型参数，减少排障时的状态漂移：`scripts/current/ebm_train_sync_mode_a.py:852-854`
  3. 每个 stage 只做一次 `K_slice` Langevin 更新：`scripts/current/ebm_train_sync_mode_a.py:856-869`
  4. 立刻把 `chain_out` 通过 node 内 ring 发给下一个 stage：`scripts/current/ebm_train_sync_mode_a.py:882-905`
  5. 同时本地做一次 train update，DDP 跨全 world 同步梯度：`scripts/current/ebm_train_sync_mode_a.py:906-959`
  6. 等待上一 stage 发来的链，再交换 `chain_buf` / `chain_next_buf`：`scripts/current/ebm_train_sync_mode_a.py:1015-1043`
- 负样本怎么产生：
  - 当前 rank 的负样本张量就是 `chain_out`
  - `chain_out` 来自 `langevin_sample(...)`：`scripts/current/ebm_train_sync_mode_a.py:860-869`
  - 训练用的是 `ddp_model(chain_out.detach()).mean()`：`scripts/current/ebm_train_sync_mode_a.py:950`
- 是否 full-K：
  - 否
  - `K=100` 在 4 stage 下被切成 `K_slice=25`
- 是否 pipeline：
  - 是
  - stage 按 node 内 `local_rank` 编号：`scripts/current/ebm_train_sync_mode_a.py:637-671`
- stage / token / chain / ring 的定义：
  - `stage`：当前 GPU 在 node 内的流水段号，等于 `local_rank`
  - `token`：`token_idx = step - stage`，`scripts/current/ebm_train_sync_mode_a.py:1050`
  - `chain`：`chain_buf` / `chain_out` / `chain_next_buf`，形状 `[local_batch, 3, 32, 32]`，`scripts/current/ebm_train_sync_mode_a.py:743-745`
  - `ring`：同一 node 内 `prev_rank -> rank -> next_rank` 的 P2P 链传递，`scripts/current/ebm_train_sync_mode_a.py:667-671`, `scripts/current/ebm_train_sync_mode_a.py:895-905`
- training 与 sampling 是否解耦：
  - 没有完全解耦
  - 它们仍在同一 step 内
  - 但 pipeline 让“上一段链传输”和“当前 rank 训练”部分重叠

### 4. 核心关键参数

- `steps`：`scripts/current/ebm_train_sync_mode_a.py:118`
- `K`：`scripts/current/ebm_train_sync_mode_a.py:136`
- `lr`：`scripts/current/ebm_train_sync_mode_a.py:131`
- `batch_size`：`scripts/current/ebm_train_sync_mode_a.py:117`
- `world_size / nproc_per_node`：PBS 里 `NPROC_PER_NODE=4`，`scripts/current/pbs_pipeline_stage2b001_convergence_eval.pbs:33`, `scripts/current/pbs_pipeline_stage2b001_convergence_eval.pbs:121`
- `langevin_sign`：`scripts/current/ebm_train_sync_mode_a.py:137-142`
- `stage2_beta` / `stage_weights`：
  - 当前最佳脚本在 PBS 里把 `STAGE_WEIGHTS=0,0,0.01,0.99`：`scripts/current/pbs_pipeline_stage2b001_convergence_eval.pbs:81-89`
  - 训练脚本里再归一化成 `alpha_stage`
- `warmup_gate`：
  - 训练脚本参数在 `scripts/current/ebm_train_sync_mode_a.py:191-194`
  - 当前最佳 run 配置里是 `WARMUP_GATE=0`，见 `runs_pipeline_strict/orig_pipeline_strict_conv_s20000_b001_lr1e-4_ws4_20260301_175900/logs/config.txt`
- `deep_only`：
  - 不是 Python runtime 参数
  - 是 PBS 层变量；在 `scripts/pbs_repro_orig_pipeline_strict_fid.pbs:55-78` 中被翻译成 `STAGE_WEIGHTS=0,0,0,1`

### 5. loss 形式

关键代码：

- `alpha_vec = compute_stage_alpha_vector(...)`：`scripts/current/ebm_train_sync_mode_a.py:930-934`
- `alpha_stage = alpha_vec[stage]`：`scripts/current/ebm_train_sync_mode_a.py:935`
- `c_stage = num_stages * alpha_stage`：`scripts/current/ebm_train_sync_mode_a.py:936`
- `f_neg = ddp_model(chain_out.detach()).mean()`：`scripts/current/ebm_train_sync_mode_a.py:950`
- `objective = f_pos - (c_stage * f_neg)`：`scripts/current/ebm_train_sync_mode_a.py:951`
- `loss_sign = -1 if langevin_sign > 0 else 1`：`scripts/current/ebm_train_sync_mode_a.py:952-955`

当前 ws4 单节点主线下，可把它写成：

```text
x_{t,s} = 当前 stage s 在 step t 的 chain_out.detach()
alpha_s(t) = w_s(t) / Σ_u w_u(t)
c_s(t) = 4 * alpha_s(t)
objective_{t,s} = f_pos - c_s(t) * f_neg(x_{t,s})
```

因为 DDP 会对 4 个 rank 的梯度做平均，所以当前主线配方的有效全局负项就是：

```text
L_t = f_pos - Σ_s alpha_s(t) * f_neg(x_{t,s})
```

这里 `x_{t,s}` 在代码里对应：

```text
scripts/current/ebm_train_sync_mode_a.py:860-869 产生的 chain_out
scripts/current/ebm_train_sync_mode_a.py:950 里被送进 ddp_model(chain_out.detach())
```

当前最佳配方 `STAGE_WEIGHTS=0,0,0.01,0.99` 时：

- `alpha_0 = 0`
- `alpha_1 = 0`
- `alpha_2 = 0.01`
- `alpha_3 = 0.99`
- 因而本地 `c_stage` 分别是 `0, 0, 0.04, 3.96`

结论非常明确：

- 权重是加在 `f_neg` 的 loss 项上
- 代码里没有对 image tensor 直接做加权平均
- 代码里也没有把“加权后的训练结果”再喂回 sampling

### 6. 提交命令示例

当前主线推荐：

```bash
qsub -v STEPS=20000,STAGE2_BETA=0.01,LR=1e-4 scripts/current/pbs_pipeline_stage2b001_convergence_eval.pbs
```

更通用的 strict pipeline launcher：

```bash
qsub -v STEPS=20000,STAGE2_BETA=0.01,LR=1e-4,K=100,BATCH_SIZE=64 scripts/pbs_repro_orig_pipeline_strict_fid.pbs
```

### 7. 输出产物

- `RUN_DIR/logs/config.txt`
- `RUN_DIR/logs/train_eval.log`
- `RUN_DIR/artifacts/checkpoints/ckpt_step*.pt`
- `RUN_DIR/train_outputs/pipeline_syncA_pipe_ws*/metrics_rank*.csv`
- `RUN_DIR/train_outputs/.../vis/step*_neg_stage*_token*.png`
- `RUN_DIR/eval/pipeline_gen_step*/{grid.png,samples.pt,stats.json}`
- `RUN_DIR/eval/metrics_compare_step*.json`
- `RUN_DIR/eval/convergence_summary.csv`
- `RUN_DIR/eval/convergence_summary.json`
- `RUN_DIR/eval/last1k_stats.json`

### 8. 当前状态判断

- 当前它就是“strict pipeline 的主线配方”。

## 三、重点解释三者的核心差异

## `single baseline` vs `ddp strict baseline` vs `pipeline strict`

### 1. 负样本生成语义

单卡 baseline：

- 一张卡自己 fresh init `x_q`
- 自己完整跑完 `K` 步 Langevin
- `x_q` 直接进入本步 loss

DDP strict baseline：

- 每个 rank 都自己 fresh init 本地负样本
- 每个 rank 都自己完整跑完 `K` 步 Langevin
- 没有跨 rank 传链
- 只有参数梯度同步

pipeline strict：

- 每个 stage 每步只跑 `K_slice = K / 4`
- stage0 在当前主线下每步 fresh init 新链：`scripts/current/ebm_train_sync_mode_a.py:857-858`
- stage1/2/3 接上一 step 来自上一 stage 的链继续推
- 每个 stage 用自己当前的 `chain_out` 参与本地 loss，再由 DDP 合成全局梯度

### 2. 参数同步语义

单卡 baseline：

- 没有同步

DDP strict baseline：

- 旧 `ebm_train_baseline_ddp_strict.py`：纯 DDP，同步梯度；日志额外 all-reduce 标量：`scripts/ebm_train_baseline_ddp_strict.py:111-116`, `scripts/ebm_train_baseline_ddp_strict.py:248-279`
- 当前 canonical `ddp_fullk`：DDP 同步梯度，外加每 step 开头显式广播参数：`scripts/current/ebm_train_sync_mode_a.py:852-854`

pipeline strict：

- 也是 DDP 同步梯度
- 另外还有 node 内 ring 链传递：`scripts/current/ebm_train_sync_mode_a.py:882-905`
- 所以它比 DDP baseline 多了一层“样本状态同步/流动”，而不是只同步参数

### 3. chain / negative sample 语义

单卡是否有 persistent chain：

- 没有
- 每步 fresh init

DDP strict 下 chain 怎么分布：

- 也没有 persistent chain
- 每个 rank 只有自己本地那一小批 fresh-init 负样本
- 每步从头采样，然后丢弃

pipeline strict 下 chain 如何在 stage 间传递：

- `chain_out` 通过 node 内 `prev_rank/next_rank` ring 传给下一段：`scripts/current/ebm_train_sync_mode_a.py:895-905`
- 当前主线因为开了 `--fresh_init`，stage0 会在每步开始时覆盖掉自己收到的返回链：`scripts/current/ebm_train_sync_mode_a.py:857-858`
- 所以当前主线更准确地说是：
  - stage0 每步注入新链
  - stage1/2/3 对这条新链做 2/4、3/4、4/4 的推进
  - stage3 发回 stage0 的链在当前配方里会被下一步 fresh init 覆盖

### 4. “权重到底加在哪里”

最终结论：

- 权重加在 `f_neg` 这一项上
- 不是给 image tensor 加权平均
- 也不是把 training 后的加权结果回灌进 sampling

当前主线代码等价于：

```text
x_{t,s} = chain_out(stage=s, step=t)
alpha_s(t) = normalize(stage_weights at step t)
L_t = f_pos - Σ_s alpha_s(t) * f_neg(x_{t,s})
```

实现层面，当前 rank 的本地式子是：

```text
objective_{t,s} = f_pos - c_s(t) * f_neg(x_{t,s})
c_s(t) = num_stages * alpha_s(t)
```

这是因为 DDP 会对 rank 梯度做平均，所以本地乘一个 `num_stages`，全局平均后正好还原成 `Σ_s alpha_s(t) * f_neg(x_{t,s})`。

## 四、pipeline strict 的一步时序：T7 / S8

这里按当前主线配方写死：

- `K=100`
- `stages_per_node=4`
- 所以 `K_slice=25`
- 当前脚本开了 `--fresh_init`
- 当前权重是 `STAGE_WEIGHTS=0,0,0.01,0.99`

### T7 时，各 GPU 手里的链

| stage / GPU | 在 T7 时的链状态 | 完成度 | token | 对 loss 的权重 |
| -- | -- | -- | -- | -- |
| stage0 / GPU0 | 本步 fresh init 的新链，经 stage0 推完 25 步 | `25 / 100` | `7` | `alpha_0=0`, `c_stage=0` |
| stage1 / GPU1 | 来自 GPU0 在 T6 产出的链，经 stage1 再推 25 步 | `50 / 100` | `6` | `alpha_1=0`, `c_stage=0` |
| stage2 / GPU2 | 来自 GPU1 在 T6 产出的链，经 stage2 再推 25 步 | `75 / 100` | `5` | `alpha_2=0.01`, `c_stage=0.04` |
| stage3 / GPU3 | 来自 GPU2 在 T6 产出的链，经 stage3 再推 25 步 | `100 / 100` | `4` | `alpha_3=0.99`, `c_stage=3.96` |

这个表直接对应代码：

- `token = step - stage`：`scripts/current/ebm_train_sync_mode_a.py:1050`
- `k_slice = K / num_stages`：`scripts/current/ebm_train_sync_mode_a.py:661-665`
- 当前 loss 权重来自 `alpha_stage -> c_stage`：`scripts/current/ebm_train_sync_mode_a.py:930-936`

### T7 完成以后，S8 各 GPU 会推进什么链

| GPU | 下一步 sampling 结果 |
| -- | -- |
| GPU0 | 会先收到 GPU3 在 T7 发回来的 token4，但在 S8 开始时被 `fresh_init` 覆盖，所以实际推进的是一个全新的 token8 |
| GPU1 | 推进 GPU0 在 T7 产出的 token7 |
| GPU2 | 推进 GPU1 在 T7 产出的 token6 |
| GPU3 | 推进 GPU2 在 T7 产出的 token5 |

这点一定要注意：

- 当前主线不是“同一条链在 4 个 stage 间绕圈 persistent 地反复跑”
- 当前主线是“stage0 每步注入新链；stage1/2/3 把它推进到更深 stage；stage3 回传的链在下一步 stage0 会被 fresh init 覆盖”

## 五、当前最佳 strict pipeline 配方（供下一步实验使用）

### 训练脚本路径

- `scripts/current/ebm_train_sync_mode_a.py`

### 提交脚本路径

- 首选：`scripts/current/pbs_pipeline_stage2b001_convergence_eval.pbs`
- 通用版：`scripts/pbs_repro_orig_pipeline_strict_fid.pbs`

### 当前关键超参

- `STAGE2_BETA=0.01`
- `lr=1e-4`
- `steps=20000`
- `K=100`
- `batch_size=64`
- `stages_per_node=4`
- `langevin_sign=1.0`
- `noise_std=1e-2`
- `step_size=1.0`
- `pos_noise_std=3e-2`
- sign fix 已合入：
  - `scripts/current/ebm_train_sync_mode_a.py:952-955`
  - 现在 loss 会根据 `langevin_sign` 自动选符号，训练目标和采样方向一致

### 当前代表性结果

来自：

- `runs_pipeline_strict/orig_pipeline_strict_conv_s20000_b001_lr1e-4_ws4_20260301_175900/eval/convergence_summary.csv`

结果：

| checkpoint | fid_inception | unique_ratio |
| -- | -- | -- |
| 5k (`ckpt_step4999.pt`) | `196.5274` | `1.0` |
| 10k (`ckpt_step9999.pt`) | `172.7536` | `1.0` |
| 20k (`ckpt_step19999.pt`) | `124.2257` | `1.0` |

同一个 summary 里还记录了：

- `fid_feature`: `4.2913 / 1.9758 / 2.3128`
- `fneg2_over_fneg3_last_1k_mean`: `0.2985`

### 为什么它现在是主线

原因有 3 个：

1. 训练入口和 full-K baseline 共用 `ebm_train_sync_mode_a.py`，结构上最好比较。
2. sign-aware loss 已经在主脚本里落地，当前 `langevin_sign=1.0` 的采样/训练目标一致。
3. 现有 sweep 里，`stage2_beta=0.01` 是明显更优的 strict 配方之一：
   - `0.01 @ 5k`：FID `157.66`
   - `0.02 @ 5k`：FID `190.67`
   - `0.03 @ 5k`：FID `164.90`
   - `0.05 @ 5k`：FID `222.12`
   - `deep_only_signfix @ 5k`：FID `206.70`
   - `warmup 0->0.01 @ 5k`：FID `261.26`

## 六、下一步改代码必须先看的文件清单

```text
1. scripts/current/ebm_train_sync_mode_a.py
   作用：当前 full-K baseline 和 strict pipeline 的共用主训练 runtime。
   为什么下一步必须先看：stage / token / chain / ring / c_stage / alpha_stage / sign fix 全在这里。

2. scripts/current/pbs_pipeline_stage2b001_convergence_eval.pbs
   作用：当前最佳 strict pipeline 配方的主提交脚本。
   为什么下一步必须先看：0.01 / 20k / 5k-10k-20k 自动评估都写死在这里。

3. scripts/pbs_repro_orig_pipeline_strict_fid.pbs
   作用：更通用的 strict pipeline launcher。
   为什么下一步必须先看：deep_only、stage2_beta、warmup 的 PBS 层逻辑全在这里。

4. scripts/current/pbs_modea_fullk_align_singlefid.pbs
   作用：当前 full-K baseline 的 canonical train+eval 提交脚本。
   为什么下一步必须先看：它决定了你现在拿什么 full-K 对照 pipeline。

5. scripts/current/mode_a_contract.py
   作用：mode-A runtime 的 header / dtype / state traversal 协议。
   为什么下一步必须先看：如果你要改 pipeline 链传输格式，这是最先会踩到的共用模块。

6. scripts/current/eval_generate.py
   作用：从 checkpoint 生成样本。
   为什么下一步必须先看：现在 full-K 和 pipeline 的离线采样都走它，而且它复用了主训练脚本里的 Langevin 实现。

7. scripts/current/eval_metrics.py
   作用：统一算 feature-FID / inception-FID / unique_ratio / diversity / energy gap。
   为什么下一步必须先看：当前所有“结果好不好”的最后落点都在这里。

8. scripts/ebm_train_baseline_ddp_strict.py
   作用：旧的 strict DDP baseline 训练脚本。
   为什么下一步必须先看：如果你要追最原始单卡语义的 DDP 翻版，它比 mode-A fullk 更接近原始 snippet。

9. scripts/pbs_repro_orig_ddp_strict_fid.pbs
   作用：旧 strict DDP baseline 的 train+eval 提交脚本。
   为什么下一步必须先看：它把旧 strict DDP 路线和 eval_generate / eval_metrics 串起来了。

10. scripts/current/ebm_train_baseline_single.py
    作用：单卡 baseline 的当前训练脚本。
    为什么下一步必须先看：单卡无 pipeline、无 DDP 的最直接对照代码就在这里。

11. scripts/current/pbs_repro_orig_single_fid.sh
    作用：单卡原始 snippet 的一体化复现脚本。
    为什么下一步必须先看：如果你要确认“论文/原始代码语义到底是什么”，它最有参考价值。

12. scripts/pbs_ebm_baseline_1gpu.sh
    作用：单卡 baseline 训练专用提交脚本。
    为什么下一步必须先看：你只想快速改单卡训练入口时，它比 repro 脚本更干净。
```

## 七、最短结论

- 单卡 baseline 主看：`scripts/current/ebm_train_baseline_single.py` + `scripts/current/pbs_repro_orig_single_fid.sh`
- 多卡 baseline 主看：`scripts/current/ebm_train_sync_mode_a.py --mode ddp_fullk` + `scripts/current/pbs_modea_fullk_align_singlefid.pbs`
- 多卡优化版主看：`scripts/current/ebm_train_sync_mode_a.py --mode pipeline` + `scripts/current/pbs_pipeline_stage2b001_convergence_eval.pbs`
- 现在最重要的代码点：
  - `chain_out` 就是 `x_{t,s}`
  - 权重加在 `f_neg` loss 项
  - 当前 pipeline 主线里，stage0 每步 fresh init，stage1/2/3 才是把 token 往深处推的真正流水线
