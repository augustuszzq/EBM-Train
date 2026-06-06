# Manifest Diff Vs Claimed Plan

This report compares the claimed O1-O5 plan against actual filesystem / scheduler state.

## Actual State For O1-O5 Logical Runs

| phase_group | completed | running | queued | failed | missing |
| --- | ---: | ---: | ---: | ---: | ---: |
| O1 | 9 | 0 | 0 | 0 | 0 |
| O2 | 9 | 0 | 0 | 0 | 0 |
| O3 | 16 | 0 | 0 | 0 | 0 |
| O4 | 18 | 0 | 0 | 0 | 0 |
| O5 | 5 | 0 | 0 | 0 | 0 |

## Duplicated Logical Runs (multiple attempts / resume fragments)

| logical_run_id | attempts | status | canonical_run_dir |
| --- | ---: | --- | --- |
| `current_cifar_paper:cifar10:ddp_P1_terminal:seed3:h500000` | 2 | completed | `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_objective_first_followup/O4/O4_ddp_fullk_K100_seed3_500k_resume_20260415_152638` |
| `current_cifar_paper:cifar10:ddp_P1_terminal_K100:seed1:h300000` | 2 | completed | `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_objective_first_followup/O5/O5_ddp_fullk_K100_seed1_300k_20260416_184021` |
| `current_cifar_paper:cifar10:ddp_P1_terminal_K25:seed1:h300000` | 2 | completed | `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_objective_first_followup/O5/O5_ddp_fullk_K25_seed1_300k_20260416_184021` |
| `current_cifar_paper:cifar10:ddp_P1_terminal_K50:seed1:h300000` | 2 | completed | `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_objective_first_followup/O5/O5_ddp_fullk_K50_seed1_300k_20260416_184021` |
| `current_cifar_paper:cifar10:pipe_P4_deepest:seed1:h500000` | 2 | completed | `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_objective_first_followup/O4/O4_pipe_strict_P4_deep_K100_seed1_500k_20260416_184021` |
| `current_cifar_paper:cifar10:pipe_P4_deepest:seed2:h500000` | 2 | completed | `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_objective_first_followup/O4/O4_pipe_strict_P4_deep_K100_seed2_500k_20260416_184021` |
| `current_cifar_paper:cifar10:pipe_P4_deepest:seed3:h500000` | 2 | completed | `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_objective_first_followup/O4/O4_pipe_strict_P4_deep_K100_seed3_500k_20260416_184021` |
| `current_cifar_paper:cifar10:pipe_P4_deepest_K100:seed1:h300000` | 2 | completed | `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_objective_first_followup/O5/O5_pipe_strict_P4_deep_K100_seed1_300k_resume_20260421_223303` |
| `current_cifar_paper:cifar10:pipe_P4_equal:seed2:h500000` | 2 | completed | `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_objective_first_followup/O4/O4_pipe_strict_P4_uniform_K100_seed2_500k_retry_20260415_152638` |
| `current_cifar_paper:cifar10:pipe_P4_equal:seed4:h300000` | 2 | completed | `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_objective_first_followup/O3/O3_pipe_strict_P4_uniform_K100_seed4_300k_resume_20260412_143159` |
| `current_cifar_paper:cifar10:pipe_P8_equal:seed1:h300000` | 3 | completed | `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_objective_first_followup/O2/O2_pipe_strict_P8_uniform_K100_seed1_300k_resume_20260424_033325` |
| `current_cifar_paper:cifar10:single_P1_terminal:seed2:h500000` | 2 | completed | `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_objective_first_followup/O4/O4_single_fullk_K100_seed2_500k_retry_20260421_223303` |
| `current_cifar_paper:cifar10:single_P1_terminal:seed3:h300000` | 4 | completed | `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_objective_first_followup/O3/O3_single_fullk_K100_seed3_300k_resume_20260421_223303` |
| `current_cifar_paper:cifar10:single_P2_deepest:seed2:h500000` | 2 | completed | `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_objective_first_followup/O4/O4_single_pipe_emul_P2_deep_K100_seed2_500k_resume_20260415_152638` |
| `current_cifar_paper:cifar10:single_P8_equal:seed1:h300000` | 2 | completed | `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_objective_first_followup/O1/O1_single_pipe_emul_P8_uniform_K100_seed1_300k_resume_20260421_223303` |
| `historical_exploratory:mainline_500k:ddp_fullk_K100_seed1_500k:ddp_P1_terminal:seed1:h500000` | 2 | completed | `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation_500k/P500K/ddp_fullk_K100_seed1_500k_resume_20260401_152616` |
| `historical_exploratory:mainline_500k:ddp_fullk_K100_seed2_500k:ddp_P1_terminal:seed2:h500000` | 2 | completed | `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation_500k/P500K/ddp_fullk_K100_seed2_500k_resume_20260401_152616` |
| `historical_exploratory:mainline_500k:ddp_fullk_K100_seed3_500k:ddp_P1_terminal:seed3:h500000` | 2 | completed | `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation_500k/P500K/ddp_fullk_K100_seed3_500k_resume_20260401_152616` |
| `historical_exploratory:mainline_500k:single_fullk_K100_seed1_500k:single_P1_terminal:seed1:h500000` | 2 | completed | `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation_500k/P500K/single_fullk_K100_seed1_500k_resume_20260401_152616` |
| `historical_exploratory:phase1_screening:A1_single_fullk_k100_s20k:single_P1_terminal:seed1:h20000` | 2 | completed | `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation/A1/A1_single_fullk_k100_s20k_20260329_040631` |
| `historical_exploratory:phase1_screening:A1_single_pipe_deep_p4_k100_s20k:single_P4_deepest:seed1:h20000` | 2 | completed | `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation/A1/A1_single_pipe_deep_p4_k100_s20k_20260329_040631` |
| `historical_exploratory:phase1_screening:A1_single_pipe_last2b001_p4_k100_s20k:single_P4_late_stage_beta_0.01:seed1:h20000` | 2 | completed | `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation/A1/A1_single_pipe_last2b001_p4_k100_s20k_20260329_040631` |
| `historical_exploratory:phase1_screening:A1_single_pipe_uniform_p4_k100_s20k:single_P4_equal:seed1:h20000` | 2 | completed | `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation/A1/A1_single_pipe_uniform_p4_k100_s20k_20260329_040631` |
| `historical_exploratory:phase1_screening:A4_ddp_fullk_eps005_k100_s20k:ddp_P1_terminal:seed1:h20000` | 2 | completed | `/eagle/lc-mpi/Zhiqing/polaris_ebm/runs_ablation/A4/A4_ddp_fullk_eps005_k100_s20k_20260330_031604` |
| `historical_exploratory:pipeline_then_weighting:E1_single_fullk_K100_s300k:single_P1_terminal:seed1:h300000` | 2 | completed | `/lus/eagle/projects/lc-mpi/Zhiqing/polaris_ebm/runs_pipeline_then_weighting/E1/E1_single_fullk_K100_s300k_local_20260404_151327` |

## Missing Current-CIFAR Logical Runs

- None

## Historical Packages Excluded From Current Paper Ledger

- `A0`
- `A1`
- `A2`
- `A3`
- `A4`
- `E1`
- `E2`
- `M1`
- `M2`
- `P2A0`
- `P500K`
- `S1`
- `S2`

## Cross-Ledger / Cross-Package Duplicate Identities

- `cifar10 / ddp_P1_terminal / seed=1 / horizon=20000` appears in historical_exploratory::A3, historical_exploratory::A4
- `cifar10 / ddp_P1_terminal / seed=1 / horizon=300000` appears in current_cifar_paper::O2, historical_exploratory::A0, historical_exploratory::E1, historical_exploratory::P2A0
- `cifar10 / ddp_P1_terminal / seed=1 / horizon=500000` appears in current_cifar_paper::O4, historical_exploratory::P500K
- `cifar10 / ddp_P1_terminal / seed=2 / horizon=300000` appears in current_cifar_paper::O3, historical_exploratory::M1
- `cifar10 / ddp_P1_terminal / seed=2 / horizon=500000` appears in current_cifar_paper::O4, historical_exploratory::P500K
- `cifar10 / ddp_P1_terminal / seed=3 / horizon=300000` appears in current_cifar_paper::O3, historical_exploratory::M1
- `cifar10 / ddp_P1_terminal / seed=3 / horizon=500000` appears in current_cifar_paper::O4, historical_exploratory::P500K
- `cifar10 / pipe_P16_deepest / seed=1 / horizon=300000` appears in current_cifar_paper::O2, historical_exploratory::E1
- `cifar10 / pipe_P2_deepest / seed=1 / horizon=300000` appears in current_cifar_paper::O2, historical_exploratory::E1
- `cifar10 / pipe_P4_deepest / seed=1 / horizon=300000` appears in current_cifar_paper::O2, historical_exploratory::E1, historical_exploratory::E2
- `cifar10 / pipe_P4_equal / seed=1 / horizon=300000` appears in current_cifar_paper::O2, historical_exploratory::E2
- `cifar10 / pipe_P4_late_stage_beta_0.01 / seed=1 / horizon=20000` appears in historical_exploratory::A2, historical_exploratory::A3, historical_exploratory::A4
- `cifar10 / pipe_P4_late_stage_beta_0.01 / seed=1 / horizon=300000` appears in historical_exploratory::A0, historical_exploratory::E2, historical_exploratory::P2A0
- `cifar10 / pipe_P8_deepest / seed=1 / horizon=300000` appears in current_cifar_paper::O2, historical_exploratory::E1
- `cifar10 / single_P16_deepest / seed=1 / horizon=300000` appears in current_cifar_paper::O1, historical_exploratory::E1
- `cifar10 / single_P1_terminal / seed=1 / horizon=300000` appears in current_cifar_paper::O1, historical_exploratory::A0, historical_exploratory::E1, historical_exploratory::P2A0
- `cifar10 / single_P1_terminal / seed=1 / horizon=500000` appears in current_cifar_paper::O4, historical_exploratory::P500K
- `cifar10 / single_P2_deepest / seed=1 / horizon=300000` appears in current_cifar_paper::O1, historical_exploratory::E1
- `cifar10 / single_P4_deepest / seed=1 / horizon=300000` appears in current_cifar_paper::O1, historical_exploratory::E1, historical_exploratory::E2, historical_exploratory::S1
- `cifar10 / single_P4_equal / seed=1 / horizon=300000` appears in current_cifar_paper::O1, historical_exploratory::E2
- `cifar10 / single_P4_late_stage_beta_0.01 / seed=1 / horizon=300000` appears in historical_exploratory::E2, historical_exploratory::S1
- `cifar10 / single_P8_deepest / seed=1 / horizon=300000` appears in current_cifar_paper::O1, historical_exploratory::E1

