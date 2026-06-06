# Current CIFAR Walltime: All 62 Logical Runs

No headline/appendix filtering. One row per logical run, read from each canonical run directory `job_info.json`; GPU count is from `manifest_row.json`.

- logical runs in FID table: 62
- rows written: 62
- missing job_info: 0

| phase | family | horizon | K | P | weight | seed | GPUs | wall h | GPU-h | status |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| O1 | single_P16_deepest | 300000 | 100 | 16 | deepest | 1 | 1 | 10.541944 | 10.541944 | succeeded |
| O1 | single_P16_equal | 300000 | 100 | 16 | equal | 1 | 1 | 10.520833 | 10.520833 | succeeded |
| O1 | single_P1_terminal | 300000 | 100 | 1 | terminal | 1 | 1 | 8.305833 | 8.305833 | succeeded |
| O1 | single_P2_deepest | 300000 | 100 | 2 | deepest | 1 | 1 | 8.458611 | 8.458611 | succeeded |
| O1 | single_P2_equal | 300000 | 100 | 2 | equal | 1 | 1 | 8.494167 | 8.494167 | succeeded |
| O1 | single_P4_deepest | 300000 | 100 | 4 | deepest | 1 | 1 | 8.746111 | 8.746111 | succeeded |
| O1 | single_P4_equal | 300000 | 100 | 4 | equal | 1 | 1 | 8.758889 | 8.758889 | succeeded |
| O1 | single_P8_deepest | 300000 | 100 | 8 | deepest | 1 | 1 | 9.339722 | 9.339722 | succeeded |
| O1 | single_P8_equal | 300000 | 100 | 8 | equal | 1 | 1 | 9.107778 | 9.107778 | succeeded |
| O2 | ddp_P1_terminal | 300000 | 100 | 1 | terminal | 1 | 4 | 8.388889 | 33.555556 | succeeded |
| O2 | pipe_P16_deepest | 300000 | 100 | 16 | deepest | 1 | 16 | 23.533889 | 376.542222 | succeeded |
| O2 | pipe_P16_equal | 300000 | 100 | 16 | equal | 1 | 16 | 22.732778 | 363.724444 | succeeded |
| O2 | pipe_P2_deepest | 300000 | 100 | 2 | deepest | 1 | 2 | 4.951389 | 9.902778 | succeeded |
| O2 | pipe_P2_equal | 300000 | 100 | 2 | equal | 1 | 2 | 4.920000 | 9.840000 | succeeded |
| O2 | pipe_P4_deepest | 300000 | 100 | 4 | deepest | 1 | 4 | 3.593333 | 14.373333 | succeeded |
| O2 | pipe_P4_equal | 300000 | 100 | 4 | equal | 1 | 4 | 3.551389 | 14.205556 | succeeded |
| O2 | pipe_P8_deepest | 300000 | 100 | 8 | deepest | 1 | 8 | 19.687500 | 157.500000 | succeeded |
| O2 | pipe_P8_equal | 300000 | 100 | 8 | equal | 1 | 8 | 3.201111 | 25.608889 | succeeded |
| O3 | ddp_P1_terminal | 300000 | 100 | 1 | terminal | 2 | 4 | 8.374722 | 33.498889 | succeeded |
| O3 | ddp_P1_terminal | 300000 | 100 | 1 | terminal | 3 | 4 | 8.405000 | 33.620000 | succeeded |
| O3 | ddp_P1_terminal | 300000 | 100 | 1 | terminal | 4 | 4 | 8.359722 | 33.438889 | succeeded |
| O3 | ddp_P1_terminal | 300000 | 100 | 1 | terminal | 5 | 4 | 8.369722 | 33.478889 | succeeded |
| O3 | pipe_P2_equal | 300000 | 100 | 2 | equal | 2 | 2 | 4.885833 | 9.771667 | succeeded |
| O3 | pipe_P2_equal | 300000 | 100 | 2 | equal | 3 | 2 | 4.888889 | 9.777778 | succeeded |
| O3 | pipe_P4_deepest | 300000 | 100 | 4 | deepest | 2 | 4 | 3.597222 | 14.388889 | succeeded |
| O3 | pipe_P4_deepest | 300000 | 100 | 4 | deepest | 3 | 4 | 3.581111 | 14.324444 | succeeded |
| O3 | pipe_P4_equal | 300000 | 100 | 4 | equal | 2 | 4 | 3.613056 | 14.452222 | succeeded |
| O3 | pipe_P4_equal | 300000 | 100 | 4 | equal | 3 | 4 | 3.605833 | 14.423333 | succeeded |
| O3 | pipe_P4_equal | 300000 | 100 | 4 | equal | 4 | 4 | 2.475833 | 9.903333 | succeeded |
| O3 | pipe_P4_equal | 300000 | 100 | 4 | equal | 5 | 4 | 3.628056 | 14.512222 | succeeded |
| O3 | single_P1_terminal | 300000 | 100 | 1 | terminal | 2 | 1 | 8.339444 | 8.339444 | succeeded |
| O3 | single_P1_terminal | 300000 | 100 | 1 | terminal | 3 | 1 | 5.434722 | 5.434722 | succeeded |
| O3 | single_P2_deepest | 300000 | 100 | 2 | deepest | 2 | 1 | 8.443333 | 8.443333 | succeeded |
| O3 | single_P2_deepest | 300000 | 100 | 2 | deepest | 3 | 1 | 8.515556 | 8.515556 | succeeded |
| O3 | single_P2_equal | 300000 | 100 | 2 | equal | 2 | 1 | 8.523056 | 8.523056 | succeeded |
| O3 | single_P2_equal | 300000 | 100 | 2 | equal | 3 | 1 | 8.524722 | 8.524722 | succeeded |
| O4 | ddp_P1_terminal | 500000 | 100 | 1 | terminal | 1 | 4 | 13.895556 | 55.582222 | succeeded |
| O4 | ddp_P1_terminal | 500000 | 100 | 1 | terminal | 2 | 4 | 13.938056 | 55.752222 | succeeded |
| O4 | ddp_P1_terminal | 500000 | 100 | 1 | terminal | 3 | 4 | 11.665000 | 46.660000 | succeeded |
| O4 | pipe_P4_deepest | 500000 | 100 | 4 | deepest | 1 | 4 | 5.950833 | 23.803333 | succeeded |
| O4 | pipe_P4_deepest | 500000 | 100 | 4 | deepest | 2 | 4 | 5.970833 | 23.883333 | succeeded |
| O4 | pipe_P4_deepest | 500000 | 100 | 4 | deepest | 3 | 4 | 5.985833 | 23.943333 | succeeded |
| O4 | pipe_P4_equal | 500000 | 100 | 4 | equal | 1 | 4 | 5.976111 | 23.904444 | succeeded |
| O4 | pipe_P4_equal | 500000 | 100 | 4 | equal | 2 | 4 | 5.970278 | 23.881111 | succeeded |
| O4 | pipe_P4_equal | 500000 | 100 | 4 | equal | 3 | 4 | 5.994722 | 23.978889 | succeeded |
| O4 | pipe_P4_equal | 500000 | 100 | 4 | equal | 4 | 4 | 5.911944 | 23.647778 | succeeded |
| O4 | pipe_P4_equal | 500000 | 100 | 4 | equal | 5 | 4 | 5.971389 | 23.885556 | succeeded |
| O4 | single_P1_terminal | 500000 | 100 | 1 | terminal | 1 | 1 | 13.888056 | 13.888056 | succeeded |
| O4 | single_P1_terminal | 500000 | 100 | 1 | terminal | 2 | 1 | 13.891667 | 13.891667 | succeeded |
| O4 | single_P1_terminal | 500000 | 100 | 1 | terminal | 3 | 1 | 13.829167 | 13.829167 | succeeded |
| O4 | single_P2_deepest | 500000 | 100 | 2 | deepest | 1 | 1 | 14.080556 | 14.080556 | succeeded |
| O4 | single_P2_deepest | 500000 | 100 | 2 | deepest | 2 | 1 | 13.580556 | 13.580556 | succeeded |
| O4 | single_P2_deepest | 500000 | 100 | 2 | deepest | 3 | 1 | 14.073056 | 14.073056 | succeeded |
| O4 | single_P2_equal | 500000 | 100 | 2 | equal | 1 | 1 | 14.176944 | 14.176944 | succeeded |
| O4 | single_P2_equal | 500000 | 100 | 2 | equal | 2 | 1 | 14.120556 | 14.120556 | succeeded |
| O4 | single_P2_equal | 500000 | 100 | 2 | equal | 3 | 1 | 14.086944 | 14.086944 | succeeded |
| O5 | ddp_P1_terminal_K100 | 300000 | 100 | 1 | terminal | 1 | 4 | 8.403889 | 33.615556 | succeeded |
| O5 | ddp_P1_terminal_K25 | 300000 | 25 | 1 | terminal | 1 | 4 | 3.550278 | 14.201111 | succeeded |
| O5 | ddp_P1_terminal_K50 | 300000 | 50 | 1 | terminal | 1 | 4 | 5.181667 | 20.726667 | succeeded |
| O5 | pipe_P4_deepest_K100 | 300000 | 100 | 4 | deepest | 1 | 4 | 2.972778 | 11.891111 | succeeded |
| O5 | pipe_P4_equal_K100 | 300000 | 100 | 4 | equal | 1 | 4 | 3.601667 | 14.406667 | succeeded |
| O5 | pipe_P4_equal_K100 | 300000 | 100 | 4 | equal | 2 | 4 | 3.553611 | 14.214444 | succeeded |
