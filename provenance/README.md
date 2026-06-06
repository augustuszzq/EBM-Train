# Provenance Snapshot

This directory contains lightweight artifacts copied from the original working
tree so collaborators can trace code to experiment records without downloading
large run outputs.

Included:

- `runs_final_bundle/`: current CIFAR paper registry and summary files.
- `runs_long_k_multinode_sweep/`: generated configs, PBS files, and summaries
  for the 16/32-GPU long-K branch.
- `runs_long_k_batch_scaling/`: generated configs, PBS files, and summaries for
  batch-size scaling.
- `runs_long_k_scaling/`: original long-K K=200/400 branch artifacts.
- `runs_long_k_overhead_microbench/`: overhead microbenchmark configs/PBS and
  summaries.
- `runs_long_k_overhead_attribution/`: low-priority overhead attribution
  configs/PBS and summaries.
- `excluded_binary_artifacts.txt`: binary artifact scan from the export step.

Excluded:

- Checkpoints.
- Dataset caches.
- Generated images.
- FID sample directories.
- Large logs and full run directories.

The provenance files are meant for traceability, not as a complete result
bundle. For final paper tables, regenerate or collect from the original run
storage when available.

