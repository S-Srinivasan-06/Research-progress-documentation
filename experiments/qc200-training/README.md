# QC200 controlled training

This is the frozen training implementation for a 200 Hz, 12-second DGDCN
adaptation on TUSZ v2.0.6. It does not claim numerical reproduction of the
paper (which used v2.0.1), event-level performance, or a new AUROC result.
Historical H0 already used patient-balanced sampling. The controlled pair
focuses on the reviewed QC markers, fresh verified normalization, strict input
validation, and the B1/W1 loss-weight comparison.

## Public backup and upstream reference

The backup retains the exact active training `.py` files. The copied DGDCN
upstream tree and adjacency pickle are omitted from the public backup because
no upstream license was identified. Before a local run, use
`../qc200-training-runtime/restore_reference.py` to fetch the required files
from upstream commit `96c7bee49c240124735ffc2ab2ad118d27a1ff97` and apply the
small documented import, shape, debug-output, and device-selection repairs.
The restore helper does not grant redistribution rights. `repo/REPRODUCTION.md`
and `repo/` are local reference material, not part of the public source backup.

`build_notebook.py` is a historical local notebook builder. It embeds the
current local source tree in a notebook and is not the public restore method;
the public notebook downloads this repository and runs the pinned restore
helper instead.

## Entry points

- `preprocess_qc200.py --config PATH` audits an existing cache or creates
  features from EDFs. Positive `preprocess_limit` values are pilots; training
  requires a full cache audit and an approved feature-provenance record.
- `execute_pipeline.py --config PATH` is the staged B1 path. It restores the
  existing archive through authenticated Drive access, compares representative
  source features, runs the full feature audit, then trains B1. It creates a
  run folder under the configured Drive parent and mirrors output/control files
  with read-back verification while the pipeline is alive.
- `train_qc200.py --config PATH` starts one fresh configured run or resumes an
  explicitly selected checkpoint. It writes to `output_dir`; place that path
  on mounted Drive for direct persistence, or use `checkpoint_sync.py` only
  after a run is paused or stopped. `checkpoint_sync.py` is not a live-run
  synchronizer.
- `finalize_qc200.py --config PATH` is the separate held-out evaluation entry
  point. It requires a frozen development-only selection record, a completed
  train/dev run, and disclosure of earlier eval access. Training itself does
  not run eval.

Use a new run ID and output folder for W1. Keep B1 and W1 identical in seed,
data, markers, architecture, augmentation, normalization population, and
schedule; change only `variant`. B1 uses unweighted BCE. W1 computes its
positive-class weight from the exact selected training pool. The automated
pipeline currently covers B1; B1-to-W1 queue sequencing is still in progress
and is not represented as completed.

The optional `cpu_fallback_watch.py` runtime helper can continue an eligible
GPU-device/driver failure from `last.pt` on CPU. It needs the original Colab
runtime and watcher process to remain alive, a verified checkpoint and Drive
run folder, explicit failure evidence, and enough remaining runtime. It does
not reconnect after Colab terminates, recover arbitrary exceptions, or make
CPU execution fast.

## Current staged status

The archive restore reached 40/40 files, and the representative source/cache
comparison passed. The Colab runtime was later reset, so completion of the full
feature audit is unconfirmed. No B1 training metrics or checkpoint are
available. The user has reconnected on CPU and reauthorized Drive; this is not
evidence of a completed recovery. The W1 queue helper is prepared but has not
been deployed. A representative comparison is not a full raw-to-cache audit.
Keep private configs, Drive IDs, participant identifiers, markers, feature
payloads, checkpoints, predictions, and run outputs out of public commits.
