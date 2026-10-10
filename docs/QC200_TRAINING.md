# QC200 controlled training: source and runtime guide

Last updated 2026-10-11. This is a separate controlled implementation for
12-second, 200 Hz clips from TUSZ v2.0.6. It is an adaptation of DGDCN, not an
exact reproduction of the paper's TUSZ v2.0.1 experiment. No new B1/W1 metrics
are reported here.

## Public backup and model provenance

The public backup preserves the frozen experiment entry-point and training
files. It does not vendor the copied upstream model tree or adjacency pickle
because the inspected upstream snapshot has no declared license. The
`restore_reference.py` helper fetches only required files from
Open-EXG/DGDCN-EEG-Seizure-Detection at commit
`96c7bee49c240124735ffc2ab2ad118d27a1ff97` and applies the documented local
import, model-shape, debug-output, and device-selection repairs. Runtime
restoration is not a license grant. The B1 run must retain its exact source
snapshot with its run artifacts.

The historical `build_notebook.py` bundles files from a local working tree and
is not the public notebook builder. Use
[`qc200_training.ipynb`](../notebooks/qc200_training.ipynb), which fetches the
public backup and invokes the pinned restore helper. The notebook has empty
outputs and no private config, IDs, raw data, checkpoints, or predictions.

## Current stage

The existing archive restore reached 40/40 files; a representative source/cache
comparison passed. The Colab runtime later reset, so completion of the full
feature audit is unconfirmed. There are no B1 training metrics or checkpoint.
A separate CPU recovery attempt was launched after reconnecting and
reauthorizing Drive. The W1 queue is deployed and waiting for B1 completion.
The
representative comparison is not a complete raw-to-cache equivalence audit.
Earlier H0 results in the project log are historical. H0 already used
patient-balanced sampling; the controlled pair is focused on the reviewed QC
markers, fresh verified normalization, strict validation, and the loss-weight
comparison.

## Entry points and persistence

| Entry point | Role | Persistence and guardrails |
|---|---|---|
| `preprocess_qc200.py --config PATH` | Audit an existing feature cache or generate features from EDF inputs. | A positive row limit is pilot-only. Training requires full cache validation and an approved provenance record. |
| `execute_pipeline.py --config PATH` | Staged B1 path: restore the existing archive, compare representative source features, validate the full cache, then train B1. | Authenticated Drive API access is required. It creates a fresh run folder, performs byte read-back checks, and mirrors run/control artifacts while the process is alive. It is not the W1 queue runner. |
| `train_qc200.py --config PATH` | Start a fresh configured B1 or W1 run, or resume an explicitly selected checkpoint. | Use a unique run ID and output folder per variant. Writing `output_dir` on mounted Drive is direct persistence. `checkpoint_sync.py` copies only a paused or stopped run and does not synchronize a live trainer. |
| `finalize_qc200.py --config PATH` | Separate held-out evaluation after training and development selection are complete. | Requires a frozen dev-only selection record, matching checkpoint/run identity, and disclosure of previous eval access. Training itself does not run eval. |
| `cpu_fallback_watch.py` | Continue eligible failed B1 training on CPU from `last.pt`. | Requires the same live Colab runtime and watcher, explicit CUDA device/driver failure evidence, no live original trainer, an existing checkpoint and verified Drive folder, and enough runtime before the supplied deadline. It cannot recover after runtime termination or mask other errors. |

B1 uses unweighted BCE. W1 computes its positive-class weight from the exact
selected training pool. Keep seed, inputs, markers, architecture, augmentation,
normalization population, and schedule fixed; change only the variant. The
staged pipeline covers B1. The separate `queue_w1.py` supervisor waits for
B1 train/dev completion and persistence, then starts W1 in its own run folder
if at least 30 minutes of trainer time plus 15 minutes for final synchronization
remain. It uses a working GPU if available, otherwise CPU. It leaves W1 queued
when the runtime budget is insufficient. A full runtime reset stops the queue.

## Reporting limits

This implementation has not produced new controlled-run metrics at the current
stage. Clip-level classification does not establish event sensitivity, false
alarms per hour, detection latency, or clinical utility. Event-level reporting
requires a validated continuous scoring and event-matching protocol. Do not
publish configs with credentials or Drive IDs, corpus locators, participant
identifiers, private markers/features, run outputs, checkpoints, or prediction
rows.
