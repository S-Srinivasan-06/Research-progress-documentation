# DGDCN EEG seizure detection reproduction log

This repository records a research reproduction attempt for the Dynamic Graph Convolutional Network with Dilated Convolution (DGDCN) seizure detector. It is a project log and companion notebook, not a claim that the paper's results have been reproduced.

The work began with a critical review of EEG seizure detection and a closer examination of the DGDCN paper and its public implementation. The present experiment focuses on the paper's 12-second detection setup, adapted to the authorized TUSZ v2.0.6 data that are already stored in the project's private Drive workspace. The run uses 200 Hz features. A possible 256 Hz experiment was discussed, but it has not been prepared or run.

## Start here

- [Project progress and decisions](docs/PROJECT_PROGRESS.md) gives the chronology, current run status, and known reproduction limits.
- [Progress notebook](notebooks/project_progress.ipynb) records the setup and includes small cells for reading a sanitized run-state file and summarizing metrics when they exist.

- [Implementation audit](docs/IMPLEMENTATION_AUDIT.md) compares the paper, authors' code, and our actual training setup.

## Current status

On 2026-10-09 at 14:23 UTC, a fresh diagnostic found a replacement CPU runtime with no training files or processes. Earlier monitor calls had returned saved cell output, so the apparent stall at epoch 4 was not a reliable live observation. Direct Drive inspection recovered aggregate metrics through epoch 7, saved at 13:33 UTC. Epoch 4 remains the best development-AUROC checkpoint at 0.7694. Both `last.pt` and `best.pt`, the normalization, configuration, and metrics are present in Drive. Checkpoint contents have not yet been reopened in the replacement runtime.

Training is currently stopped. Recovery requires a GPU runtime, Drive authorization, restoration of the saved feature archives, and explicit checkpoint resume. The reason the previous runtime ended is not established. No final evaluation artifact was found in the saved run folder. See [the recorded development metrics](results/development_metrics.jsonl) and [the diagnosis](docs/PROJECT_PROGRESS.md#10-runtime-replacement-and-stale-monitor-output).

## Reproduction setup

The working baseline is adapted from public DGDCN code at commit `96c7bee49c240124735ffc2ab2ad118d27a1ff97`. The plan uses 12-second clips, 19 electrode nodes from the released implementation, log-amplitude FFT features, and a patient-balanced sampler on the official patient-separated train, development, and evaluation partitions. The model is checked on a CUDA device before archive recovery. Training-only normalization is computed from the training clips.

This is an adaptation, not a strict replication of the article. The paper describes TUSZ v2.0.1 and 22 derivations; this experiment uses TUSZ v2.0.6 and the 19-node representation in the released code. Changes to preprocessing, data version, sampling, and compute environment must accompany any reported result.

## Data handling

The EEG corpus is restricted research data. This public record contains no recordings, labels, file IDs, folder IDs, credentials, access links, or Kaggle dataset references. Obtain the corpus through its official access process and follow its data-use terms. The local outlier report was set aside at the user's direction; its suggested exclusions have not been applied.

## Reporting rule

Only report training or evaluation metrics after they appear in saved run outputs. Clip-level metrics are not event-level sensitivity, false alarms per day, latency, or evidence of clinical utility. The current development results are preliminary: seven epochs, one run, and no confidence intervals. Held-out evaluation is still pending. A completed model run would still be a benchmark reproduction attempt, not prospective clinical validation.
