# DGDCN EEG seizure detection reproduction log

This repository records a research reproduction attempt for the Dynamic Graph Convolutional Network with Dilated Convolution (DGDCN) seizure detector. It is a project log and companion notebook, not a claim that the paper's results have been reproduced.

The work began with a critical review of EEG seizure detection and a closer examination of the DGDCN paper and its public implementation. The present experiment focuses on the paper's 12-second detection setup, adapted to the authorized TUSZ v2.0.6 data that are already stored in the project's private Drive workspace. The run uses 200 Hz features. A possible 256 Hz experiment was discussed, but it has not been prepared or run.

## Start here

- [Project progress and decisions](docs/PROJECT_PROGRESS.md) gives the chronology, current run status, and known reproduction limits.
- [Progress notebook](notebooks/project_progress.ipynb) records the setup and includes small cells for reading a sanitized run-state file and summarizing metrics when they exist.
- [Implementation audit](docs/IMPLEMENTATION_AUDIT.md) compares the paper, authors' code, and our actual training setup.
- [Sensitivity diagnosis](docs/SENSITIVITY_DIAGNOSIS.md) explains the completed clip-level result and measured sensitivity-first threshold tradeoffs.
- [Data quality review](docs/DATA_QUALITY_REVIEW.md) explains the verified anomaly policy, annotation convention, and requirements for the next training view.
- [Sanitized data quality counts](results/data_quality_review.json) contains aggregate scan and marker reconciliation results only.
- [Sanitized final evaluation metrics](results/final_eval_metrics.json) contains aggregate counts and scores only.
- [Primary operating point](results/primary_operating_point.json) records the prespecified development-recall-0.95 cutoff and its eval clip metrics.
- [All measured threshold comparisons](results/sensitivity_threshold_report.json) contains sanitized aggregate threshold tradeoffs.

## Current status

As of the 2026-10-10 verification, the run completed eight epochs and stopped by the configured early-stopping rule. The best checkpoint is epoch 4 by development AUROC. The original held-out evaluation reported AUROC 0.8401, clip recall 0.2860, and specificity 0.9685 at threshold 0.701882, selected by maximum development F1. A later threshold-only analysis selected cutoffs on development positives and evaluated them once on held-out clips. Its primary dev-0.95 cutoff gives eval recall 0.9758, with a 77.57% false-positive rate and 7.32% precision. This is an operating-point tradeoff, not improved model discrimination; event detection was not evaluated. See the [baseline metrics](results/final_eval_metrics.json), [primary operating point](results/primary_operating_point.json), [threshold report](results/sensitivity_threshold_report.json), [sensitivity diagnosis](docs/SENSITIVITY_DIAGNOSIS.md), and [project chronology](docs/PROJECT_PROGRESS.md#13-completed-run-and-sensitivity-diagnosis).

## Reproduction setup

The working baseline is adapted from public DGDCN code at commit `96c7bee49c240124735ffc2ab2ad118d27a1ff97`. It uses 12-second clips, 19 electrode nodes from the released implementation, log-amplitude FFT features, and a patient-balanced sampler on the official patient-separated train, development, and evaluation partitions. The patient-equal contribution preference remains in force; no sampler change is part of the current sensitivity analysis. Training-only normalization is computed from the training clips.

This is an adaptation, not a strict replication of the article. The paper describes TUSZ v2.0.1 and 22 derivations; this experiment uses TUSZ v2.0.6 and the 19-node representation in the released code. Changes to preprocessing, data version, sampling, and compute environment must accompany any reported result.

## Data handling

The EEG corpus is restricted research data. This public record contains no recordings, individual annotations, file IDs, folder IDs, credentials, access links, or Kaggle dataset references. Obtain the corpus through its official access process and follow its data-use terms. The original outlier reports were checked against the source data; the [data quality review](docs/DATA_QUALITY_REVIEW.md) records the confirmed technical exclusions and limitations.

## Reporting rule

Only report metrics after they appear in saved aggregate outputs. Identify the split, clip-level scope, checkpoint rule, and threshold-selection rule. The original cutoff maximized development F1 at the development-AUROC-selected checkpoint. Subsequent target-recall cutoffs were selected using development-positive scores only, frozen, and then evaluated once. Development negatives were not scored, so dev specificity and precision are unavailable for those cutoffs. Clip-level false negatives and recall are not missed-event counts or event sensitivity, false alarms per day, latency, or evidence of clinical utility. This is one adapted benchmark run without confidence intervals, not prospective clinical validation.
