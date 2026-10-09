# Project progress and decisions

Last updated: 2026-10-09 13:03:46 UTC. Times below are UTC where a runtime log supplied a timestamp. Earlier project discussions did not have a reliable timestamp, so those entries are ordered by phase rather than assigned a date.

## Objective

Keep a clear, auditable record while attempting to reproduce the 12-second DGDCN EEG seizure detector. The immediate experiment uses TUSZ v2.0.6 at 200 Hz and the model implementation released by the paper's authors. The goal is to determine whether the published benchmark experiment can be run from the available code and data, document mismatches, and report results only if training and evaluation finish.

The team also discussed processing a separate 256 Hz version because many recordings have that native sampling rate. That remains a possible later experiment. No 256 Hz feature set or model run is included here.

## Decisions

- Use the existing TUSZ v2.0.6 200 Hz feature and marker archives; do not resample the EDF corpus again for this run.
- Retain the official patient-separated train, development, and evaluation partitions.
- Balance the training clip pool by class and use the configured patient-balanced sampler. Record this difference when comparing with the paper.
- Keep the outlier scan aside. Do not exclude physiologically unusual seizures or recordings based on its automatic suggestions in this run.
- Save run logs, configuration, normalization, checkpoints, and final predictions to the authorized Drive workspace as they are produced.
- Do not publish the restricted corpus, its annotations, file or folder identifiers, or access credentials in this repository.

## Chronology

### 1. Literature and paper review

The research started as a critical review of automated EEG seizure detection. The DGDCN article was used as a starting point for paper appraisal and code inspection, not as the entire scope. The review distinguishes selected-clip classification from continuous seizure-event detection and clinical usefulness.

### 2. DGDCN release inspection

The public release was found to contain execution and configuration problems, hard-coded machine paths, and inconsistencies between its 19-node implementation and the paper's description of 22 derivations. The 60-second path also was not cleanly parameterized in the released entry point. The reproduction work therefore targets the 12-second setup and treats any result as an adaptation to the released code and current dataset version.

### 3. Data and preprocessing

The authorized TUSZ v2.0.6 corpus was prepared into 40 feature archive batches and corresponding marker archives at 200 Hz. The existing archives contain the transformed model inputs, so recovery copies and extracts these archives; it does not repeat EDF resampling. An outlier report was produced and then kept aside. Its proposed exclusions were not applied.

### 4. First Kaggle attempt

On 2026-10-09, a private Kaggle notebook was configured with GPU access. Its synthetic model forward and backward check passed. Staging stopped after three feature batches because a shared-file download request hit Google's download quota. The run did not start normalization or training.

The archives were subsequently submitted to a private Kaggle dataset. Kaggle accepted the upload requests, but the account's dataset metadata and file-list APIs returned HTTP 403, so privacy, file completeness, and readiness could not be verified. Training was not launched on Kaggle.

### 5. Colab GPU recovery attempt

Colab allocated a Tesla T4. After Google Drive authorization was restored, a 1 KiB range request to a previously quota-blocked feature archive returned HTTP 206. A CUDA forward and backward check passed, and a checkpoint-sync worker created a Drive run folder and uploaded its initial state file.

The recovery controller downloaded and verified batches sequentially from Drive, staged them into the existing 200 Hz feature directory, and removed only each batch's downloaded temporary archives after its staging checkpoint was verified. At **2026-10-09 12:22:16 UTC**, 36 of 40 batches were staged. The request for the next archive ended with an `IncompleteRead` while receiving a ranged response. The controller stopped safely. No training normalization or model training had begun. The verified staging checkpoints for the completed batches remain in Colab, and a copy of the failure state and controller log was saved to Drive.

### 6. Restart plan after interrupted transfer

The next recovery should keep the existing staging root so batches with valid checkpoints can be skipped. The remaining archive downloads need bounded range requests that resume from the byte offset already written after a network interruption. Once all 40 batches are verified, consolidate the marker lists, compute normalization from the training partition only, and start the CUDA training process. Drive synchronization should retain the run state, configuration, checkpoints, and final outputs across a Colab reset.

### 7. Colab recovery retry

At **2026-10-09 12:33 UTC**, a second Colab controller was started on the Tesla T4. It created a fresh Drive run folder and passed a new state-file write check. The controller salvaged **402,653,184 bytes** of the interrupted transfer for batch 37, then rechecked and skipped previously verified staging checkpoints instead of repeating those extractions. The download code now requests bounded byte ranges and resumes at the length already present if a connection cuts off.

### 8. Feature restoration, normalization, and initial training

By the latest verified status at **2026-10-09 13:03:46 UTC**, restoration and stage verification had completed for **40/40** batches. The feature store contained **374,112 feature files (34,880,512,545 bytes)** and **373,792 marker rows**. Training-only normalization completed, and its normalization file and run configuration were saved to Drive. The training process was active on the Tesla T4; Drive synchronization had no reported error.

Two development epochs had completed. Epoch 1: loss 0.3664, accuracy 0.8930, F1 0.1625, precision 0.1078, recall 0.3299, specificity 0.9113, AUROC 0.7477, average precision 0.0774, threshold 0.5005. Epoch 2: loss 0.2625, accuracy 0.9034, F1 0.1720, precision 0.1178, recall 0.3186, specificity 0.9224, AUROC 0.7604, average precision 0.0963, threshold 0.4524. Epoch 2 was the best development-AUROC checkpoint at that snapshot. The development set had 73,387 clips: 2,310 positive and 71,077 negative. An all-negative classifier would therefore reach **96.85% accuracy**, illustrating why the model's 90.34% accuracy is not evidence of useful discrimination by itself.

These are preliminary **development clip-level** results from two epochs and one training run. No confidence interval or across-seed variability is available. The held-out evaluation partition has not been run, and no event-level sensitivity, false alarms per 24 hours, latency, or clinical utility has been measured. Checkpoints (`last.pt`, `best.pt`), metrics, and controller logs were syncing to Drive when this status was captured.

## Experiment specification

| Item | Current choice |
|---|---|
| Task | Binary EEG seizure detection on 12-second clips |
| Corpus | TUSZ v2.0.6, restricted access |
| Feature sampling rate | 200 Hz |
| Input representation | Log-amplitude FFT features, 100 bins, 19 nodes |
| Partitions | Official patient-separated train, development, and evaluation |
| Training sampling | Balanced clip pool with patient-balanced sampling weights |
| Accelerator checked | NVIDIA Tesla T4 in Colab; synthetic forward/backward passed |
| Normalization | Computed from training clips only; saved at latest snapshot |
| Current epoch count | 2 completed at the latest verified snapshot; training remained active |
| Latest development result | Epoch 2 AUROC 0.7604; F1 0.1720; precision 0.1178; recall 0.3186; specificity 0.9224; accuracy 0.9034 |
| Held-out evaluation | Not run |

## Reproduction limits

This setup does not match the paper exactly: the paper used TUSZ v2.0.1 and describes 22 derivations, while the released model code uses 19 nodes and the available corpus is v2.0.6. Patient-balanced sampling also changes the training distribution. Preserve these distinctions in any later comparison.

The released article's segment-level AUROC or accuracy cannot be treated as continuous event sensitivity, false alarms per 24 hours, or detection latency. Those measures require a continuous scoring pipeline and are not generated by the present clip-classification run alone. The two development epochs are an early run snapshot, not a final or held-out result; training may have advanced since the timestamp above. Do not draw clinical conclusions from these results.

## Update procedure

When the run changes, add a dated entry above or below with the observed timestamp, completed batches, active phase, checkpoint-sync status, and any error. State explicitly whether normalization, training, and evaluation have started. Link to output filenames only when they contain no corpus identifiers or access details. Do not add credentials, restricted data locators, or EEG files.
