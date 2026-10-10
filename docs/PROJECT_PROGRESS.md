# Project progress and decisions

Last updated: 2026-10-10 08:16 UTC. Times below are UTC where a runtime log supplied a timestamp. Earlier project discussions did not have a reliable timestamp, so those entries are ordered by phase rather than assigned a date.

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

These were preliminary **development clip-level** results from the first two epochs of one training run. No confidence interval or across-seed variability is available. The held-out evaluation partition has not been run, and no event-level sensitivity, false alarms per 24 hours, latency, or clinical utility has been measured. Checkpoints (`last.pt`, `best.pt`), metrics, and controller logs were syncing to Drive when this status was captured.

### 9. Training update

At **2026-10-09 13:13:42 UTC**, the training controller and Drive checkpoint synchronizer were both alive, with no Drive sync error. Four epochs had completed. Epoch 3 reached development AUROC **0.7672** and average precision **0.1135**. Epoch 4 reached the best development AUROC so far, **0.7694**, with loss **0.3728**, accuracy **0.9340**, F1 **0.1879**, precision **0.1534**, recall **0.2424**, specificity **0.9565**, average precision **0.1238**, and development-selected threshold **0.7019**. The AUROC increased from epoch 2's 0.7604, but remains **0.1176 below** the paper's reported 12-second value of 0.887. These are development metrics and do not establish held-out performance. Training remained active; final evaluation had not started.

### 10. Runtime replacement and stale monitor output

A fresh diagnostic at **2026-10-09 14:23:42 UTC** found a runtime with about 267 seconds of uptime, no `nvidia-smi`, only `.config` and `sample_data` in `/content`, and no training output directory or training processes. The earlier monitor returned saved notebook output; its process-alive flags did not describe this new runtime. The earlier suggestion of an epoch-4 stall was therefore unsupported.

Direct Drive reads showed **seven completed epochs**. The saved metrics file was modified at 13:33:59 UTC, `last.pt` at 13:33:57 UTC, and the controller log at 13:34:02 UTC. Epochs 5, 6, and 7 had development AUROC 0.7591, 0.7671, and 0.7675, respectively. Epoch 4 remains best by development AUROC (0.7694). The `best.pt` file remains present, last modified at 13:13:39 UTC. The checkpoint files' internal contents have not yet been loaded after this interruption. The saved run folder contains no final evaluation artifact.

The available evidence establishes runtime replacement and stale monitoring output. It does not identify whether the previous session ended because of a user action, quota, disconnection policy, or another cause. No new training exception appears after the final saved epoch. The DataLoader worker warning is not sufficient evidence of a deadlock.

Recovery must explicitly load the saved training checkpoint; the controller's existing launch command does not automatically pass the trainer's supported `--resume` argument. Preserve both the latest checkpoint and the best development checkpoint. Restore the existing features and saved training normalization after GPU allocation and Drive authorization. Future monitoring must include a fresh execution timestamp and invocation marker, and reject old notebook outputs as live evidence.

### 11. Full-paper and implementation fidelity review

The paper, released source, embedded training bundle, and preprocessing code were reviewed after the interrupted run. The audit confirmed preservation of the core network and two-output maximum head. It found material differences in scalar versus per-electrode normalization, patient-weighted training sampling, and clip populations, plus unresolved graph-helper and paper/code ambiguities. No single cause of the AUROC gap has been demonstrated. The next scientific comparison should establish a release-oriented baseline and vary these factors separately. The available dataset remains a different release from the paper's dataset. [Full audit](IMPLEMENTATION_AUDIT.md). No new model training was launched during this review.

### 12. CPU checkpoint recovery

At **2026-10-10 05:33:55 UTC**, the saved checkpoint was reopened and verified with seven completed epochs, epoch 4 still the best checkpoint, and seven metric rows available. The CPU continuation controller and checkpoint synchronizer started in a new output run folder. Continuation is set to begin at epoch 8. CPU continuation is an authorized option when GPU runtime is unavailable. The checkpoint preserves training normalization, optimizer, scheduler, Python/NumPy/CPU Torch RNG, and early-stopping state. CUDA RNG does not drive CPU dropout, and hardware continuation is not bitwise identical.

At the latest fresh check, **2026-10-10 05:37:14 UTC**, the controller and synchronizer were alive with no reported error. Batches **1-3 of 40** were complete, and batch 4 was transferring. The controller is restaging the **40 saved feature and marker archive pairs**, totaling **30,071,690,466 compressed bytes**. This is restoration of the existing 200 Hz transformed features, not EDF resampling. Training had not started, and evaluation had not started. The runtime cause remains unknown. The run-folder and corpus identifiers are intentionally omitted.

### 13. Completed run and sensitivity diagnosis

The latest verification at **2026-10-10 08:16:21 UTC** records eight completed epochs, early stopping, best checkpoint epoch 4, and a completed one-time held-out evaluation. At the stored threshold of **0.701882**, eval AUROC was **0.840075**, average precision **0.295845**, accuracy **0.928199**, F1 **0.320178**, precision **0.363585**, clip recall **0.286029**, and specificity **0.968545**. The evaluation contained 38,384 clips: 2,269 positive and 36,115 negative. The corresponding counts were TP 649, FN 1,620, FP 1,136, and TN 34,979. These are clip counts, not seizure-event counts.

The checkpoint was selected by development AUROC; its original threshold was selected by maximum development F1. A CPU-only, threshold-only analysis later selected cutoffs from development positive scores and then evaluated them once on eval. At the primary prespecified dev-recall-0.95 cutoff of **0.075218**, eval clip recall was **0.975760**, with **0.775689** false-positive rate and **0.073243** precision: 28,014 false-positive clips out of 36,115 eval negatives. This changes the operating point only; AUROC and average precision remain unchanged, and no model improvement is claimed. All target points and the 0.5 reference are in the [sensitivity diagnosis](SENSITIVITY_DIAGNOSIS.md) and [sanitized threshold report](../results/sensitivity_threshold_report.json); the [primary operating point record](../results/primary_operating_point.json) gives its selection rule and counts. Diagnostic score inputs and run artifacts are saved in the authorized Drive workspace; this repository contains sanitized aggregates only.

The patient-equal sampler remains unchanged. Its expected positive draw fraction was calculated as **0.167791** for 22,446 balanced-pool clips across 574 patients. The expected draw share differs from the pool's 50% positive fraction; the effect of this class mix on score quality is untested. A possible controlled loss-weight ablation can preserve patient-equal contribution and use an expected positive weight near 4.96; no such retraining has been run.

The cached completion summary has only partial epoch 8 development fields, while the public development history contains complete rows through epoch 7. No epoch 8 row was added to `development_metrics.jsonl` because its complete original row was not available locally.

## Experiment specification

| Item | Current choice |
|---|---|
| Task | Binary EEG seizure detection on 12-second clips |
| Corpus | TUSZ v2.0.6, restricted access |
| Feature sampling rate | 200 Hz |
| Input representation | Log-amplitude FFT features, 100 bins, 19 nodes |
| Partitions | Official patient-separated train, development, and evaluation |
| Training sampling | Balanced pool with patient-equal replacement sampling; expected positive draw fraction 0.167791 across 574 patients; preference retained |
| Accelerator checked | NVIDIA Tesla T4 in Colab; synthetic forward/backward passed; CPU continuation authorized |
| Normalization | Computed from training clips only; saved at latest snapshot |
| Completed training | 8 epochs; stopped by configured early stopping |
| Best development checkpoint | Epoch 4; AUROC 0.769421 |
| Held-out evaluation | Complete once; AUROC 0.840075, clip recall 0.286029, specificity 0.968545 |
| Threshold rule | Checkpoint selected by development AUROC; threshold 0.701882 selected by maximum development F1 |
| Sensitivity-first operating points | Dev target recalls 0.85, 0.90, 0.95, and 0.99 measured once on eval; see sensitivity report |
| Event-level metrics | Not computed; current result is 12-second clip classification |

## Reproduction limits

This setup does not match the paper exactly: the paper used TUSZ v2.0.1 and describes 22 derivations, while the released model code uses 19 nodes and the available corpus is v2.0.6. Patient-balanced sampling also changes the training distribution. Preserve these distinctions in any later comparison.

The released article's segment-level AUROC or accuracy cannot be treated as continuous event sensitivity, false alarms per 24 hours, or detection latency. Those measures require original event intervals and a defined continuous scoring and event-matching pipeline; they are not generated by the present clip-classification run alone. The completed evaluation is one held-out clip-level assessment of an adapted setup, with no confidence interval or across-seed estimate. Do not draw clinical conclusions from these results.

## Update procedure

When the run changes, add a dated entry above or below with the observed timestamp, completed phase, and any error. Report only sanitized aggregate metrics, with split, clip-level scope, checkpoint rule, and threshold-selection rule. Do not publish prediction rows, credentials, restricted data locators, participant identifiers, or EEG files.
