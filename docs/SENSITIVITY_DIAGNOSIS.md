# Sensitivity diagnosis and operating-point plan

Status verified 2026-10-10. This report describes the completed 200 Hz, 12-second clip-classification run and a subsequent threshold analysis. The model weights and score ranking stayed fixed while the decision threshold changed. This distinction follows the [scikit-learn threshold-selection documentation](https://scikit-learn.org/1.5/modules/classification_threshold.html).

## Completed held-out clip result

| Metric | Eval result |
|---|---:|
| AUROC | 0.840075 |
| Average precision | 0.295845 |
| Accuracy | 0.928199 |
| F1 | 0.320178 |
| Precision | 0.363585 |
| Recall (sensitivity) | 0.286029 |
| Specificity | 0.968545 |
| Threshold | 0.701882 |

At this cutoff the 38,384 eval clips comprise 649 true-positive clips, 1,620 false-negative clips, 1,136 false-positive clips, and 34,979 true-negative clips. The false-negative count refers to positive 12-second clips, not whole seizures. The [sanitized JSON record](../results/final_eval_metrics.json) contains the aggregate values.

## Threshold selection and sensitivity priority

The best checkpoint was selected by development AUROC at epoch 4. The threshold 0.701882 was selected to maximize development F1 at that checkpoint, then applied once to eval. F1 balances precision and recall; this cutoff did not encode the user's preference to prioritize sensitivity. A CPU-only post-run analysis then selected additional thresholds from development positive scores only and evaluated them once on the held-out clips. Evaluation scores were not used to select thresholds. Development negative scores were unavailable, so development specificity and precision cannot be reported for the target-recall thresholds.

The primary prespecified operating point targets 0.95 development recall. The secondary operating points target 0.85, 0.90, and 0.99. For each target, the selector chose the highest observed development-positive score threshold that reaches or exceeds the target recall.

| Policy | Threshold | Dev positive-only recall | Eval recall | Eval specificity (FPR) | Eval precision | Eval FN / FP clips |
|---|---:|---:|---:|---:|---:|---:|
| Existing maximum-dev-F1 baseline | 0.701882 | 24.24% | 28.60% | 96.85% (3.15%) | 36.36% | 1,620 / 1,136 |
| Fixed 0.5 reference | 0.500000 | 50.09% | 61.97% | 86.68% (13.32%) | 22.62% | 863 / 4,811 |
| Dev recall target 0.85 | 0.217881 | 85.02% | 90.13% | 55.79% (44.21%) | 11.35% | 224 / 15,966 |
| Dev recall target 0.90 | 0.153576 | 90.00% | 94.14% | 42.78% (57.22%) | 9.37% | 133 / 20,664 |
| Dev recall target 0.95, primary | 0.075218 | 95.02% | 97.58% | 22.43% (77.57%) | 7.32% | 55 / 28,014 |
| Dev recall target 0.99 | 0.020343 | 99.00% | 99.47% | 4.73% (95.27%) | 6.16% | 12 / 34,408 |

At the primary dev-0.95 operating point, eval recall is 97.58%, but 28,014 of 36,115 negative clips are false positives, for a 77.57% false-positive rate and 7.32% precision. This is a measured clip-level threshold tradeoff, not a model improvement: AUROC (0.840075) and average precision (0.295845) are unchanged at every threshold. The [sanitized operating-point report](../results/sensitivity_threshold_report.json) contains the full aggregate values. The available public development history contains complete rows through epoch 7. The cached completion summary provides only partial epoch 8 fields, so no epoch 8 row was added to the development metrics file.

## Training factors and interpretation limits

The training objective used unweighted binary cross-entropy on a class-balanced pool, followed by patient-equal sampling with replacement. The measured expected positive draw fraction under this sampler is 0.167791 for the 22,446-clip balanced pool across 574 patients. Patient-equal contribution remains the current user preference and run configuration; changing the sampler is not proposed. Because BCE was unweighted, a possible future controlled training ablation is to set positive loss weight near 4.96, the expected negative-to-positive draw ratio, while retaining the sampler. This has not been trained or evaluated, and the sampler's class mix is not established as the cause of score quality. Normalization used per-channel moments from the balanced training pool. The run completed eight epochs and stopped by its configured early-stopping rule, with epoch 4 selected by dev AUROC. Limited training duration is a hypothesis, not an established cause.

The graph audit found that the available helper uses elementwise multiplication in its second Chebyshev support. The upstream code imports a helper that is absent from the inspected release tree, so the authors' exact helper behavior is uncertain. This is a code-fidelity risk with unmeasured impact, not an established causal defect for this result.

## Threshold policy helper

The first-party helper in `scripts/sensitivity_policy.py` supports the historical F1 rule and a target-sensitivity rule with `p >= threshold` semantics. Seven standalone unit checks passed. The F1 test compares against a small reference precision-recall curve stub because scikit-learn is unavailable in the local environment. These checks cover the helper only; no new model training was performed during this diagnosis. The [primary operating-point configuration](../results/primary_operating_point.json) records the exact threshold and input specification. Diagnostic artifacts were also saved to Drive.

## Event-level metrics remain unavailable

All reported metrics here count 12-second clips. Current labels mark a clip positive when annotated seizure overlap reaches at least one second. A seizure may cover several clips, several events may fall in one clip, and clip false negatives cannot be interpreted as missed seizures.

Event sensitivity, missed event counts, false alarms per hour, and latency require matching each clip score to its exact recording interval and the source seizure intervals, then defining event matching tolerance and alarm-merging rules. [SzCORE (DOI: 10.1111/epi.18113)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12489712/) explains why event scoring is needed to count detected seizures and false alarms. Those calculations have not been done here. This clip evaluation does not establish event-level or clinical performance.
