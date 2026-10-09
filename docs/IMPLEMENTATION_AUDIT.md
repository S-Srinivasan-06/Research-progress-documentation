# DGDCN implementation and reproduction audit

Audit date: 2026-10-09. Sources: the full article, authors' released commit `96c7bee49c240124735ffc2ab2ad118d27a1ff97`, the local reproduction source, the embedded Colab setup bundle, and saved aggregate run outputs. This is a static implementation audit, not a new training experiment. No training or scientific configuration was changed during this audit.

## Finding

The recorded run has not met the numerical reproduction target. Its best development AUROC is 0.769421 at epoch 4, among seven saved epochs. A result within one absolute percentage point of the paper's 0.887 would lie between 0.877 and 0.897. These numerical endpoints define a target, not a statistically justified equivalence margin. Our current development result and the publication's result do not share a fully established evaluation protocol. There is no final held-out result from this interrupted run.

The model was largely preserved, but preprocessing and sampling were changed before establishing a baseline faithful to the release. Consequently, the current run is an adaptation. The observed gap cannot yet be attributed to training duration, dataset version, one code defect, or the original publication's validity.

## What the paper specifies

The article uses TUSZ v2.0.1, 200 Hz EEG, non-overlapping 12- or 60-second clips, seizure overlap of at least one second, log-FFT features, and training-set normalization. It describes 22 channels and 100 spectral features. Temporal attention feeds spatial attention; graph convolution uses spatial attention multiplied elementwise with Chebyshev supports, followed by dilated temporal convolution. The optimizer is Adam with learning rate 0.0001, batch size 20, polynomial order 3, and 64 graph and temporal filters. It uses cosine scheduling and early stopping after five validation-loss non-improvements. Augmentation scales amplitude and reflects channels across the midline. [Article, Sections 2 and 3](https://pmc.ncbi.nlm.nih.gov/articles/PMC12383979/)

The reported 12-second AUROC is 0.887. Specificity is inconsistent: 76.6% in Table 1 versus 78.6% in the prose. The exact table split, averaging procedure, run count, classifier/loss, and several implementation choices are insufficiently specified to reconstruct a unique experiment. [Article, Sections 3 and 4](https://pmc.ncbi.nlm.nih.gov/articles/PMC12383979/)

## Architecture actually executed

The embedded training bundle matches the audited model and runner. The active constructor still selects `DGDCN_submodule_1`. The input is `(batch,12,19,100)`, internally rearranged to `(batch,19,100,12)`. Four blocks compute temporal attention, derive spatial attention from the temporally weighted input, and apply spatial-attention-modulated graph convolution to the original block input. Each block then applies the active temporal convolution (kernel 3, dilation 2), a residual connection, and normalization. The final convolution and flattening feed fully connected layers with widths 128, 64, and 2. Taking the maximum of the two final outputs produces the single logit consumed by binary cross-entropy with logits. This unusual head has been retained.

The active block defines other temporal convolutions but calls only its `time_conv2` branch. The released softmax axes are also retained. These details are observations of code execution paths, not claims that they exactly reconstruct the undisclosed experiment behind the publication.

## Differences established from code and artifacts

| Component | Authors' released code or artifacts | Current run | Implication |
|---|---|---|---|
| Network input | 19 electrode nodes | 19 nodes | Matches the released code; the paper describes a different channel representation |
| Classification head | Two outputs, followed by their maximum as a single binary logit | Preserved | The scalar output does not mean the classifier was replaced with a single-output layer |
| Graph construction | Fixed adjacency; sample-specific spatial attention modulates fixed supports | Preserved | The basis is not rebuilt for each EEG clip |
| Graph helper | Model imports a missing `utils_DGDCN`; available `utils.py` uses elementwise multiplication in its polynomial recurrence | Import repaired to available `utils.py`; recurrence retained | The executed recurrence differs from a matrix Chebyshev recurrence; the exact helper used by the authors is unknown |
| Normalization | Supplied active 12-second mean and standard-deviation pickle objects are scalars | Per-electrode mean and standard deviation, shape `(1,19,1)` | A confirmed change to the input transform; upstream fitting population is not established |
| Training sampling | Equal-count negative undersampling, then ordinary clip sampling | Balanced clip pool followed by patient-weighted sampling with replacement | Changes which clips the optimizer sees and may change the sampled class ratio |
| Corpus | Release associated with TUSZ v2.0.1 | TUSZ v2.0.6 | Dataset identity differs |
| Evaluation stage | Publication reports final average metrics | Seven development epochs; run interrupted | Cannot treat the current metric as final replication performance |

Source pointers: authors' [model](https://github.com/Open-EXG/DGDCN-EEG-Seizure-Detection/blob/96c7bee49c240124735ffc2ab2ad118d27a1ff97/model/DGDCN/model/DGDCN_r.py), [graph utilities](https://github.com/Open-EXG/DGDCN-EEG-Seizure-Detection/blob/96c7bee49c240124735ffc2ab2ad118d27a1ff97/model/DGDCN/lib/utils.py), and [detection loader](https://github.com/Open-EXG/DGDCN-EEG-Seizure-Detection/blob/96c7bee49c240124735ffc2ab2ad118d27a1ff97/data/dataloader_detection.py). Current code evidence: private archived `train_kaggle200.py`, functions `scaler_from_balanced_training`, `balanced_train_rows`, and `main`; model supplied by `build_colab_training_cells.py`. Aggregate observations are in [development_metrics.jsonl](../results/development_metrics.jsonl).

## Clip population differs materially

Counts below were obtained from the authors' active `file_markers_bugua` files and our saved marker-consolidation report. They are list-entry counts, not verified unique clips or patient counts.

| Partition | Released negative entries | Released positive entries | Current negative entries | Current positive entries |
|---|---:|---:|---:|---:|
| Training before undersampling | 36,491 | 14,606 | 250,798 | 11,223 |
| Development | 14,890 | 4,392 | 71,077 | 2,310 |
| Evaluation | 10,114 | 2,624 | 36,115 | 2,269 |

These are substantially different evaluation populations. Class prevalence alone does not determine AUROC, but selection of different negative and positive clips can change conditional score distributions and difficulty. The counts do not establish the cause of the performance gap. The repository's second marker collection has different counts again. The collection that generated the published table is not conclusively established.

## Labels, channels, and provenance

The current producer generates fresh markers from sibling binary annotation CSV files and applies a minimum of one second of unioned seizure overlap. This follows the article's stated rule. The released helper computes an any-overlap label, but ordinary upstream training uses precomputed marker labels; the helper alone does not establish the rule used to generate the supplied marker lists. That question remains unresolved.

The current code preserves source `train/dev/eval` paths, explicitly names the evaluation marker files `test`, and checks patient separation. This audit found an explicit alias rather than evidence of an accidental dev/eval swap. No new source-corpus scan was performed.

The FFT path uses one-second steps, retains the first 100 bins of the 200-sample FFT, and takes log amplitudes. Both implementations select the same 19 electrode names without rereferencing. Current channel eligibility additionally checks uniqueness and a common reference suffix. These rules can change the eligible population. The historical embedded preprocessing bundle differs from the current working copy, and no numerical comparison against the saved feature cache was performed. Static inspection therefore establishes the intended construction, not end-to-end numerical equivalence of all stored clips.

## Why the graph recurrence matters

The available helper computes `2 * L * previous - earlier` with NumPy arrays, so multiplication is elementwise. For the third support this gives `2 * (L elementwise-squared) - I`. A matrix Chebyshev recurrence instead uses `2 * (L @ L) - I`, which includes paths through intermediate nodes. The implemented operation can therefore change graph propagation. Its impact on seizure AUROC has not been measured in this project. Because the upstream import points to an absent helper, replacing this operation should be a separately documented equation-consistent experiment, rather than silently described as recovery of the authors' executed code.

## Experiments needed before interpreting the gap

1. Freeze the existing run as the patient-balanced, per-electrode-normalized adaptation. Preserve its checkpoints and history.
2. Establish a release-oriented baseline on the available corpus: ordinary shuffled sampling from the balanced clip pool, scalar normalization fitted only on the declared training population, preserved head and graph behavior, and a fixed seed and software environment. The original scalar values should not be copied onto new data. The authors' exact normalization fitting population remains an open question.
3. Compare normalization and patient sampling separately on the same clips and patient partitions. Do not change multiple factors and then assign the improvement to one of them.
4. Compare the retained graph recurrence with matrix multiplication in a separately named experiment. Preserve the same initialization and training conditions where possible. Other equation/code discrepancies require their own explicit decisions.
5. Select checkpoints and thresholds using development data, then assess the selected model once on held-out data. Report AUROC, average precision, sensitivity, specificity, F1, class counts, patient counts, and uncertainty. Multiple predetermined seeds are needed to assess stability.

Numerical agreement on the newer corpus would remain evidence about that adaptation. It would not establish reconstruction of the original dataset experiment. More epochs alone have not been shown to close the gap.

## Questions for the authors

- Which model, graph helper, marker collection, and configuration produced the published table?
- Was the actual input 19 electrode signals or 22 bipolar derivations?
- How were the supplied marker lists and scalar normalization files generated, and which training population was used?
- Which partition, threshold rule, seeds, and averaging procedure produced the results?
- Can they supply the exact experiment snapshot or checkpoint and resolve the reported specificity discrepancy?

No message has been sent to the authors. These are unresolved reproducibility questions, not evidence of misconduct.

## Confidence

High confidence that the audited model core was preserved and that normalization and training sampling differ from the release. High confidence that the saved marker populations differ. Low confidence about how much any one difference explains the measured AUROC gap: no controlled ablation has been run. The complete article was reviewed, including its equations, experimental setup, result tables, limitations, and data statement; no supplementary implementation specification was listed.
