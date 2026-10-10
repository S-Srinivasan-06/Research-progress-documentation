# Data quality review for the 200 Hz experiment

Date: 2026-10-10.

The original outlier reports were reviewed against their detector code and the source corpus before applying exclusions. The resulting filter defines a new input view for the existing 19-electrode, 12-second model. It preserves the source recordings, annotations, official partitions, and historical baseline results.

## What the original flags establish

The first detector's flatness check read a short block of raw EDF bytes. It did not check the complete recording or respect individual signal blocks. Its 124 reported header anomalies were the same 124 flatness warnings, rather than 124 independently verified corrupt headers. The later detector inspected several sampled data records. Both reports provide candidates for review; neither establishes whole-record signal failure by itself.

The new audit checks EDF header fields and payload length across the corpus. For the reported flat or dead-channel candidates, it checks every digital sample of the required EEG channels. Exactly constant required channels are excluded from this model's input view. This waveform check is limited to reported candidates; it is not an exhaustive waveform audit of every recording.

Missing required electrodes are model eligibility failures. Recordings shorter than 12 seconds cannot supply a complete clip. Neither condition establishes that the recording is clinically useless. Large amplitudes, clipping detected by sparse sampling, startup transients, long seizures, unusual ages, and high channel counts remain review flags rather than automatic removals.

## Annotation convention

The official annotation guide explains that a short background entry marks a seizure-free recording so that its annotation file is not empty. Requiring explicit background coverage across every negative clip would therefore misinterpret the corpus. The guide was retrieved and checked during this review. [NEDC annotation guidelines, Annotation Labels section](https://isip.piconepress.com/publications/reports/2026/eeg/annotations/guidelines_v04.docx).

The compatibility view preserves the existing labeling rule: a complete clip is positive when the union of valid seizure intervals overlaps it by at least one second. Other complete clips are negative under the corpus annotation convention. Zero-duration rows contribute no seizure time and are omitted from the derived annotation overlay. The source annotation files are preserved.

A byte-identical duplicate recording can be removed from the input view while retaining its equivalent copy. The pair was compared directly without computing content hashes. A difference in the length of its background annotation is not, by itself, evidence that the remaining EEG is unannotated.

## Verified filtered view

The audit scanned 8,140 source EDFs. It excluded 630 recordings from the model view: 625 lacked one of the 19 required electrodes, four had a required channel that was constant across every sample, and one was a byte-identical duplicate whose equivalent recording was retained. The 625 montage-ineligible files had already been omitted by the previous preprocessing run.

We recovered and parsed all 240 saved marker tables, containing 373,792 cached clips, then compared their exact recording, clip, and label entries with the source QC manifest. The new marker set contains 373,636 clips. It removes exactly 156 training negatives, which match the four constant-channel recordings and the duplicate. It adds no rows and changes no development or evaluation rows. The six loader-ready marker tables were bundled and uploaded to the existing collaborator training folder in Drive. No sharing settings changed. No EDFs or feature archives were modified.

Across the original source recordings, the number of complete 12-second windows drops from 440,601 to 373,636 in this model-eligible view. Positive windows drop from 16,154 to 15,802, and negative windows from 424,447 to 357,834. These source-manifest counts describe eligible windows; they are not the cached marker counts used by the prior training run. The filtered evaluation marker set is unchanged, so previous evaluation metrics remain numerically tied to the same evaluation clips.

All 134 distinct reported flat/dead candidates were matched and checked across their full required-channel digital samples. None were unreadable or ambiguous. The saved annotation overlay omits one zero-duration event row while retaining the other valid seizure intervals. It flags 336,620 retained negative clips for annotation-coverage review; these flags do not remove clips or change labels.

The local QC workspace holds the reason-coded recording and clip manifests and exact marker delta. It contains individual dataset paths and is not part of the public progress repository.

The filtered 1:1 training pool contains 11,223 positive and 11,223 negative clips across 575 patients. Under the existing patient-balanced replacement sampler, the expected positive draw probability is 0.167957. The corresponding class-weight value is 4.9539, but this is only a candidate for a controlled loss-weight ablation. The prior run used unweighted binary cross-entropy; weighting is not silently added by this data cleanup. The baseline sampler probability was 0.167791, reproduced from the original markers.

Public aggregate counts are in [data_quality_review.json](../results/data_quality_review.json). The detailed filenames and reason-coded rows remain in the local restricted-data workspace.

Before another training run, recompute normalization from the retained training features and calculate the effective seizure fraction after patient balancing. If using a positive loss weight, derive it from the actual training sampling distribution and compare it with unweighted loss. Neither anomaly removal nor class weighting guarantees improved AUROC.

The saved baseline scores remain results from the original input view. Any retrained result must state its new cohort, exclusion policy, training sampler, normalization, and threshold-selection procedure. No new training result follows from this audit alone.

EDF layout and calibration were checked against the [official EDF specification](https://www.edfplus.info/specs/edf.html).
