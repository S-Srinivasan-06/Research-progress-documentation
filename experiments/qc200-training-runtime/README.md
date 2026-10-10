# QC200 runtime helpers

These helpers sit outside the frozen training source, so runtime changes do not
change the training implementation's source identity.

## Restore the model reference

`restore_reference.py --destination PATH` fetches only the files used by the
trainer from Open-EXG/DGDCN-EEG-Seizure-Detection at commit
`96c7bee49c240124735ffc2ab2ad118d27a1ff97`. It applies the local changes for
the available utility import, configurable flatten dimensions, removal of
shape debug prints, and CPU/current-device selection. Existing differing files
cause an error rather than being replaced. The inspected upstream snapshot has
no declared license; the helper downloads from the public source at runtime and
does not grant permission to redistribute upstream files.

## CPU continuation

`cpu_fallback_watch.py` only considers the active B1 training phase after a
nonzero training-child exit with an explicit CUDA device/driver failure in the
log and a matching pipeline status. It also requires no live original trainer,
an existing nonempty `last.pt`, an eligible persistence guard, a verified
Drive run folder, and enough time before the supplied runtime deadline.

The helper resumes that run on CPU and mirrors changed files to the existing
Drive folder with byte read-back checks. It may replay a partial epoch. It does
not recover validation/code/data errors, start a new run, start beside a live
trainer, or reconnect after the Colab runtime terminates. CPU speed and Colab
limits still apply. Full runtime termination requires a separate reconnect,
Drive authorization, and file-restore step.

## Sequential W1 experiment

`queue_w1.py --b1-config PATH --w1-run-id ID --deadline-utc TIMESTAMP --source-dir PATH`
waits for completed B1 train/dev results and stopped original processes. It
revalidates the shared feature view, copies validation metadata, and starts a
fresh W1 model in a separate output and Drive folder. It retains the same seed,
architecture and sampling; the positive BCE weight changes. It requires 30
minutes for training plus 15 minutes for final synchronization by default.
CPU is selected when CUDA is unusable. Failed W1 training is not retried
automatically. The queue is session-bound and must be relaunched after a full
runtime reset. It does not access final evaluation scores.
