# QC200 preprocessing contract

This experiment adapts released DGDCN code to TUSZ v2.0.6. The paper used v2.0.1. A close metric would not establish reproduction of the paper's cohort.

Retain official patient partitions and the independently reviewed filtered markers. Complete, nonoverlapping 12-second clips are positive at at least one second of unioned valid seizure overlap. Short background annotation rows are sentinels and do not limit the usable negative duration. Zero-duration seizure rows are omitted from the derived event view. Raw annotations remain intact.

The channel order is FP1, FP2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, FZ, CZ, PZ. Require one occurrence of each electrode and a common reference suffix. Decode selected channels explicitly and fail on read errors, nonfinite samples or a constant required channel. Do not substitute zeros for failed reads.

For fidelity to the historical adapter, use full-recording `scipy.signal.resample` to `floor(duration_seconds)*200` samples when the original rate differs. This floor behavior is recorded as a limitation; changing it requires a different feature version. Retain calibrated physical signal units returned by the EDF decoder. All selected channels must have the same original sample rate.

Each clip becomes twelve one-second windows. Use `scipy.fftpack.fft` with 200 samples, retain bins 0 through 99 including DC, replace exactly zero magnitudes with 1e-8, and take natural logarithms. Store float32 arrays with shape `(12,19,100)`. No artifact cleaning, amplitude trimming, demographic exclusion or further physiological outlier deletion is added.

Use the QC marker view to remove exactly 156 cached training negatives. Keep all other cached identities unchanged. Historical features can be reused only after source provenance and numerical equivalence checks. The example configuration deliberately leaves provenance approval false. Full-cache shape/type/finite-value validation is required before training. That audit does not establish raw-to-feature equivalence on its own.

Fit fresh per-electrode mean/std on the seeded balanced training pool before augmentation. Dev and eval do not contribute moments or sampling parameters. Patient contributions have equal total expected sampling mass within that pool. This does not imply equal class probability. B1 uses unweighted training BCE; W1 uses `(1-q)/q` computed from the exact selected pool. Both use unweighted dev BCE for early stopping.

Reflection and global log-amplitude scaling remain as in the historical runner. The retained reflection list omits P3/P4; this is preserved for the initial controlled pair and must be explicitly changed in a separate ablation.
