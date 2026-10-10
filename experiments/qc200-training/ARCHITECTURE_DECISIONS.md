# Architecture decisions

The active `repo/` snapshot comes from the locally archived reproduction source. It retains `make_model_2` selecting `DGDCN_submodule_1`, four blocks, 19 electrode nodes, 100 input FFT bins, K=3 graph supports, and 64 graph/temporal filters. Input is `(batch,12,19,100)`.

Retain temporal attention followed by spatial attention, attention modulation of fixed Chebyshev supports, the active temporal kernel of size 3 with dilation 2, residual connections and normalization. Retain the two-output linear head reduced by maximum to a binary logit. Retain the available helper's elementwise Chebyshev recurrence. These choices describe the released adaptation, not confirmed publication implementation details.

A matrix recurrence, conventional single-logit head, different attention normalization, full scalp reflection, alternative channel representation, or scalar feature normalization must receive its own experiment identity. B1 and W1 differ only in training positive-class loss weight. Neither resumes H0.

Checkpoint selection uses dev AUROC. Early stopping uses unweighted dev loss with patience five. Final evaluation is an explicit separate entry point with a frozen selection record. Historical eval results were already inspected, so future eval results are exploratory benchmark measurements.
