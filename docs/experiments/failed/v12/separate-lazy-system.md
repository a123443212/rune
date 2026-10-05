# Failed: separate lazy-eval system beside adaptive

Attempt: design an independent lazy evaluation framework next to the v0.4 cheap-plus-refine machinery.

Result: dropped during the audit. L1 reuses threshold routing directly and L2 is a thin margin check on top. A second system would double the semantics to test for no measured benefit.
