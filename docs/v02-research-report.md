# RUNE v0.2 Research Report

## 1. v0.1 bottleneck

Inference: the 256->128 head and QKV projections dominate; the attention
core (~2.5 us) and GAB (~0.2 us) are negligible. Data: no teacher labels.
Loss: ranking used unprovenanced pseudo-siblings. See `v01-audit.md`.

## 2. Relational mixer representation

Implemented with configurable gate and residual alpha. Parity-verified
against C++, trains end-to-end. Whether it improves representation is
unanswered: no 25M+ comparison exists. Hypothesis intact, evidence pending.

## 3. Dynamic GAB overhead

Under suspicion. Op count says +~1k MACs; the noisy machine says more.
Verdict rule is fixed in advance: dies on NPS regression without metric
gain at 100M. Not yet runnable for lack of data.

## 4. Token counts 8/6/10

Layouts implemented with shared per-group tables (6 merges, 10 splits).
Cost ordering confirmed (6 < 8 < 10). Quality ordering unknown.

## 5. Dimensions 24/32/40

Same status: cost ordering confirmed, capacity trade-off unmeasured.
int8 compatibility preserved in all variants.

## 6. Ranking loss

L0/L1 driver built on true sibling pairs with teacher provenance; L1 runs
only on promoted candidates. Untested beyond a 128-position smoke.

## 7. Disagreement sampling

S1 pipeline built (minority ratio in config, composition before/after,
labeling-cost warning). No teacher scores exist, so no saving claim exists.

## 8. INT8 effect

On random weights: no sign/WDL flips, diffs ~1e-6 (int8) / ~1e-8 (int16),
size roughly halved. Effect on trained strength: unmeasured.

## 9. Candidate for v0.3

None selected. Selection needs 100M reports that do not exist yet.

## 10. Refuted hypotheses

- "Attention core is the inference bottleneck" — refuted by measurement
  (head and QKV dominate).
- "Small pools always split cleanly" — refuted; fixed with fallback.
- Everything else: insufficient evidence, kept as open questions, not as
  claims. No statement in this report asserts RUNE beats any engine or
  network; no controlled engine match has been run.
