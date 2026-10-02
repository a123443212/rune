# RUNE v0.2 Experiment Plan

## Research questions

- RQ1 (tokens): do 6/10 tokens or 24/40 dims beat 8x32 on metrics per cost?
  Configs: `configs/v02/tokens_{6,8,10}.yaml`, `dims_{24,40}.yaml`.
- RQ2 (mixer): does the gated relational mixer (A2) beat v0.1 attention (A1)
  and MLP (A0) at matched cost? Does dynamic bias (A3) add anything over
  static (A2)? Config: `configs/v02/screen_arch.yaml`.
- RQ3 (ranking): does L1 (true sibling pairs) improve ordering/learning
  over L0 on the same arch/data/seed? Config: `configs/v02/loss_L1.yaml`,
  driver `training/experiments/ranking_ablation.py`, pairs from
  `tools/dataset/build_siblings.py` (teacher recorded per pair).
- RQ4 (sampling): does S1 (stratified + 20% disagreement) beat S0 after the
  clean baseline? Configs: arch file (S0) vs `configs/v02/sampling_S1.yaml`.
  Composition before/after via `tools/analysis/disagreement_report.py`.

## Mandatory matrix

- Architecture: A0 rune_mlp, A1 rune_attn_gab, A2 rune_rel_s, A3 rune_rel_d.
- Loss: L0 value+WDL, L1 +ranking (candidates only, sibling pairs).
- Sampling: S0 stratified, S1 +disagreement minority.
- Quantization: Q0 fp32, Q1 int8 (int16 measured in between).

No full Cartesian product. One axis varies per comparison; the fixed axes
are recorded in every `metrics.json`.

## Order (spec section 22)

1. v0.1 audit — done (`docs/experiments/v01-audit.md`).
2. Microbench bottleneck analysis — done, `rune_stage_bench` + infer bench.
3. Relational mixer experiment — code + parity done, training pending data.
4. GAB static vs dynamic — code done, training + quiet-bench pending.
5. Token count/dim ablation — code done, training pending.
6. Candidate selection — pending 100M reports.
7. Ranking-loss experiment — pipeline done, needs teacher + pairs.
8. Disagreement experiment — pipeline done, needs teacher scores.
9. Quantization benchmark — tooling done, needs trained weights.
10. Engine match — tooling ready (metadata + 95% CI), needs candidates.

## Budgets

Screening 25M/50M/100M for all candidates; 250M explicit only;
500M/1B template only. Promotion is human, on per-metric reports, never a
single score. Small match differences are reported with uncertainty, never
as conclusions.
