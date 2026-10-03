# RUNE v0.3 — v0.2 Audit (evidence-gated, input to v0.3)

Scope: everything v0.2 built, measured against v0.3's question
(information per dimension / byte / operation). A component is
PROVEN only with a trained comparison under identical conditions.
Bench, parity, and smoke runs measure cost and correctness, never gain.

Current honest status (from `docs/experiments/rune-v02-results.md`
and `docs/v02-research-report.md`): **no 25M+ trained runs exist, no
teacher labels exist, no engine matches exist.** Every strength claim
about v0.2 components is therefore UNKNOWN, not refuted, not confirmed.

## 1. What was inspected

- Code: `training/models/relational.py` (FlexEmbedder, RelationalMixer,
  RelationalHead), `training/models/token_layout.py` (6/8/10 layouts),
  `training/models/rune_models.py` (v0.1 MLP/ATTN/GAB paths),
  `training/losses/composite.py` (L0/L1), `training/trainer/trainer.py`,
  `training/experiments/screening.py`, `training/export/export.py`
  (fp32/int16/int8, checksum for RUNE-03).
- C++: `core/architectures/relational/`, `core/accumulators/`
  (flex path, token layouts), `core/model_io/` (format 1, additive
  fields), `benchmarks/stage_bench.cpp` (+ dense section already added).
- Configs: all 11 files under `configs/v02/` (screen_arch, tokens_6/8/10,
  dims_24/40, gate_hard_sigmoid, loss_L1, sampling_S1, explicit_250m,
  template_500m_1b).
- Results: `rune-v02-results.md` (code-level evidence only),
  `v02-audit.md`, `v02-research-report.md`, `v01-audit.md`.

## 2. Verdicts

### KEEP (frozen into v0.3 Core or harness)

| Component | Evidence | v0.3 role |
| --------- | -------- | --------- |
| Grouped feature set `grouped_hkav2_fullthreats_v01` | Frozen since v0.1, parity-verified | Frozen, no redesign in same experiment as arch change (§20) |
| Grouped/flex accumulator math + clipping | Incremental == refresh, C++/torch parity 1e-4 | Frozen |
| 8x32 identity representation as baseline | Only layout with full compat (MLP + export + INT paths) | A0 baseline, `configs/v03/core.yaml` |
| INT16 embedding path | Measured neutral on random weights (diff ~3e-8, consistency 1.0) + ~30% smaller files | KEEP, closest thing to a measured efficiency gain |
| Screening harness (provenance, seed derivation, metrics.json, promotion gate) | Code-reviewed, smoke-run at 400/800 positions | KEEP, extended with dense models + budget fields |
| Export roundtrip + checksum + tamper rejection | Covered by `test_dense.py` / `test_dense.cpp` | KEEP, mandatory for every new variant (§30) |
| Head 256->128->32 shape discipline | Dominant eval cost 33-40 us, kept fixed for clean comparisons | FROZEN during representation work (§13) |
| clip gate + alpha=1.0 defaults | Zero-cost defaults, all parity tests pass through them | Defaults, variants need equal-budget gain to displace |

### REMOVE (from core candidate; retained only as experimental comparator or deleted)

| Component | Evidence | v0.3 role |
| --------- | -------- | --------- |
| Dynamic GAB (B2, +128 params at 8x32) | Measured overhead above op-count expectation on noisy machine, zero benefit evidence, pre-registered kill rule in v0.2 plan | STRIPPED FROM CORE. Survives only as a dead comparator if someone reruns it at 100M with per-us accounting; dies permanently on NPS regression without metric gain |
| QKV projections as default mixer cost | 15-57 us noisy, dominant mixer cost, no trained gain | REMOVED from core path. v0.3 core mixer is none/grouped-MLP. Relational mixer stays behind `research.allow_experimental` solely for the §13 check (good representation + simple mixer vs + v0.2 mixer) |
| 10-token / 24-40 uniform scaling as a quality claim | Cost ordering confirmed (10x40 strictly most expensive), quality ordering UNKNOWN | REMOVED as a claim. The axis is reworked into fixed-budget variable-dim experiments (§7), never uniform capacity increases |
| Ranking loss as default-on | 128-position smoke only, no ordering gain measured | REMOVED from default. L1 driver stays for candidates only, true siblings only, with value/WDL regression check (§19) |
| Disagreement sampling as default-on | Pipeline only, labeling cost and gain both unmeasured | REMOVED from default. Stage-gated after S0 baseline, labeling cost reported (§20) |
| Full Transformer / Mamba / MoE / contrastive / bilinear / giant head | No v0.2 evidence pointing at any of them; bottlenecks measured are head-first-layer and QKV cost, not capacity starvation | BANNED for v0.3 (§34), parked in `docs/future-work.md` |

### REWORK (idea survives, implementation changes)

| v0.2 form | Problem | v0.3 form |
| --------- | ------- | --------- |
| Token layouts 6/8/10 (merge/split heuristics) | Changes total capacity AND grouping at once, so no comparison isolates allocation | Fixed-budget variable dims: same ~256 total, 8 entries vary. Allocation is a hypothesis under test, re-derived from representation analysis, never intuition-hard-coded (§6-§7) |
| Per-token DxD projection implicit in embeddings | Parameter redundancy unmeasured | Explicit P0 (per-token pool) vs P1 (shared + token-specific scale/bias) at equal-ish params (§9, §23) |
| Gate registry (clip / hard_sigmoid on attention scores) | Gates the wrong thing for v0.3's question | Tiny bounded channel gate after tokenization, quant-friendly, kill-on-no-gain (§10) |
| Quantization global-only, random-weight numbers | Says nothing about trained nets or which token suffers | Per-token sensitivity analysis first (dynamic range, saturation, error per T0..T7), then scale-granularity test kept small (§18) |
| Noisy-machine latency ranges | Ratios usable, absolutes not publishable | Quiet-machine protocol before any inference claim; per-stage split (feature/accumulator/pool/gate/mixer/head) to find cost per useful information (§15-§16) |

### UNKNOWN (needs 25M+ data before any verdict)

- Static GAB effect (cost ~0.2-1 us is real, effect is not).
- Gate/alpha variant effects at fixed budget.
- 6-token compact quality (cheapest RQ1 test, MLP-only first per v0.3 Core order).
- Ranking L1 ordering gain without value/WDL regression.
- Disagreement S1 value vs labeling price.
- Optimal total dims (128/192/256/320 all untested).
- Whether any relational mixer is needed once representation improves (§13).

## 3. Answers to the v0.3 entry questions

1. **V0.2 bottleneck thực sự là gì?** Measured: head first layer
   (256->128) dominates eval, QKV dominates mixer, attention core
   (~2.5 us) and GAB (~0.2-1 us) are negligible. Data-side: no teacher
   labels, so no learning curve exists at all. The v0.1 claim "attention
   is the bottleneck" is refuted; everything about representation
   quality is unmeasured.
2. **Component nào có gain thật?** None on strength. INT16 is the only
   efficiency item with measurements (accuracy-neutral on random
   weights, smaller files). Everything else is UNPROVEN either way.
3. **Component nào chỉ tăng compute?** Dynamic GAB, QKV projections,
   10x40 uniform scaling (cost measured, benefit zero-evidence).
4. **Attention có thật sự cần không?** Unknown — exactly what the §13
   experiment (good representation + simple mixer vs + v0.2 mixer) must
   decide. Core must not assume it.
5. **Ranking/disagreement có đáng không?** Unknown, both gated.
6. **Failed experiments:** `split_by_game` starving val/test on small
   pools (fixed: deterministic fallback + fail-fast guard);
   `tools/screening/*` sys.path off-by-one masked by pytest (lesson:
   smoke-test CLIs standalone too).

## 4. Consequence for v0.3 core candidate

v0.3 Core = cheapest proven-working configuration (8x32, grouped MLP,
L0, S0, Q0/Q1) plus measurement harness, with every unproven component
behind explicit opt-in. Nothing unproven is deleted (deletion would
destroy the experiments that can answer the questions); defaults are
narrowed instead. Representation work (variable dims, pooling, shared
projection, channel gate) enters only through the A0-A3 / P0-P1 ablation
ladder at fixed budget, smallest scale first (25M), promotion by
per-metric reports, never a single score.

## 5. Pre-existing v0.3 code noted by this audit

`RUNE-03-A/B/C/D` dense paths (C++ + Python + bindings + parity tests
+ checksum IO + stage-bench section) already exist. Mechanism coverage
is complete (none / per-token / +gate / shared). What is NOT done and
this audit requires: variable-dim allocation hypotheses at fixed budget
(all presets currently ship [32]*8), budget ladder configs
(128/192/256/320), representation/redundancy analysis tooling,
per-token quantization sensitivity, and the v0.3 docs + report. Those
are the follow-up items, in that order.
