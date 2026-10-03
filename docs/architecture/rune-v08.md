# RUNE v0.8 Architecture: Closed-Loop Active Learning

Question: same or better quality with fewer useful positions and
lower total teacher cost. Better positions, not more positions.

## Loop

```
Master Pool (streamed, never fully in RAM)
  ↓ candidate sampler
Current RUNE Model (seed first — never active-learn from noise, §17)
  ↓ scoring (S1–S5, each stored separately)
Selection (top-K, diversity, coverage floor, deterministic)
  ↓ teacher labeling (lookup first: reuse, never re-query, §28)
Append to dataset (immutable round artifacts, never overwrite)
  ↓ retrain / fine-tune (T0/T1/T2 documented per round)
New model → rescore remaining pool → repeat
```

Rounds are versioned `active-round-NNN` with model, dataset,
teacher, score, and selection ids. Stopping is configurable
(marginal gain, budget, exhaustion, collapse, target hit).

## Signals (no assumed winner)

```
S1 disagreement      |teacher − student| (needs labeled pool: re-scoring)
S2 uncertainty       student u or WDL entropy (works unlabeled)
S3 instability       child spread (needs pairs data)
S4 ranking disagree  sibling order flips (needs pairs data)
S5 rarity            1 − bucket_freq/peak (computed in-engine, two passes)
```

`score = Σw·norm(component) / Σw`, weights configurable,
components exposed in the audit trail. Single-signal baselines
run before any multi-signal claim (§7). Unlabeled pools can only
use student-side signals — a documented asymmetry, not a bug.

## Selection (Rust, metadata-only)

Bounded heap top-K → diversity bucket caps (phase×material) with
refill rescan → stratified coverage floor (configurable ratio,
marked `stratified` in the record) → requested/reused split.
Two-level sharded selection (per-shard top-K′, global merge) is
EXACT when K′ covers the shard and measured-approximate otherwise
(65–70% at K′=400–1000 on 4.5k/3-shard test; converges by K′≥shard).
Tie-breaks use identity hash, never input order. Neural inference
never enters Rust (§30).

## Coverage and safety (§9–§10, §38, §44–§45)

Per-round coverage across phase/material/eval-range/source axes
vs pool and vs batch; minimum random/stratified floor; immutable
inputs; quality gates reject corrupt labels; extreme disagreement
is bucketed, never auto-trusted as gold.

## Cost honesty (§13–§14)

Every round ledgers labels requested vs reused, teacher compute,
wall time, storage, plus student training compute. Efficiency
claims use quality-vs-labels AND quality-vs-total-compute curves.
Teacher supervision is never called free; synth/random-init
stand-ins are labeled as pipeline validation only.

## Training modes (§39–§41)

T0 scratch, T1 continue (via `init_ckpt`, round budget measured
fresh, cumulative positions recorded), T2 mixed + configurable
replay. Forgetting is tested on fixed probe subsets (quiet/endgame/
common); rising probe error is forgetting until proven otherwise.

## Explicit non-goals (§63)

RL, MCTS, policy training, generative data models, self-play
generators, architecture search, composite winner scores,
test-set tuning, production dependency on selection metadata.
