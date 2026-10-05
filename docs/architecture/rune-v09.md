# RUNE v0.9 Architecture: Teacher Quality + Target Engineering

Question: how trustworthy is what RUNE learns from. Richer,
calibrated, provenance-complete targets — never a bigger model
(§3: strongest v0.8 candidate stays frozen while targets vary).

## Teacher reference hierarchy (§4)

One engine, several effort levels, no perfection axioms:

```
Teacher-Low → Teacher-Medium → Teacher-High → Teacher-Reference
```

Effort = depth/nodes/time budget, fixed per level. Higher effort
is a more expensive opinion, not ground truth — convergence is
measured (§5), never assumed.

## Multi-depth records (§5–§6)

On a representative subset, every position carries all levels:

```
score_low / medium / high / reference (+ WDL each)
Δ_depth = |eval(depth_n) − eval(depth_n−1)|
WDL flips, ranking flips, uncertainty proxy
```

Full master at reference depth is never required; the subset
pays for the convergence curve once, the cascade spends wisely
forever after.

## Target schema (§10–§11, §38)

```
value, WDL (hard class + soft probabilities), ranking pairs
(strict/soft/tie), uncertainty, stability scalars, search metadata
(depth, nodes, limits, threads, hash MB, engine/network ids+hashes,
generation config hash)
```

Every field optional except position identity. Every sample has a
validity state (VALID / PROVISIONAL / UNSTABLE / INVALID) assigned
by documented config rules; INVALID never trains silently.
Provenance traces position → teacher → config → target version.

## Quality then weighting (§18–§19)

Confidence proxy from depth convergence + teacher agreement +
WDL/rank consistency — explainable arithmetic, no new network.
`L_target = w(x)·L(student, teacher)` with bounded `w`, uniform
control always running. Uncalibrated weights stay banned.

## Cascade (§21–§22, §34)

Deterministic metadata routing over a fixed depth ladder:

```
Teacher-Low → stable? accept : escalate → … (never “until perfect”)
```

No neural router. Per-level budgets measured separately so T0–T3
comparisons price the cascade honestly.

## Storage (§36–§39)

Versioned target artifacts (never overwritten, never silently
migrated), position↔target join with duplicate/missing/stale/
mismatch detection, packed layout benchmarked for bytes/position
vs read/decode cost. Corrupt targets fail loudly; recovery counts
skips.

## Explicit non-goals (§61)

RL, MCTS, policy nets, giant ensembles, generative data models,
self-play infra, large Transformers, new search algorithms.
New directions go to `docs/future-work.md`, not into this version.
