# RUNE v0.3 Core

## Definition

v0.3 Core is the minimal proven-working configuration. It is the default;
everything else is experimental and requires explicit opt-in.

```
Board
-> grouped_hkav2_fullthreats_v01 features (frozen)
-> incremental grouped accumulator (frozen)
-> 8 x 32 tokens (frozen default)
-> Grouped MLP mixer (RUNE-MLP, frozen)
-> 128/32 value+WDL head (frozen)
-> L0 loss (value + WDL), S0 sampling (stratified), Q0/Q1 export
```

Pinned by `configs/v03/core.yaml`. Arch id stays `RUNE-MLP` for
compatibility; "v0.3 Core" is a configuration status, not a new network.

## Frozen (do not touch without a core-change proposal)

Feature set and version, accumulator math and clipping, token defaults,
MLP shapes, head shapes, `.rune` format handling of v0.1/v0.2 files,
INT16/INT8 embedding paths, experiment harness (screening, reports,
promotion, match metadata).

## Experimental registry (opt-in only, code retained)

| Item | Gate | Re-entry bar |
| ---- | ---- | ------------ |
| Attention / relational mixer (A1/A2) | `research.allow_experimental` | 25M+ A0-vs-A2, metric gain, no NPS regression |
| Static GAB as effect claim | same | gain over no-bias at fixed cost |
| Dynamic GAB (A3) | same | gain over static that survives per-us accounting |
| Gate/alpha variants | same | gain over clip/alpha=1.0, same budget |
| 6-token compact | same | FIRST perexperiment: MLP-only 25M vs core |
| 10-token, 24/40 dims | same | gain per param and per us, not just parity |
| Ranking L1 | L1 driver only, candidates only | ordering gain without value/WDL regression |
| Disagreement S1 | stage-gated after S0 | gain with labeling cost included |

Re-entry is per-metric evidence at 100M (selection) and 250M
(confirmation), never a single score. A component that fails twice at
matched conditions is removed from the registry, not kept as an option.

## First three v0.3 experiments (in order)

1. 6-token MLP vs core 8x32 MLP at 25M (RQ1, cheapest question first).
2. Winner of (1) vs relational-static at 25M/50M (RQ2, attention necessity).
3. L1 on the winner of (2) with true siblings (RQ3). S1 only after (2)
   establishes the baseline (RQ4).

## Non-goals

No new architectures, no head redesign, no quant changes beyond INT16,
no 500M+ runs before 100M selection. Anything else goes to future-work.
