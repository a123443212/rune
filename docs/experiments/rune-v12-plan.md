# RUNE v0.12 plan (running log)

## Questions under test

RQ1 static to strength, RQ2 scale and calibration, RQ3 uncertainty in tree, RQ4 latency versus strength, RQ5 safe lazy eval, RQ6 transposition and context consistency.

## Matrices

```text
E0 non-adaptive full | E1 adaptive | E2 adaptive+lazy | E3 +selective precision
Q0 FP32 | Q1 INT8 | Q2 selective (only if justified)
Runtimes: C++ generic, C++ compiled, Rust generic, Rust compiled (same model, same settings)
Search: single-thread and multi-thread, NPS, nodes, time-to-depth, match result
```

## Method rules

Same model, same hardware, same threads, same hash size, same time control, same openings, recorded model hash, engine commit, runtime and compiler versions, dataset id, and seed. Ranges, never single digits. A faster evaluator with worse search behavior is not an improvement. Static accuracy up plus match strength down gets written up as a regression with numbers, not accepted.

## Search-native set

Sampled from real benchmark position files across quiet, tactical, endgame, king-attack, and random lines (`tools/search/build_search_set.py`), teacher-labelled in the analysis pipeline, kept separate from regular validation. Ranking changes between the two sets count as a finding, not a failure to hide.

## Status

Contract, scale, cache, lazy, integration layers, calibration tools, search bench, search diff, tactical and endgame suites, configs, and CI are implemented. Controlled alpha-beta matches run through `tools/match/play_alpha_match.py` where bindings exist. Search-native fine-tuning stays an experiment gated on a clear gap, with a regular validation set held out so nothing overfits to search nodes.
