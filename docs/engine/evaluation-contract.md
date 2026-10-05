# Evaluation contract for engine builders

Production search consumes `value` and `wdl`. `uncertainty`, `refine`, and `mode` are optional signals for lazy and adaptive paths. Everything else (traces, tokens, mixer internals, feature lists) is analysis metadata and stays out of the default path via `to_minimal` and `needs_only`.

Scale chain: raw network output, canonical value in [-1, 1] with NaN mapped to 0, engine score as mills clamped away from the mate guard band at +-9000. WDL always normalized. `value_from_wdl` gives the consistency check; mismatches beyond 0.35 are data or head problems, not an invitation for hidden rescaling.

Side to move is negamax always. Cache keys include every state the evaluation depends on. Model lifecycle is load, validate, compile, freeze, search, unload, with no mid-search reloads.
