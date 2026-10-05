# Failed: evaluator cache inside engine TT

Attempt: give the RUNE runtime its own transposition-table-shaped cache overlapping the engine TT.

Result: not built. The evaluator cache stays a small, explicitly keyed side cache for harnesses without an engine TT. Where an engine TT exists, adding a second one is overhead without a measured hit-rate justification.
