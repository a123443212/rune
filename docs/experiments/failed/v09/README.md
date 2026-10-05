# v0.9 failed experiments (index)

Detailed entries live in `docs/experiments/rune-v09-results.md`.
Track-killing findings get standalone files here. None yet —
no v0.9 learning leg has run, so nothing has earned a post-mortem.

## Near-misses fixed during construction (not track failures)

- Hash-state teacher nondeterminism (2% exact-match → 100% via
  Clear Hash): infrastructure, not noise. Deterministic mode now
  clears per position; consistency tool proves it.
- Loss weighting fallback swallowing quality-only configs:
  restructured so uniform+quality uses quality weights.
- `verify-targets` failing on legitimately unlabeled records:
  corrupt-only failure semantics.
- Ranked-heap tie-break by input order broke two-level/monotonic
  reasoning: identity-hash tie-break (found via K′-sweep).
