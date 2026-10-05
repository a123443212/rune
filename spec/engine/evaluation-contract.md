# RUNE evaluation contract (normative)

`evaluate(position)` returns exactly `value`, `wdl`, and optionally `uncertainty` plus routing info. Production search uses only what it needs. Debug and analysis metadata never rides the default hot path.

## Output

```text
value        float32 side-to-move relative, canonical range [-1, 1]
wdl          [win, draw, loss] non-negative, sums to 1
uncertainty  float32 in [0, 1] when the model provides it, else 0
refine       bool routing decision when adaptive, else false
mode         full | cheap | lazy-skipped
```

Python `training/engine/contract.py`, Rust `rune-search/src/contract.rs`, C++ `core/engine/eval_contract.*` implement the same type. Minimal projection (`to_minimal`, `needs_only`) strips everything except value and WDL for the search loop.

## Scale

```text
network output (tanh, [-1, 1])
  -> canonical value (clamp, NaN -> 0)
  -> engine score (round(value * 1000), mate guard +-8999)
```

Side-to-move convention is negamax everywhere: positive means good for the player to move. WDL is normalized on construction. `value_from_wdl = win - loss` must agree with value within 0.35 or the pair is flagged inconsistent; the fix belongs in targets or calibration, never in arbitrary post-processing.

Mate and extremes: the network is bounded and never emits mate scores. Values at or beyond 0.95 are extreme evaluations, not mates. Engine-level mate handling stays explicit and separate.

## Side to move

White-to-move and Black-to-move versions of one board must differ in FEN, in cache key, in sign behavior, and in WDL orientation. Tests pin this in all three languages. There is no POV ambiguity: every stored value is STM-relative.

## Cache identity

Cache keys cover board plus side to move plus castling plus en-passant semantics plus model hash plus evaluation mode. A result from another model, version, or mode never hits. Model hash is part of the identity, so `network A` cache never serves `network B` silently.

## Lifecycle

```text
load -> validate -> compile -> freeze -> search -> unload
```

No reload happens mid-search. Hot-swap, if ever supported, needs explicit synchronization. Weights are shared immutable; evaluator state is thread-local and never cloned per thread beyond its scratch space.
