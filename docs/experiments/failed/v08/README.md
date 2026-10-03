# v0.8 failed experiments (index)

Detailed entries live in `docs/experiments/rune-v08-results.md`
(Failed experiments / fixes); this directory holds standalone
post-mortems for track-killing findings. None yet — no learning
claim has been tested, so nothing has earned a post-mortem.

## Near-misses fixed during construction (not track failures)

- Teacher key-set split (`teacher_value` vs `teacher_v` across
  labelers): unified with legacy aliases; convention documented.
- Trainer batch-shape collision (len-5 teacher vs context):
  explicit len-6 None-ctx form.
- Distill metric prefix mismatch (`dist_*` vs expected keys).
- YAML screening config loaded as JSON in rounds orchestrator.
- Game-less round pools breaking game-grouped split: audit now
  carries `game_hash`, rounds preserve true game grouping.
- Two-level selection non-monotonic in K′: diversity applied at
  both levels; fixed to score-only local stage (65–97%→exact
  convergence verified).
- Coverage reader crashing on binary shards: format-sniffing load.
