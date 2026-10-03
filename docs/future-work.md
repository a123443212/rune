# Future Work (post-v0.1, do not implement during v0.1)

Ideas parked here instead of leaking into the v0.1 scope:

- Full-size HalfKAv2 feature set and a standalone SFNN feature path.
- Full integer attention and head matmuls (int8/int16 data path end to end).
- Phase-conditioned or dynamic geometric bias.
- Multi-head and multi-layer attention variants.
- Mamba/SSM mixer behind the architecture interface.
- Low-rank bilinear interaction block.
- Self-supervised pretraining and contrastive objectives.
- Phase experts and adapter-style specialization.
- Dynamic sparse layers and learned codebooks.
- Architecture search over token count and dimension.
- Search integration (alpha-beta + transposition table) for real match testing.
- Rust dataset tooling if PGN parsing or dedup becomes the bottleneck.
- AVX2/NEON SIMD kernels after correctness is locked and profiling shows need.

# Parked during v0.2 (do not implement during v0.2)

- Head redesign (the 256->128 layer is the eval bottleneck, but the head
  stays fixed until mixer comparisons conclude).
- Move-derived incremental feature updates (per-node full extraction
  dominates walk NPS).
- Teacher labeling pipeline (blocks RQ3/RQ4 training).
- Quiet-machine latency protocol with pinned frequency for publishable
  inference numbers.

# Parked during v0.3 (do not implement during v0.3)

- Full Transformer / multi-head / multi-layer attention.
- Mamba/SSM mixer, MoE, large recurrent networks.
- Self-supervised pretraining, contrastive learning, reconstruction
  or consistency losses.
- Complex bilinear blocks, giant heads, expert routing.
- Architecture search or random-search over allocations.
- Dynamic GAB revival: needs 100M per-us evidence over static, twice.
- Channel gate revival if A3 fails: needs a new hypothesis, not a retry.
- Non-uniform shared projection (shared width > 32): only if
  representation analysis shows sharing helps but width binds.
- Per-token quantization scales: only after per-token sensitivity
  analysis proves global scales are the bottleneck.

# Parked during v0.4 (do not implement during v0.4)

- MoE, full Transformer, Mamba/SSM over the tokens.
- Recurrent multi-step refinement ("refine until good enough").
- Large expert routing, reinforcement learning, neural search policy.
- Giant uncertainty networks (difficulty stays one linear layer until
  proven insufficient, then a stated hypothesis — never silent growth).
- L_compute as a reported scalar objective; quality, rate, avg and
  worst-case latency stay separate forever.
- FP16 refinement on CPUs where it shows no measured benefit.
- Search-aware evaluation beyond stability analysis (v0.5 topic,
  not v0.4 implementation).

# Parked during v0.5 (do not implement during v0.5)

- Full move policy network, neural move ordering, RL, alpha-beta
  replacement, MCTS, learned search trees.
- Recurrent multi-step search networks, large Transformer, MoE.
- Extra uncertainty forms beyond the bounded scalar (variance heads,
  ensembles, MC-dropout at inference) without a cost-first hypothesis.
- Learned stability scalar in the engine loop before child-spread
  analysis proves signal (stability head trains, engine use gated).
- Test-set threshold tuning; any calibration claim without buckets,
  rank correlation, and stratification.
- Quantization schemes beyond INT8 base + checked-uncertainty until
  P-legs report.

# Parked during v0.6 (do not implement during v0.6)

- Full policy network, RL, MCTS, giant Transformer, MoE, recurrent
  search network, full neural move ordering.
- Teacher ranking-pair data pipeline without bindings (move gen is
  C++-only; D4 waits for `build_siblings.py` on a host with
  bindings, or a Python move generator out of scope here).
- QAT-training (fake-quant in the optimizer loop); Q2 currently
  means INT8 export + full eval, honestly labeled.
- Adaptive-family students (builders support widths; configs stay
  dense-first until D-legs show signal).
- Difficulty-aware sampling configs (sampler exists; leg runs after
  A-ladder, per plan order).
