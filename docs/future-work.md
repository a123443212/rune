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
