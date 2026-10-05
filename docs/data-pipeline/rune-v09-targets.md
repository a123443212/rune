# RUNE Data Pipeline: Targets (v0.9)

How teacher supervision flows from engine to trainer without
losing provenance, quality signals, or cost accounting — and
without ever letting an uncharacterized label steer training.

## Target pipeline

```
master pool (legal → deduped → balanced, v0.7/v0.8 machinery)
  ↓ multi-depth labeling on subset (low/med/high/ref + WDL each)
stability analysis (deltas, flips, buckets; no magic thresholds)
  ↓ validity states (VALID/PROVISIONAL/UNSTABLE/INVALID per config)
selective high-cost labeling (cascade on deterministic metadata)
  ↓ quality weighting/filtering (bounded, uniform control present)
  → v0.8 active loop (now with teacher-stability/agreement/quality
     as ablated components, §32)
  ↓ final training dataset (immutable, hashed, manifest-pinned)
```

## Provenance record (per target generation run, §56)

```yaml
teacher: {id, engine_version, network_id, network_hash,
          search_limit, threads, hash}
dataset: {id, hash}
target:  {schema, version, seed}
```

`teacher_hash` covers the full record, not just (fen, value, id).
Target stores are versioned (`target-v01`, `-v02`, …); migration
is explicit or rejected.

## Determinism and consistency (§12–§13)

`teacher --deterministic`: fixed threads/hash/settings/version/
network/seed. Same position + same config must yield same target;
discrepancies trigger infra investigation (nondeterminism, hash,
threads, serialization, settings) before anyone says "noise".

## Cost doctrine (§23, §31, §50)

Cascade levels priced separately (T0 all-expensive vs T1 cheap
vs T2 cascade vs T3 active+cascade on equal teacher compute, not
just equal positions). Teacher generation, labeling, storage, and
student training reported as one total-compute-to-quality curve.
Diminishing returns per depth level decide the production teacher
effort (§53–§54) — possibly shallow, on evidence.
