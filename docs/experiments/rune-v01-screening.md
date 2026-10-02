# RUNE v0.1 Rapid Validation (Screening)

## Workflow

```
FAST SCREENING (10M -> 25M -> 50M -> 100M, all architectures, loss A)
  -> compare per-metric reports under identical conditions
  -> human promotion decision (tool presents evidence, never auto-selects)
  -> CANDIDATE 250M (explicit config only)
  -> DEEP 500M-1B+ (template only, justified by 250M evidence)
```

Loss discipline: Stage A (`Value + WDL`) screens architectures.
Stage B (`+ Ranking`) runs only on promoted candidates, so architecture gain
and loss gain stay separated. Disagreement sampling runs only after the
clean + random/stratified baseline exists, with teacher labeling cost tracked
separately.

## How to run

```bash
python tools/dataset/build_dataset.py --out data/pool.jsonl --games 5000
python tools/screening/run_screening.py --config configs/v01/screening/screen_lossA.yaml --pool data/pool.jsonl
python tools/screening/report.py --runs runs/rune_v01_screening --out-dir reports/screening
python tools/screening/plot.py --runs runs/rune_v01_screening --out-dir reports/screening
python tools/benchmark/infer_bench.py --build-dir build --out reports/screening/infer.json
python tools/screening/promote.py --runs runs/rune_v01_screening
```

Screening entry configs: `configs/v01/screening/screen_lossA.yaml` (default),
`screen_lossB_ranking.yaml` (candidates only), `explicit_250m.yaml`,
`template_500m_1b.yaml`. The pipeline stops at 100M unless a 250M+ config
is explicitly supplied. Checkpoints are never overwritten; resume continues
from the previous milestone.

Fairness rules enforced by the runner: one shared clean pool, one fixed
game-grouped split and seed, one teacher version for every model. Comparing
models trained on different position counts is not a valid architecture
comparison.

## Smoke validation (plumbing only, not evidence)

A smoke run on 545 synthetic random-playout positions, milestones
1k/2k/4k, all three models, verified: sequential checkpoints with metadata,
per-milestone reports and plots, inference bench split, promotion evidence
table. Val losses were non-zero and finite, grad NaN count zero. This proves
the workflow executes; it says nothing about architectures.

## Inference cost split (measured, `rune_stage_bench`)

| Model | Accumulator | Attention mixer | Head | Full eval (incr) |
| ----- | ----------- | --------------- | ---- | ---------------- |
| RUNE-MLP | ~13-16 us | 0 | ~33-40 us | ~45 us |
| RUNE-ATTN | ~13-16 us | ~15-18 us (QKV dominates; QKT+GAB+Vmix ~2.5 us) | ~33 us | ~50-64 us |
| RUNE-ATTN-GAB | ~13-16 us | ~16-25 us | ~33 us | ~50-65 us |

Attention is not the bottleneck; the 256->128 head is. GAB adds ~0.2 us
(bias add + clip over 64 elements).

## Report questions

1. **Do models converge stably?** Unknown at screening scale. Smoke runs show
   finite losses and zero NaN grad steps; real answer needs the 10M-100M runs
   with grad-mean/max trend recorded in every `metrics.json`.
2. **How do learning curves differ?** No data yet. `plot.py` produces
   val-loss / WDL-acc / rank-acc / throughput curves once milestones complete.
3. **Attention inference overhead?** Measured: roughly +5-20 us per eval over
   MLP on this machine, almost entirely QKV projections.
4. **GAB extra cost?** Measured: negligible (~0.2 us); +64 parameters.
5. **Is grouped MLP a strong baseline?** No data yet. It trains slightly
   faster per position (fewer params) and is the mandatory reference for H1.
6. **Which model deserves 250M?** Undecided. Promotion requires the 100M
   reports; `promote.py` prints evidence but selects nothing automatically.
7. **What lacks data?** Everything about strength and data efficiency:
   no teacher labels, no 10M+ runs, no engine matches, no strength-vs-cost
   curves. No data-efficiency claim is made.
8. **Next bottlenecks?** Inference: the head, not attention. Training:
   dataloader throughput on real data is unmeasured. Data: teacher labeling
   pipeline does not exist yet.

## Deliverable status

Rapid configs, checkpoint-per-milestone, seeds, metrics logging, comparison
reports, plots, microbench suite, inference bench, promotion workflow,
ranking ablation config, disagreement config: implemented and smoke-tested.
Real screening results: pending teacher-labeled data and compute.
