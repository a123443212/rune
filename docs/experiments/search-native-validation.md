# Search-native validation

Regular validation asks whether the network is good on curated positions. Search-native validation asks whether it stays good on the positions search actually visits.

## Pipeline

```text
benchmark position files
  -> build_search_set sampling (material and phase tracked)
  -> evaluator outputs on real nodes
  -> teacher labels in the analysis pipeline
  -> calibration, boundary, ranking, and WDL reports
```

Only a subset needs labels. The goal is a Search-Native Evaluation Set, not a relabelled tree.

## What v0.12 found

On 20 positions across quiet, tactical, endgame, king-attack, and random lines, FP32-as-teacher versus INT8-as-pred gives MAE 2.55e-06 and RMSE 2.99e-06, all inside the near-equal bucket, zero sign errors, zero WDL mismatch, zero boundary flips at alpha 0.1 and beta 0.3, and zero ranking errors over 30 child pairs with mean delta error 3.57e-06.

Caveat, stated plainly: the shipped fixtures are untrained, so evaluations cluster near -0.018 and every bucket is near-equal. The machinery and the agreement are proven; the distribution is not. The same caveat applied to the v0.10 adaptive sweep, and it applies here. A trained model should re-run this pipeline before any strength claim.

## Distribution shift watch

Training, validation, and actual search-node distributions are compared on evaluation, phase, material, uncertainty, tacticality, and refinement rate. A model that wins on the dataset but meets a different distribution in search gets flagged here, which is exactly the case RQ5 of v0.11 predicted and v0.12 now measures.

## Fine-tuning policy

Search-native fine-tuning is tried only on a clear gap, keeps a regular validation set aside, and never mutates the production dataset. Rounds and versioning follow v0.8. No feedback loop is created automatically.
