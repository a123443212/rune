# Failed: Python in the search hot path

Attempt: route leaf evaluations through the Python reference evaluator inside alpha-beta.

Result: per-call overhead dominates by orders of magnitude and NPS collapses. Python stays in analysis, training, calibration, and orchestration. The reference search in `training/engine/search.py` exists for replay and debugging, never for strength measurement.
