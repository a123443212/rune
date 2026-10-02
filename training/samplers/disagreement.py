import random


def score_records(records):
    scored = []
    for r in records:
        if "teacher_value" not in r or "student_value" not in r:
            continue
        gap = abs(r["teacher_value"] - r["student_value"])
        wdl_dis = 1 if r.get("teacher_wdl", r.get("wdl")) != r.get("student_wdl",
                                                                   r.get("wdl")) else 0
        scored.append({**r, "dis_gap": gap, "wdl_disagree": wdl_dis})
    return scored


def disagreement_metrics(scored):
    if not scored:
        return {}
    gaps = [r["dis_gap"] for r in scored]
    phases = {}
    for r in scored:
        ph = r.get("phase", -1)
        phases.setdefault(ph, []).append(r["dis_gap"])
    return {
        "n": len(scored),
        "mean_abs_gap": sum(gaps) / len(gaps),
        "max_gap": max(gaps),
        "wdl_disagree_rate": sum(r["wdl_disagree"] for r in scored) / len(scored),
        "mean_gap_by_phase": {str(k): sum(v) / len(v) for k, v in phases.items()},
    }


def sample_mixture(records, n, mode="stratified", disagreement_ratio=0.2, seed=0):
    from training.datasets import pipeline as P

    rng = random.Random(seed)
    if disagreement_ratio <= 0:
        if mode == "stratified":
            return P.sample_stratified(records, n, seed=seed)
        return P.sample_random(records, n, seed=seed)
    scored = sorted(score_records(records), key=lambda r: r["dis_gap"], reverse=True)
    n_dis = int(n * disagreement_ratio)
    top = scored[:n_dis]
    top_ids = {id(r) for r in top}
    rest = [r for r in records if id(r) not in top_ids]
    n_rest = n - len(top)
    if mode == "stratified":
        base = P.sample_stratified(rest, n_rest, seed=seed) if rest else []
    else:
        base = P.sample_random(rest, n_rest, seed=seed) if rest else []
    out = top + base
    rng.shuffle(out)
    return out
