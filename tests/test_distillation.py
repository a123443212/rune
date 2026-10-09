import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch

from training.losses.distillation import (
    DistillCompositeLoss,
    WeightedDistillationLoss,
    confidence_weights,
)
from training.models.students import STUDENT_BUDGETS, build_adaptive_student, build_dense_student


def test_confidence_weights_bounded():
    u = torch.tensor([0.0, 0.5, 1.0])
    for mode in ("uniform", "confidence", "difficulty"):
        w = confidence_weights(u, mode)
        assert bool(((w >= 0.25) & (w <= 1.0)).all())
    assert torch.equal(confidence_weights(u, "uniform"), torch.ones(3))
    assert confidence_weights(u, "confidence")[0] > confidence_weights(u, "confidence")[2]
    assert confidence_weights(u, "difficulty")[2] > confidence_weights(u, "difficulty")[0]
    with pytest.raises(ValueError):
        confidence_weights(u, "bogus")


def test_weighted_distill_matches_uniform():
    torch.manual_seed(0)
    s = torch.randn(8)
    t = torch.randn(8)
    plain = WeightedDistillationLoss()(s, t, torch.ones(8))
    import torch.nn.functional as F

    assert abs(plain.item() - F.mse_loss(s, t).item()) < 1e-6


def test_distill_composite_task_and_alpha():
    torch.manual_seed(1)
    n = 6
    sv, tv = torch.randn(n, requires_grad=True), torch.randn(n)
    sw = torch.randn(n, 3, requires_grad=True)
    tw = torch.randn(n, 3)
    value = torch.randn(n)
    wdl = torch.randint(0, 3, (n,))
    fn = DistillCompositeLoss(task="value_wdl", alpha=0.5)
    parts = fn(sv, sw, tv, tw, value, wdl)
    assert parts["task_value"].item() > 0
    assert parts["task_wdl"].item() > 0
    assert parts["distill"].item() > 0
    assert parts["distill_wdl"].item() > 0
    assert parts["rank"].item() == 0.0
    pure = DistillCompositeLoss(task="value_wdl", alpha=0.0)
    assert pure(sv, sw, tv, tw, value, wdl)["distill"].item() > 0
    nodist = DistillCompositeLoss(task="value_wdl", alpha=0.5)
    p0 = nodist(sv, sw, None, None, value, wdl)
    assert p0["distill"].item() == 0.0
    assert p0["distill_wmean"].item() == 0.0


def test_distill_weighted_vs_uniform_control():
    torch.manual_seed(2)
    n = 8
    sv = torch.randn(n)
    tv = torch.randn(n)
    value = torch.randn(n)
    wdl = torch.randint(0, 3, (n,))
    sw = torch.randn(n, 3)
    tu = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    uni = DistillCompositeLoss(weight_mode="uniform")(sv, sw, tv, None, value, wdl)
    con = DistillCompositeLoss(weight_mode="confidence")(
        sv, sw, tv, None, value, wdl, teacher_u=tu)
    assert uni["distill_wmean"].item() == 1.0
    assert 0.25 <= con["distill_wmean"].item() <= 1.0
    assert con["distill"].item() != uni["distill"].item()


def test_student_budgets_shrink():
    sizes = {}
    for name in ("S1", "S2", "S3", "S4"):
        m = build_dense_student(budget=name)
        m.eval()
        sizes[name] = m.parameter_count()
        assert m.model_spec()["head_h1"] == STUDENT_BUDGETS[name]["head_h1"]
        assert m.model_spec()["token_dims"] == [STUDENT_BUDGETS[name]["token_dim"]] * 8
    assert sizes["S1"] > sizes["S2"] > sizes["S3"] > sizes["S4"]
    with pytest.raises(ValueError):
        build_dense_student(budget="S9")


def test_student_forward_and_export():
    m = build_dense_student(budget="S2")
    m.eval()
    ids = [torch.randint(0, 64, (2, 4)) for _ in range(9)]
    masks = [torch.ones(2, 4) for _ in range(9)]
    with torch.no_grad():
        v, w = m(ids, masks)
    assert v.shape == (2,) and w.shape == (2, 3)
    a = build_adaptive_student(budget="S3")
    assert a.parameter_count() < build_adaptive_student(budget="S1").parameter_count()


def test_student_report_fields():
    from training.models.students import student_report

    r = student_report(build_dense_student(budget="S2"))
    assert {"params", "param_bytes", "cheap_params", "spec"} <= set(r.keys())
    assert r["param_bytes"] == r["params"] * 4
