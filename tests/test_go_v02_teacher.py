import torch


def test_mcts_teacher_distills():
    from training.engine.go_mcts_teacher import distilled_label, distill_states, teacher_cost
    from training.games.go_v02 import GoGameV02
    from training.models.resnet import build_resnet
    torch.manual_seed(5)
    g = GoGameV02(size=9)
    m = build_resnet(board=9, channels=8, blocks=1, in_planes=8, feature_set="go_planes_v02")
    m.eval()
    s = "/".join(["." * 9] * 9) + " b - 7.5 1 0"
    r = distilled_label(g, m, s, simulations=4, seed=1)
    assert r["game"] == "go"
    assert r["teacher"] == "mcts_go_v02"
    assert r["move"] in g.legal(s)
    assert len(r["policy"]) == 82
    assert abs(sum(r["policy"]) - 1.0) < 1e-9
    assert r["sims"] == 4
    recs = distill_states(g, m, [s, s], simulations=4, seed=1)
    assert len(recs) == 2
    c = teacher_cost(recs)
    assert c["positions"] == 2
    assert c["simulations"] == 8
    assert c["labeling_seconds"] >= 0.0


def test_shard_pipeline_roundtrip(tmp_path):
    from training.datasets.go_shard import shard_pipeline
    from training.engine.go_mcts_teacher import distill_states
    from training.games.go_v02 import GoGameV02
    from training.models.resnet import build_resnet
    torch.manual_seed(5)
    g = GoGameV02(size=9)
    m = build_resnet(board=9, channels=8, blocks=1, in_planes=8, feature_set="go_planes_v02")
    m.eval()
    s = "/".join(["." * 9] * 9) + " b - 7.5 1 0"
    recs = distill_states(g, m, [s, s, s], simulations=2, seed=1)
    out = str(tmp_path / "go_data")
    manifest = shard_pipeline(g, recs, out, num_shards=2)
    assert manifest["positions"] == 1
    assert manifest["validated"] == 3
    assert len(manifest["shards"]) == 2
    import json
    with open(out + "/manifest.json") as f:
        back = json.load(f)
    assert back["positions"] == 1
    total = 0
    for sh in back["shards"]:
        assert len(sh["hash"]) == 16
        with open(out + "/" + sh["path"]) as f:
            for line in f:
                json.loads(line)
                total += 1
    assert total == 1


def test_sprt_math():
    from tools.match.go_sprt import elo_to_score, score_to_elo, sprt_bounds, sprt_decide, sprt_llr
    assert abs(elo_to_score(0.0) - 0.5) < 1e-9
    assert abs(score_to_elo(0.5)) < 1e-9
    assert sprt_llr(0, 0, 0) == 0.0
    lower, upper = sprt_bounds()
    assert lower < 0.0 < upper
    assert sprt_decide(upper + 1.0, lower, upper) == "accept"
    assert sprt_decide(lower - 1.0, lower, upper) == "reject"
    assert sprt_decide(0.0, lower, upper) == "continue"
    assert sprt_llr(10, 0, 0) > sprt_llr(5, 5, 0)
