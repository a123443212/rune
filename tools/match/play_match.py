import argparse
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class EvalStats:
    def __init__(self):
        self.evals = 0
        self.refined = 0
        self.seconds = 0.0
        self.unc_sum = 0.0
        self.unc_n = 0

    def refinement_rate(self):
        return self.refined / max(1, self.evals)

    def avg_latency_us(self):
        return self.seconds * 1e6 / max(1, self.evals)

    def mean_uncertainty(self):
        return self.unc_sum / max(1, self.unc_n)


def eval_position(model, fen, stats, mode="adaptive", threshold=-1e30):
    t0 = time.perf_counter()
    try:
        out = model.eval_fen(fen, mode, threshold)
    except TypeError:
        out = model.eval_fen(fen)
    dt = time.perf_counter() - t0
    stats.evals += 1
    stats.seconds += dt
    if len(out) == 6:
        v, _, _, refined, u, _ = out
        stats.refined += 1 if refined else 0
        stats.unc_sum += u
        stats.unc_n += 1
        return v
    if len(out) == 4:
        v, _, _, refined = out
        stats.refined += 1 if refined else 0
        return v
    v, _ = out
    return v


def elo_from_score(score, games):
    if games == 0:
        return 0.0, 0.0
    s = min(max(score / games, 1e-6), 1 - 1e-6)
    elo = -400.0 * math.log10(1.0 / s - 1.0)
    se = math.sqrt(s * (1 - s) / games)
    err = 1.96 * se * 400.0 / (s * (1 - s) * math.log(10))
    return elo, err


def greedy_move(board, model, rng, stats, epsilon=0.0):
    moves = board.legal_moves()
    if not moves:
        return None
    if rng.random() < epsilon:
        return rng.choice(moves)
    import rune_bindings as rb

    stm = board.side_to_move()
    best, best_v = None, None
    for m in moves:
        mv = rb.Move()
        mv.from_sq, mv.to_sq, mv.promo = m[0], m[1], m[2]
        if not board.make_move(mv):
            continue
        v = eval_position(model, board.to_fen(), stats)
        board.unmake_move()
        own = v if stm == 0 else -v
        if best_v is None or own > best_v:
            best, best_v = m, own
    return best if best is not None else rng.choice(moves)


def play_game(white_model, black_model, white_stats, black_stats, opening_fen, rng):
    import rune_bindings as rb

    b = rb.Board(opening_fen)
    for _ in range(200):
        moves = b.legal_moves()
        if not moves:
            return 0.5
        stm = b.side_to_move()
        model = white_model if stm == 0 else black_model
        stats = white_stats if stm == 0 else black_stats
        m = greedy_move(b, model, rng, stats)
        mv = rb.Move()
        mv.from_sq, mv.to_sq, mv.promo = m[0], m[1], m[2]
        b.make_move(mv)
    v = eval_position(white_model, b.to_fen(), white_stats)
    return 1.0 if v > 0.2 else (0.0 if v < -0.2 else 0.5)


def load_model_for_match(arch_id, runepath, build_dir):
    sys.path.insert(0, build_dir)
    import rune_bindings as rb

    from training.export.export import load_exported_arrays
    from training.models.rune_models import EXPORT_ORDER, build_model

    header, arrays = load_exported_arrays(runepath)
    if header["arch"] == "RUNE-REL-02":
        model = rb.FlexModel(header["tokens"], header["token_dim"], header.get("gate", "clip"),
                             header.get("alpha", 1.0), header["geometric_bias"] == "dynamic")
    elif header["arch"].startswith("RUNE-03-"):
        model = rb.DenseModel(header["arch"].split("-")[-1], header["token_dims"],
                              header.get("pooling", "none"),
                              bool(header.get("pool_clip", True)),
                              bool(header.get("gate_on", False)),
                              int(header.get("head_h1", 128)),
                              int(header.get("head_h2", 32)))
    elif header["arch"] == "RUNE-04":
        model = rb.AdaptiveModel(header["token_dim"], header.get("cheap_pooling", "none"),
                                 float(header.get("threshold", 0.5)))
    elif header["arch"] == "RUNE-05":
        model = rb.AdaptiveModel(header["token_dim"], header.get("cheap_pooling", "none"),
                                 float(header.get("threshold", 0.5)), True)
    else:
        model = rb.RuneModel(header["arch"])
    for g in range(8):
        arr = arrays[f"emb{g}"]
        if arr.dtype.name == "int8":
            arr = arr.astype("float32") * header["scales"][f"emb{g}"]
        model.set_embedding(g, [float(x) for x in arr.reshape(-1)])
    names = list(EXPORT_ORDER[header["arch"]]) if header["arch"] not in ("RUNE-REL-02", "RUNE-04", "RUNE-05") and not header["arch"].startswith("RUNE-03-") else None
    if header["arch"].startswith("RUNE-03-"):
        names = []
        if header.get("pooling", "none") == "per_token":
            for t in range(8):
                names += [f"pool_w{t}", f"pool_b{t}"]
        elif header.get("pooling", "none") == "shared":
            names += ["pool_S"]
            for t in range(8):
                names += [f"pool_s{t}", f"pool_b{t}"]
        if header.get("gate_on", False):
            names += ["gate_a", "gate_b"]
        names += ["w1", "b1", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"]
    if header["arch"] in ("RUNE-04", "RUNE-05"):
        names = []
        if header.get("cheap_pooling", "none") == "shared":
            names += ["pool_S"]
            for t in range(8):
                names += [f"pool_s{t}", f"pool_b{t}"]
        names += ["cw1", "cb1", "cwv", "cbv", "cww", "cbw", "dw", "db",
                  "wq", "bq", "wk", "bk", "wvv", "bvv", "gabS",
                  "w1", "b1", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"]
        if header["arch"] == "RUNE-05":
            names += ["uw", "ub", "sw", "sb"]
    if names is None:
        names = ["wq", "bq", "wk", "bk", "wvv", "bvv", "gabS"]
        if header["geometric_bias"] == "dynamic":
            names += ["dynU", "dynW"]
        names += ["w1", "b1", "w2", "b2", "wvo", "bvo", "wwdl", "bwdl"]
    flat = []
    for n in names:
        flat.extend([float(x) for x in arrays[n].reshape(-1)])
    model.set_arch_tensors(names, flat)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", required=True)
    ap.add_argument("--model-b", required=True)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--tc", default="greedy-1ply")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--engine-version", default="rune-greedy-0.2")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    from training.trainer.system_stats import git_commit, hardware_info

    rng = random.Random(args.seed)
    ma = load_model_for_match(None, args.model_a, args.build_dir)
    mb = load_model_for_match(None, args.model_b, args.build_dir)
    openings = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
        "rnbqkb1r/pp2pppp/5n2/2pp4/3P4/2N5/PPP1PPPP/R1BQKBNR w KQkq - 0 1",
    ]
    score = 0.0
    stats_a = EvalStats()
    stats_b = EvalStats()
    for i in range(args.games):
        if i % 2 == 0:
            res = play_game(ma, mb, stats_a, stats_b, openings[i % len(openings)], rng)
            score += res
        else:
            res = play_game(mb, ma, stats_b, stats_a, openings[i % len(openings)], rng)
            score += 1.0 - res
    elo, err = elo_from_score(score, args.games)
    print(f"score A: {score}/{args.games} elo: {elo:.1f} +/- {err:.1f} (95% CI)")
    if abs(elo) < err:
        print("difference within noise: no conclusion")
    print(f"A evals: {stats_a.evals} refined: {stats_a.refinement_rate():.3f} "
          f"avg_us: {stats_a.avg_latency_us():.2f} mean_u: {stats_a.mean_uncertainty():.3f}")
    print(f"B evals: {stats_b.evals} refined: {stats_b.refinement_rate():.3f} "
          f"avg_us: {stats_b.avg_latency_us():.2f} mean_u: {stats_b.mean_uncertainty():.3f}")
    meta = {
        "engine_version": args.engine_version,
        "network_a": args.model_a,
        "network_b": args.model_b,
        "hardware": hardware_info(),
        "git_commit": git_commit(),
        "time_control": args.tc,
        "threads": args.threads,
        "hash_mb": 0,
        "openings": openings,
        "games": args.games,
        "seed": args.seed,
        "score_a": score,
        "elo_a": elo,
        "elo_err95": err,
        "model_a_evals": stats_a.evals,
        "model_a_refinement_rate": stats_a.refinement_rate(),
        "model_a_avg_latency_us": stats_a.avg_latency_us(),
        "model_a_mean_uncertainty": stats_a.mean_uncertainty(),
        "model_b_evals": stats_b.evals,
        "model_b_refinement_rate": stats_b.refinement_rate(),
        "model_b_avg_latency_us": stats_b.avg_latency_us(),
        "model_b_mean_uncertainty": stats_b.mean_uncertainty(),
    }
    if args.report:
        import json

        with open(args.report, "w") as f:
            json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
