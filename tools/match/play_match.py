import argparse
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def elo_from_score(score, games):
    if games == 0:
        return 0.0, 0.0
    s = min(max(score / games, 1e-6), 1 - 1e-6)
    elo = -400.0 * math.log10(1.0 / s - 1.0)
    se = math.sqrt(s * (1 - s) / games)
    err = 1.96 * se * 400.0 / (s * (1 - s) * math.log(10))
    return elo, err


def greedy_move(board, model, rng, epsilon=0.0):
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
        v, _ = model.eval_fen(board.to_fen())
        board.unmake_move()
        own = v if stm == 0 else -v
        if best_v is None or own > best_v:
            best, best_v = m, own
    return best if best is not None else rng.choice(moves)


def play_game(white_model, black_model, opening_fen, rng):
    import rune_bindings as rb

    b = rb.Board(opening_fen)
    for _ in range(200):
        moves = b.legal_moves()
        if not moves:
            return 0.5
        stm = b.side_to_move()
        model = white_model if stm == 0 else black_model
        m = greedy_move(b, model, rng)
        mv = rb.Move()
        mv.from_sq, mv.to_sq, mv.promo = m[0], m[1], m[2]
        b.make_move(mv)
    v, _ = white_model.eval_fen(b.to_fen())
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
    else:
        model = rb.RuneModel(header["arch"])
    for g in range(8):
        arr = arrays[f"emb{g}"]
        if arr.dtype.name == "int8":
            arr = arr.astype("float32") * header["scales"][f"emb{g}"]
        model.set_embedding(g, [float(x) for x in arr.reshape(-1)])
    names = list(EXPORT_ORDER[header["arch"]]) if header["arch"] != "RUNE-REL-02" else None
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
    for i in range(args.games):
        if i % 2 == 0:
            res = play_game(ma, mb, openings[i % len(openings)], rng)
            score += res
        else:
            res = play_game(mb, ma, openings[i % len(openings)], rng)
            score += 1.0 - res
    elo, err = elo_from_score(score, args.games)
    print(f"score A: {score}/{args.games} elo: {elo:.1f} +/- {err:.1f} (95% CI)")
    if abs(elo) < err:
        print("difference within noise: no conclusion")
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
    }
    if args.report:
        import json

        with open(args.report, "w") as f:
            json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
