def cp_to_value(cp):
    cp = max(-10000, min(10000, cp))
    return 2.0 / (1.0 + 10.0 ** (-cp / 400.0)) - 1.0


def parse_result_token(tok):
    t = tok.strip()
    if "." in t:
        return {"1.0": 2, "0.5": 1, "0.0": 0}.get(t)
    return {"0": 0, "1": 1, "2": 2}.get(t)


def parse_score_token(tok):
    try:
        return int(tok.strip())
    except ValueError:
        pass
    try:
        return int(round(float(tok.strip())))
    except ValueError:
        return None


def parse_line(line):
    t = line.strip()
    if not t or t.startswith("#"):
        return None
    if "|" in t:
        bar = t.find("|")
        fen = t[:bar].strip()
        rest = [p.strip() for p in t[bar + 1:].split("|")]
        score = parse_score_token(rest[0])
        if score is None:
            return None
        result = parse_result_token(rest[1]) if len(rest) > 1 else None
        if len(rest) > 1 and result is None:
            return None
        return {"fen": fen, "score_cp": score, "result_white": result}
    toks = t.split()
    if len(toks) < 7:
        return None
    fen = " ".join(toks[:6])
    tail = toks[6:]
    if len(tail) == 1:
        score = parse_score_token(tail[0])
        if score is None:
            return None
        return {"fen": fen, "score_cp": score, "result_white": None}
    if len(tail) == 2:
        score = parse_score_token(tail[0])
        result = parse_result_token(tail[1])
        if score is None or result is None:
            return None
        return {"fen": fen, "score_cp": score, "result_white": result}
    return None


def to_teacher(pos, stm_white):
    v = cp_to_value(pos["score_cp"])
    v_stm = v if stm_white else -v
    r = pos["result_white"]
    if r is None:
        w = 1 if abs(v_stm) < 0.15 else (0 if v_stm > 0 else 2)
    else:
        w = (2 - r) if stm_white else r
    return v_stm, w


def stm_of_fen(fen):
    return fen.split()[1] == "w" if len(fen.split()) > 1 else True


def convert_file(path):
    records = []
    skipped = 0
    with open(path) as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            pos = parse_line(t)
            if pos is None:
                skipped += 1
                continue
            stm_white = stm_of_fen(pos["fen"])
            v, w = to_teacher(pos, stm_white)
            records.append({
                "fen": pos["fen"],
                "teacher_v": v,
                "teacher_w": w,
                "teacher_value": v,
                "teacher_wdl": w,
                "teacher_cp": pos["score_cp"],
                "value_perspective": "side_to_move",
                "teacher_provenance": {"source": "plain_import", "file": path},
            })
    return records, skipped
