import math


def score_to_elo(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -400.0 * math.log10(1.0 / p - 1.0)


def elo_to_score(elo):
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def sprt_llr(wins, losses, draws, elo0=0.0, elo1=70.0):
    p0 = elo_to_score(elo0)
    p1 = elo_to_score(elo1)
    n = wins + losses + draws
    if n == 0:
        return 0.0
    llr = 0.0
    if wins > 0:
        llr += wins * math.log(max(p1, 1e-9) / max(p0, 1e-9))
    if losses > 0:
        llr += losses * math.log(max(1 - p1, 1e-9) / max(1 - p0, 1e-9))
    if draws > 0:
        q0 = 1.0 - abs(2 * p0 - 1.0)
        q1 = 1.0 - abs(2 * p1 - 1.0)
        llr += draws * math.log(max(q1, 1e-9) / max(q0, 1e-9))
    return llr


def sprt_bounds(alpha=0.05, beta=0.05):
    upper = math.log((1 - beta) / max(alpha, 1e-9))
    lower = math.log(max(beta, 1e-9) / (1 - max(alpha, 1e-9)))
    return lower, upper


def sprt_decide(llr, lower, upper):
    if llr >= upper:
        return "accept"
    if llr <= lower:
        return "reject"
    return "continue"
