import time


class SearchStats:
    def __init__(self):
        self.nodes = 0
        self.evals = 0
        self.refined = 0
        self.cutoffs = 0
        self.tt_hits = 0
        self.root_score = 0.0
        self.root_move = None
        self.seconds = 0.0
        self.per_type = {}

    def note_node(self, ntype):
        self.nodes += 1
        self.per_type[ntype] = self.per_type.get(ntype, 0) + 1

    def nps(self):
        return self.nodes / self.seconds if self.seconds > 0 else 0.0

    def to_dict(self):
        return {"nodes": self.nodes, "evals": self.evals, "refined": self.refined, "cutoffs": self.cutoffs, "tt_hits": self.tt_hits, "root_score": self.root_score, "root_move": self.root_move, "seconds": self.seconds, "nps": self.nps(), "per_type": dict(self.per_type)}


def node_type(depth, ply, alpha, beta, is_root):
    if is_root:
        return "root"
    if (beta - alpha) > 1:
        return "pv"
    if ply % 2 == 0:
        return "cut"
    return "leaf"


class AlphaBeta:
    def __init__(self, evaluator, lazy_cfg=None, tt=None):
        self.ev = evaluator
        self.lazy = lazy_cfg
        self.tt = tt
        self.stats = SearchStats()

    def search(self, board, depth, alpha=-1e9, beta=1e9):
        t0 = time.perf_counter()
        self.stats = SearchStats()
        score, move = self._negamax(board, depth, 0, alpha, beta, True)
        self.stats.seconds = time.perf_counter() - t0
        self.stats.root_score = score
        self.stats.root_move = move
        return score, move, self.stats

    def _eval_leaf(self, board, alpha, beta, ntype):
        self.stats.note_node(ntype)
        self.stats.evals += 1
        return self.ev.evaluate(board)

    def _negamax(self, board, depth, ply, alpha, beta, is_root):
        ntype = node_type(depth, ply, alpha, beta, is_root)
        if depth <= 0:
            v = self._eval_leaf(board, alpha, beta, ntype)
            return v, None
        moves = board.legal_moves()
        if not moves:
            v = self._eval_leaf(board, alpha, beta, ntype)
            return v, None
        ordered = self.ev.order_moves(board, moves)
        best = None
        best_v = None
        for m in ordered:
            if not board.make_move(m):
                continue
            v, _ = self._negamax(board, depth - 1, ply + 1, -beta, -alpha, False)
            v = -v
            board.unmake_move()
            self.stats.note_node(ntype)
            if best_v is None or v > best_v:
                best_v = v
                best = m
                if is_root:
                    self.stats.root_move = m
            if v > alpha:
                alpha = v
            if alpha >= beta:
                self.stats.cutoffs += 1
                break
        if best_v is None:
            best_v = self._eval_leaf(board, alpha, beta, ntype)
        return best_v, best
