import math
import random


class MctsNode:
    def __init__(self, prior=0.0):
        self.visits = 0
        self.total = 0.0
        self.prior = prior
        self.children = {}
        self.value = 0.0
        self.expanded = False


class PuctSearch:
    def __init__(self, simulations=800, cpuct=1.25, dirichlet_eps=0.25, dirichlet_alpha=0.3, fpu=0.0, seed=1):
        self.simulations = simulations
        self.cpuct = cpuct
        self.dirichlet_eps = dirichlet_eps
        self.dirichlet_alpha = dirichlet_alpha
        self.fpu = fpu
        self.rng = random.Random(seed)

    def search(self, root_state, legal_fn, apply_fn, eval_fn):
        root_moves = legal_fn(root_state)
        if not root_moves:
            return None, {"simulations": 0, "root_visits": 0}
        v0, p0 = eval_fn(root_state)
        if len(p0) != len(root_moves):
            p0 = [1.0 / len(root_moves)] * len(root_moves)
        priors = list(p0)
        if self.dirichlet_eps > 0:
            noise = [self.rng.gammavariate(self.dirichlet_alpha, 1.0) for _ in root_moves]
            s = sum(noise) or 1.0
            noise = [x / s for x in noise]
            priors = [(1 - self.dirichlet_eps) * p + self.dirichlet_eps * n for p, n in zip(priors, noise)]
        nodes = {0: MctsNode()}
        states = {0: root_state}
        values = {0: v0}
        expanded = {0: True}
        next_id = 1
        child_map = {0: []}
        for i, m in enumerate(root_moves):
            nid = next_id
            next_id += 1
            nodes[nid] = MctsNode(priors[i])
            states[nid] = apply_fn(root_state, m)
            values[nid] = 0.0
            expanded[nid] = False
            child_map[0].append((nid, m))
        child_map.update({nid: [] for nid in list(nodes) if nid != 0})
        for _ in range(self.simulations):
            idx = 0
            path = [0]
            while expanded.get(idx, False) and child_map.get(idx):
                kids = child_map[idx]
                tot = sum(nodes[c].visits for c, _ in kids)
                sq = math.sqrt(tot)
                best = None
                best_score = -1e30
                for c, _ in kids:
                    ch = nodes[c]
                    q = ch.total / ch.visits if ch.visits else values[idx] + self.fpu
                    u = self.cpuct * ch.prior * sq / (1 + ch.visits)
                    s = q + u
                    if s > best_score:
                        best_score = s
                        best = c
                idx = best
                path.append(idx)
            st = states[idx]
            moves = legal_fn(st)
            if not moves:
                leaf = 0.0
                expanded[idx] = True
                values[idx] = leaf
            elif not expanded.get(idx, False):
                vv, pp = eval_fn(st)
                values[idx] = vv
                expanded[idx] = True
                if len(pp) != len(moves):
                    pp = [1.0 / len(moves)] * len(moves)
                for mi, m in enumerate(moves):
                    nid = next_id
                    next_id += 1
                    nodes[nid] = MctsNode(pp[mi])
                    states[nid] = apply_fn(st, m)
                    values[nid] = 0.0
                    expanded[nid] = False
                    child_map[idx].append((nid, m))
                    child_map[nid] = []
                leaf = vv
            else:
                leaf = values[idx]
            val = leaf
            for pi in reversed(path):
                nodes[pi].visits += 1
                nodes[pi].total += val
                val = -val
        kids = child_map[0]
        best_m = None
        best_v = -1
        for c, m in kids:
            if nodes[c].visits > best_v:
                best_v = nodes[c].visits
                best_m = m
        total = sum(nodes[c].visits for c, _ in kids)
        return best_m, {"simulations": self.simulations, "root_visits": total, "best_visit": best_v}
