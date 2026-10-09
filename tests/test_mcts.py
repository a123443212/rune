def test_mcts_picks_winning_move():
    from training.search.mcts import PuctSearch
    def legal(s):
        return ["a", "b"] if s == "root" else []
    def apply(s, m):
        return "win" if m == "a" else "loss"
    def ev(s):
        if s == "root":
            return 0.0, [0.6, 0.4]
        if s == "win":
            return 1.0, []
        return -1.0, []
    search = PuctSearch(simulations=200, dirichlet_eps=0.0, seed=1)
    m, stats = search.search("root", legal, apply, ev)
    assert m == "a"
    assert stats["root_visits"] == 200


def test_mcts_empty_legal():
    from training.search.mcts import PuctSearch
    s = PuctSearch(simulations=10, seed=1)
    m, stats = s.search("x", lambda _: [], lambda a, b: b, lambda _: (0.0, []))
    assert m is None


def test_mcts_deterministic():
    from training.search.mcts import PuctSearch
    def legal(s):
        return ["a", "b", "c"]
    def apply(s, m):
        return s + m
    def ev(s):
        n = len(s)
        return 0.1 * (n % 3 - 1), [0.5, 0.3, 0.2]
    a, _ = PuctSearch(simulations=100, seed=7).search("r", legal, apply, ev)
    b, _ = PuctSearch(simulations=100, seed=7).search("r", legal, apply, ev)
    assert a == b
