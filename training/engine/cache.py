import hashlib


def board_part(fen):
    parts = fen.split(" ")
    board = parts[0] if len(parts) > 0 else fen
    stm = parts[1] if len(parts) > 1 else "w"
    cast = parts[2] if len(parts) > 2 else "-"
    ep = parts[3] if len(parts) > 3 else "-"
    return "%s %s %s %s" % (board, stm, cast, ep)


def cache_key(fen, model_hash, mode):
    core = board_part(fen)
    raw = "%s|%s|%s" % (core, model_hash, mode)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class EvalCache:
    def __init__(self, model_hash, mode="full", capacity=4096):
        self.model_hash = str(model_hash)
        self.mode = str(mode)
        self.capacity = int(capacity)
        self.store = {}
        self.order = []
        self.hits = 0
        self.misses = 0

    def get(self, fen):
        k = cache_key(fen, self.model_hash, self.mode)
        if k in self.store:
            self.hits += 1
            return self.store[k]
        self.misses += 1
        return None

    def put(self, fen, value):
        k = cache_key(fen, self.model_hash, self.mode)
        if k in self.store:
            self.store[k] = value
            return k
        if len(self.store) >= self.capacity:
            old = self.order.pop(0)
            self.store.pop(old, None)
        self.store[k] = value
        self.order.append(k)
        return k

    def hit_rate(self):
        n = self.hits + self.misses
        return self.hits / n if n else 0.0
