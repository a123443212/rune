import sys
import json
import os
out = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 200
boards = []
boards = boards + ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"]
boards = boards + ["rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR"]
boards = boards + ["rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR"]
boards = boards + ["r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R"]
sides = []
sides = sides + ["w"]
sides = sides + ["b"]
castles = []
castles = castles + ["KQkq"]
castles = castles + ["Kk"]
castles = castles + ["Qq"]
castles = castles + ["-"]
eps = []
eps = eps + ["-"]
eps = eps + ["e3"]
eps = eps + ["e6"]
eps = eps + ["d6"]
halves = []
halves = halves + ["0"]
halves = halves + ["10"]
halves = halves + ["20"]
halves = halves + ["30"]
rows = []
i = 0
stop = False
while not stop:
    b = boards[i % 4]
    s = sides[(i // 4) % 2]
    c = castles[(i // 8) % 4]
    e = eps[(i // 32) % 4]
    h = halves[(i // 128) % 4]
    full = i + 1
    fen = b + " " + s + " " + c + " " + e + " " + h + " " + str(full)
    v = 0.1 if (i % 2 == 0) else -0.1
    w = 1 if (i % 2 == 0) else 0
    g = i // 3
    rows = rows + [{"fen": fen, "value": v, "wdl": w, "game_id": "smoke" + str(g), "ply": 8}]
    i = i + 1
    stop = i >= top
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
dst = open(out, "w")
for r in rows:
    dst.write(json.dumps(r) + "\n")
print(str(len(rows)) + " " + out)



