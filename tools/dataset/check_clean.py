import sys
sys.path.insert(0, ".")
from training.datasets.pipeline import clean_pipeline
from training.datasets.pipeline import load_jsonl
from training.datasets.pipeline import split_by_game
pool = load_jsonl(sys.argv[1])
kept, stats = clean_pipeline(pool)
print(stats)
parts = split_by_game(kept, seed=42)
print("train " + str(len(parts["train"])))
print("val " + str(len(parts["val"])))
print("test " + str(len(parts["test"])))

