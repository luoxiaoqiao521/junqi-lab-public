"""Train both encodings, then compare them on disjoint development layouts."""
import argparse
from collections import Counter
import json
from pathlib import Path

from compare_generations import call, sha, WEIGHT_HASHES
from evaluate_transfer import compare, totals
from learn_short import EXE, RULES, np


def run(start, position, folder):
    mode = "train-position" if position else "train-openings"
    training = json.loads(call(mode, start, "65000"))
    weights = np.array(training["weights"])
    if weights.shape != ((6,) if position else (5,)) or not np.isfinite(weights).all():
        raise ValueError("invalid trained weights")
    if position and (weights[-1] == 0 or not training.get("position_fixture")):
        raise ValueError("unverified position feature")
    path = folder / ("position.npy" if position else "control.npy")
    np.save(path, weights, allow_pickle=False)
    restored = np.load(path, allow_pickle=False)
    if not np.array_equal(weights, restored):
        raise ValueError("weight reload differs")
    raw = call("transfer-probe", restored, "66000", "128", "position" if position else "short", "0")
    rows = [json.loads(line) for line in raw.splitlines()]
    if [(r["seed"], r["seat"]) for r in rows] != [(s, v) for s in range(66000, 66128) for v in range(4)]:
        raise ValueError("development blocks mismatch")
    return rows, training, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if sha(args.parent) != WEIGHT_HASHES[0]:
        raise ValueError("parent hash mismatch")
    args.out.mkdir(parents=True, exist_ok=False)
    parent = np.load(args.parent, allow_pickle=False)
    old, old_train, old_path = run(parent, False, args.out)
    new, new_train, new_path = run(parent, True, args.out)
    labels = {1: "win", .5: "draw", 0: "loss", None: "truncated"}
    transitions = Counter(labels[a["score"]] + "_to_" + labels[b["score"]]
                          for a, b in zip(old, new))
    result = dict(control=totals(old), position=totals(new), comparison=compare(old, new),
                  transitions=dict(transitions), training={"control": old_train, "position": new_train},
                  hashes={p.name: sha(p) for p in (EXE, RULES, args.parent, old_path, new_path)},
                  rows={"control": old, "position": new})
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
