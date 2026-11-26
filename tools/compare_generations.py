"""One frozen parent-child comparison, with a same-budget fresh-start control."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from evaluate_transfer import compare, totals
from learn_short import EXE, RULES, np

WEIGHT_HASHES = (
    "43e055a01cb41c7ec3c19cc969ebe3fe6adec9e39a103417333fd9aa3fe7e624",
    "f7caabe6b9fbc594e0dc431279d21129c38a85327d4c7469c594fb84c2b87fb3",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def call(mode, weights, *args):
    return subprocess.run([str(EXE), mode, str(RULES), *map(str, weights), *args],
                          check=True, capture_output=True, text=True, timeout=120).stdout


def train(start, target):
    report = json.loads(call("train-openings", start, "63000"))
    weights = np.asarray(report["weights"])
    if weights.shape != (5,) or not np.isfinite(weights).all() or np.array_equal(start, weights):
        raise ValueError("training failed to update finite weights")
    np.save(target, weights, allow_pickle=False)
    loaded = np.load(target, allow_pickle=False)
    if not np.array_equal(loaded, weights):
        raise ValueError("reloaded weights differ")
    return loaded, report


def evaluate(weights):
    raw = call("transfer-probe", weights, "64000", "128", "short", "0")
    rows = [json.loads(line) for line in raw.splitlines()]
    if [(r["seed"], r["seat"]) for r in rows] != [(s, v) for s in range(64000, 64128) for v in range(4)]:
        raise ValueError("holdout layout mismatch")
    result = totals(rows)
    score = result["wins"] + .5 * result["draws"]
    result["bounds"] = [score/len(rows), (score+result["truncated"])/len(rows)]
    return rows, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (sha(args.parent), sha(args.initial)) != WEIGHT_HASHES:
        raise ValueError("frozen input weight mismatch")
    parent, initial = (np.load(path, allow_pickle=False) for path in (args.parent, args.initial))
    args.out.mkdir(parents=True, exist_ok=False)
    parent_child = args.out / "child.npy"
    fresh_child = args.out / "fresh.npy"
    child, child_training = train(parent, parent_child)
    fresh, fresh_training = train(initial, fresh_child)
    rows, scores = {}, {}
    for name, weights in (("parent", parent), ("child", child), ("fresh", fresh)):
        rows[name], scores[name] = evaluate(weights)
    report = dict(training={"continued": child_training, "fresh": fresh_training},
                  hashes={p.name: sha(p) for p in [EXE, RULES, args.parent, args.initial,
                                                  parent_child, fresh_child]},
                  scores=scores, continued_vs_parent=compare(rows["parent"], rows["child"]),
                  continued_vs_fresh=compare(rows["fresh"], rows["child"]), rows=rows)
    (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))


if __name__ == "__main__":
    main()
