"""Train on normal openings and compare two frozen policies on development layouts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from evaluate_transfer import compare, run, totals
from learn_short import EXE, RULES, np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    initial = np.load(args.initial, allow_pickle=False)
    if initial.shape != (5,) or not np.isfinite(initial).all():
        raise ValueError("invalid initial model")
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    command = [str(EXE), "train-openings", str(RULES), *map(str, initial)]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
    train_seconds = time.monotonic() - start
    training = json.loads(result.stdout)
    weights = np.asarray(training["weights"])
    if (weights.shape != initial.shape or not np.isfinite(weights).all() or
            np.array_equal(weights, initial)):
        raise ValueError("opening training did not produce a finite update")
    candidate = args.out / "trained.npy"
    np.save(candidate, weights, allow_pickle=False)
    if not np.array_equal(np.load(candidate, allow_pickle=False), weights):
        raise ValueError("reloaded weights differ")
    before, a = run(args.initial)
    after, b = run(candidate)
    report = dict(training=training, train_seconds=train_seconds, evaluation_seconds=[a,b],
                  initial=totals(before), trained=totals(after), comparison=compare(before,after),
                  hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [EXE, RULES, args.initial, candidate]},
                  rows={"initial": before, "trained": after})
    (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
