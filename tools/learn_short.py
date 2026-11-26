"""Fit a contextual policy to sampled team results from legal C++ rollouts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build/junqi.exe"
RULES = ROOT / "config/ruleset.json"


def collect(first, count):
    run = subprocess.run(
        [str(EXE), "learning-probe", str(RULES), str(first), str(count)],
        check=True, capture_output=True, text=True, timeout=90)
    return [json.loads(line) for line in run.stdout.splitlines()]


def encode(case):
    rows = np.array([[a[2], a[3], a[4], a[5] / 11] for a in case["actions"]])
    return np.column_stack((rows, rows[:, 0] * rows[:, 1]))


def probabilities(features, weights):
    logits = features @ weights
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


def update(weights, features, p, selected, score):
    return weights + .015 * (2 * score - 1) * (features[selected] - p @ features)


def evaluate(cases, weights):
    bounds, immediate = [], []
    for case in cases:
        p = probabilities(encode(case), weights)
        outcomes = case["actions"]
        lower = sum(prob * a[6] for prob, a in zip(p, outcomes) if a[6] is not None)
        unknown = sum(prob for prob, a in zip(p, outcomes) if a[6] is None)
        bounds.append((lower, lower + unknown))
        immediate.append(sum(prob for prob, a in zip(p, outcomes) if a[6] == 1 and a[7] == 1))
    return {"score_bounds": np.mean(bounds, axis=0).tolist(),
            "immediate_win_probability": float(np.mean(immediate))}


def train(cases, weights):
    rng = np.random.default_rng(7801)
    encoded = [encode(case) for case in cases]
    for _ in range(60):
        for i in rng.permutation(len(cases)):
            features = encoded[i]
            p = probabilities(features, weights)
            move = rng.choice(len(p), p=p)
            weights = update(weights, features, p, move, cases[i]["actions"][move][6])
    return weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=False)
    training, held_out = collect(20000, 192), collect(30000, 96)
    complete = [c for c in training if all(a[6] is not None for a in c["actions"])]
    if not complete or not held_out:
        raise ValueError("insufficient reachable short tasks")
    print(f"Collected {len(complete)} complete training and {len(held_out)} held-out tasks", flush=True)
    init = np.zeros(5)
    trained = train(complete, init.copy())
    if not np.isfinite(trained).all() or np.array_equal(init, trained):
        raise ValueError("invalid learned weights")
    delayed = next(((c, i) for c in complete + held_out for i, a in enumerate(c["actions"])
                    if a[6] == 1 and a[7] > 1 and a[8] and
                    np.any(encode(c)[i] != encode(c).mean(axis=0))), None)
    if delayed:
        case, i = delayed
        x = encode(case)
        p = probabilities(x, init)
        if np.array_equal(update(init, x, p, i, case["actions"][i][6]), init):
            raise ValueError("delayed team win did not contribute to update")
    for name, weights in (("initial", init), ("trained", trained)):
        np.save(args.out / f"{name}.npy", weights, allow_pickle=False)
        restored = np.load(args.out / f"{name}.npy", allow_pickle=False)
        for case in held_out:
            x = encode(case)
            if not np.array_equal(probabilities(x, weights), probabilities(x, restored)):
                raise ValueError("reloaded output differs")
    result = dict(train_range=[20000, 20191], held_out_range=[30000, 30095],
                  training_cases=len(training), complete_cases=len(complete),
                  held_out_cases=len(held_out), delayed_eliminated_win=bool(delayed),
                  before=evaluate(held_out, init), after=evaluate(held_out, trained),
                  seconds=round(time.monotonic() - started, 3),
                  features=["occupied", "hq", "visible_flag", "own_rank_scaled", "occupied_hq"],
                  hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [RULES, EXE, *args.out.glob("*.npy")]})
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
