"""Compare L1 weights in the predeclared normal-opening development blocks."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

from learn_short import EXE, RULES, np


def run(weights):
    start = time.monotonic()
    params = np.load(weights, allow_pickle=False)
    if params.shape != (5,) or not np.isfinite(params).all():
        raise ValueError("invalid short policy weights")
    result = subprocess.run(
        [str(EXE), "transfer-probe", str(RULES), *map(str, params)],
        check=True, capture_output=True, text=True, timeout=180)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    keys = [(row["seed"], row["seat"]) for row in rows]
    if keys != [(seed, seat) for seed in range(40000, 40032) for seat in range(4)]:
        raise ValueError("unexpected layout blocks or seats")
    return rows, time.monotonic() - start


def totals(rows):
    count = Counter(r["score"] for r in rows)
    return dict(wins=count[1], draws=count[.5], losses=count[0],
                truncated=count[None], opportunities=sum(r["opportunities"] for r in rows),
                decisions=sum(r["decisions"] for r in rows))


def compare(initial, trained):
    def identity(row):
        return row["seed"], row["seat"], row["first"]

    keys = [identity(row) for row in initial]
    if not keys or keys != [identity(row) for row in trained]:
        raise ValueError("layout, seat or first-seat pairing mismatch")
    seeds = set()
    for offset in range(0, len(keys), 4):
        seed, _, first = keys[offset]
        if seed in seeds or keys[offset:offset+4] != [(seed, seat, first) for seat in range(4)]:
            raise ValueError("incomplete or duplicate layout block")
        seeds.add(seed)
    lo = [(b["score"] or 0) - (a["score"] if a["score"] is not None else 1)
          for a, b in zip(initial, trained)]
    hi = [(b["score"] if b["score"] is not None else 1) - (a["score"] or 0)
          for a, b in zip(initial, trained)]
    block_lo = np.asarray(lo).reshape(-1, 4).mean(axis=1)
    block_hi = np.asarray(hi).reshape(-1, 4).mean(axis=1)
    blocks = np.random.default_rng(40100).integers(0, len(block_lo), (4096, len(block_lo)))
    return dict(difference_bounds=[statistics.mean(lo), statistics.mean(hi)],
                exploratory_block_interval=[float(np.quantile(block_lo[blocks].mean(axis=1), .025)),
                                            float(np.quantile(block_hi[blocks].mean(axis=1), .975))])


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--weights", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    paths = [args.weights / name for name in ("initial.npy", "trained.npy")]
    initial, a = run(paths[0])
    trained, b = run(paths[1])
    result = dict(initial=totals(initial), trained=totals(trained),
                  seconds=[a, b], comparison=compare(initial, trained),
                  hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [EXE, RULES, *paths]},
                  rows={"initial": initial, "trained": trained})
    with args.out.open("x", encoding="utf8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
