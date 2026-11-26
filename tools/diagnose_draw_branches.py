"""Compare actions from the same validated late-game replay prefix."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

from learn_short import EXE, ROOT, RULES


def probe(seed, temp):
    replay = temp / f"{seed}.jsonl"
    subprocess.run([str(EXE), "simulate", str(seed), str(replay), str(RULES),
                    str(ROOT / "config/observation.json"), "heuristic-ns"],
                   capture_output=True, check=True, timeout=20)
    run = subprocess.run([str(EXE), "draw-branches", str(replay)],
                         text=True, capture_output=True, check=True, timeout=20)
    lines = run.stdout.splitlines()
    if len(lines) == 1:
        return None
    _, ply, seat, pressure, worlds, source, target = lines[1].split()
    quiet = (int(source), int(target))
    worlds = int(worlds)
    outcomes = defaultdict(list)
    for line in lines[2:]:
        tag, *values = line.split()
        world, profile, source, target, win, draw, loss, trunc = map(int, values)
        if tag != "BRANCH" or win + draw + loss + trunc != 8:
            raise ValueError("invalid branch response")
        outcomes[source, target].append([world, profile, win, draw, loss, trunc])
    pairs = {(world, profile) for world in range(worlds) for profile in range(2)}
    if quiet not in outcomes or any(len(rows) != len(pairs) or
            {(r[0], r[1]) for r in rows} != pairs for rows in outcomes.values()):
        raise ValueError("world or profile coverage mismatch")
    if any(row[2:] != [0, 8, 0, 0] for row in outcomes[quiet]):
        raise ValueError("quiet branch no longer guarantees a draw")
    candidates = []
    for move, rows in outcomes.items():
        if move == quiet:
            continue
        bounds = [[(win+.5*draw)/8, (win+.5*draw+trunc)/8]
                  for _, _, win, draw, _, trunc in rows]
        if worlds < 2:
            label = "single_world"
        elif all(lo > .5 for lo, _ in bounds):
            label = "above_in_all_tested_cells"
        elif all(hi <= .5 for _, hi in bounds):
            label = "no_gain_in_tested_cells"
        else:
            label = "sensitive_or_inconclusive"
        candidates.append(dict(move=move, cells=rows, empirical_bounds=bounds, label=label))
    return dict(seed=seed, ply=int(ply), seat=int(seat), no_progress=int(pressure),
                worlds=worlds, quiet=quiet, candidates=candidates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="junqi-draw-") as temp:
        cases = [probe(seed, Path(temp)) for seed in range(67000, 67016)]
    cases = [case for case in cases if case is not None]
    counts = Counter(c["label"] for case in cases for c in case["candidates"])
    result = dict(seeds=[67000, 67015], exe_sha256=hashlib.sha256(EXE.read_bytes()).hexdigest(),
                  rules_sha256=hashlib.sha256(RULES.read_bytes()).hexdigest(),
                  cases=cases, counts=counts, seconds=time.monotonic()-start)
    with args.out.open("x", encoding="utf8") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(f"Draw branch diagnosis complete: {len(cases)} verified draw prefixes, {dict(counts)}")


if __name__ == "__main__":
    main()
