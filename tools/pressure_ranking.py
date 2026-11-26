"""Matched late-game comparison across legal histories and compatible worlds."""
import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path
import subprocess
import tempfile
import time

from compare_generations import sha
from learn_short import EXE, ROOT, RULES


def bound(row):
    wins, draws, _, unknown = row
    return (wins + .5 * draws) / 8, (wins + .5 * draws + unknown) / 8


def sign(a, b):
    lo_a, hi_a = bound(a)
    lo_b, hi_b = bound(b)
    return 1 if lo_a > hi_b else -1 if hi_a < lo_b else 0


def analyze(output, seed):
    lines = output.splitlines()
    if len(lines) == 1:
        return None
    tag, *header = lines[1].split()
    end, world_count = lines[-1].split()
    if tag != "PRESSURE" or end != "WORLDS":
        raise ValueError("unexpected pressure result")
    worlds = int(world_count)
    cells, actions, first_steps, hqs, moved = {}, {}, {}, {}, {}
    for line in lines[2:-1]:
        tag, *args = line.split()
        if tag == "MOVED":
            source, target, seen = map(int, args)
            moved[source,target] = bool(seen)
            continue
        if tag == "HQ":
            source, target, hq = map(int, args)
            hqs[source,target] = bool(hq)
            continue
        if tag == "ACTION":
            source, target, occupied = map(int, args)
            actions[source,target] = occupied
            continue
        if tag == "FIRST":
            cell, profile, source, target, result, samples = map(int, args)
            key = (cell, profile, (source,target))
            if key in first_steps or result not in range(5) or samples != (0 if result else 8):
                raise ValueError("invalid first-step evidence")
            first_steps[key] = (result,samples)
            continue
        cell, profile, source, target, *counts = map(int, args)
        key = (cell, profile, (source, target))
        if tag != "BRANCH" or sum(counts) != 8 or key in cells:
            raise ValueError("invalid rollout cell")
        cells[key] = counts
    moves = sorted({key[2] for key in cells})
    if (set(moves) != set(actions) or set(cells) != set(first_steps) or
            set(cells) != set(product(range(worlds * 2), range(2), moves))):
        raise ValueError("incomplete paired action grid")
    quiet = [m for m in moves if actions[m] == 0]
    attack = [m for m in moves if actions[m] == 1]
    ranks = []
    for q,a in product(quiet,attack):
        directions = [(sign(cells[2*w,p,a],cells[2*w,p,q]),
                       sign(cells[2*w+1,p,a],cells[2*w+1,p,q]))
                      for w,p in product(range(worlds),range(2))]
        label = next((f"{lo:+d}_to_{hi:+d}" for lo,hi in [(-1,1),(1,-1)]
                      if worlds >= 2 and all(d == (lo,hi) for d in directions)), None)
        comparison = dict(quiet=q, attack=a, directions=directions, consistent_reversal=label)
        if a in hqs:
            comparison["attack_hq"] = hqs[a]
        if a in moved:
            comparison["target_moved"] = moved[a]
        ranks.append(comparison)
    return dict(seed=seed, source_group=f"synthetic-{seed}", worlds=worlds,
                pressure=list(map(int,header)), comparisons=ranks,
                first_steps=[[cell,p,*move,*evidence] for (cell,p,move),evidence in first_steps.items()],
                cells=[[cell,p,*move,*counts] for (cell,p,move),counts in cells.items()])


def probe(seed, folder, mode="pressure-ranking"):
    log = folder / f"{seed}.jsonl"
    subprocess.run([str(EXE), "simulate", str(seed), str(log), str(RULES),
                    str(ROOT / "config/observation.json"), "heuristic-ns"],
                   capture_output=True, check=True, timeout=20)
    result = subprocess.check_output([str(EXE), mode, str(log)], text=True, timeout=45)
    return analyze(result, seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="junqi-rank-") as temp:
        cases = [probe(seed, Path(temp)) for seed in range(69000,69016)]
    cases = [case for case in cases if case is not None]
    directions = Counter(direction for case in cases for direction in {
        row["consistent_reversal"] for row in case["comparisons"]} if direction)
    result = dict(seeds=[69000,69015], cases=cases, reversal_prefixes=directions,
                  role="development-diagnostic", eligible_as_training_label=False,
                  executed_rollouts=sum(row[-1] for c in cases for row in c["first_steps"]),
                  eligible_for_feature=any(count>=2 for count in directions.values()),
                  exe_sha256=sha(EXE), rules_sha256=sha(RULES), seconds=time.monotonic()-started)
    with args.out.open("x", encoding="utf8") as out:
        json.dump(result,out,indent=2)
        out.write("\n")
    print(f"Pressure ranking: {len(cases)} matched prefixes; gate={result['eligible_for_feature']}")


if __name__ == "__main__":
    main()
