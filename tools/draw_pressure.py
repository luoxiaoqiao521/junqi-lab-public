"""Replay a fixed case and challenge the target identity and draw pressure."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import time

from compare_generations import sha
from learn_short import EXE, ROOT, RULES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="junqi-pressure-") as tmp:
        replay = Path(tmp) / "67010.jsonl"
        subprocess.run([str(EXE), "simulate", "67010", str(replay), str(RULES),
                        str(ROOT / "config/observation.json"), "heuristic-ns"],
                       check=True, capture_output=True, timeout=20)
        output = subprocess.check_output([str(EXE), "draw-target", str(replay)], text=True, timeout=45)
    worlds, contrast = [], None
    for line in output.splitlines():
        kind, *fields = line.split()
        if kind == "TARGET":
            world, swap_slot, rank = map(int, fields)
            worlds.append(dict(world=world, swap_slot=swap_slot, rank=rank, branches=[], first_steps=[]))
        elif kind == "PRESSURE":
            ply, seat, low, high, source, target, *path = map(int, fields)
            contrast = dict(ply=ply, seat=seat, low=low, high=high, quiet=[source, target],
                            legal_cycle=list(zip(path[::2], path[1::2])), branches=[])
        elif kind == "BRANCH":
            row = list(map(int, fields))
            if sum(row[4:]) != 8:
                raise ValueError("invalid branch sample count")
            (contrast if contrast is not None else worlds[-1])["branches"].append(row)
        elif kind == "FIRST":
            worlds[-1]["first_steps"].append(list(map(int, fields)))
    if not worlds or any(len(w["branches"]) != 4 or len(w["first_steps"]) != 4 for w in worlds):
        raise ValueError("target-world coverage missing")
    if contrast is None or len(contrast["branches"]) != 4:
        raise ValueError("matched pressure contrast missing")
    rank_count = len({w["rank"] for w in worlds})
    result = dict(seed=67010, source_group="synthetic-67010", role="development-diagnostic",
                  eligible_as_training_label=False,
                  worlds=worlds, distinct_target_ranks=rank_count, pressure=contrast,
                  exe_sha256=sha(EXE), rules_sha256=sha(RULES), seconds=time.monotonic()-started)
    with args.out.open("x", encoding="utf8") as out:
        json.dump(result, out, indent=2)
        out.write("\n")
    print(f"Matched pressure contrast verified; {len(worlds)} layouts, {rank_count} target identities")


if __name__ == "__main__":
    main()
