"""Bounded offline discovery, with separate nomination and terminal continuations."""
import argparse
from collections import Counter
from itertools import groupby
import json
from pathlib import Path
from statistics import mean, median
import subprocess
import time

from measure_deadline_opening import LABEL, ROOT, RULES, SCORE, build, file_sha, model_roster, require

START, COUNT = 1201700, 12
RUN = ROOT / "build/tactic-discovery/pilot-1"
ARMS = ("frozen", "shortlist", "outside", "random_outside")


def parse_root(lines):
    _, seed, ply, seat, legal, listed, fallback = lines[0].split()
    root = dict(zip(("seed", "ply", "seat", "legal", "listed", "fallback"),
                    map(int, (seed, ply, seat, legal, listed, fallback)), strict=True))
    if len(lines) == 2 and lines[1].startswith("X "):
        root["excluded"] = lines[1][2:]
        return root
    screen, actions, arms, pairs = [], {}, [], []
    for line in lines[1:]:
        kind, *fields = line.split()
        if kind == "W":
            screen.append(int(fields[0]))
        elif kind == "S":
            move, member, value, *ends = fields
            move = int(move)
            require(move not in actions and member in ("0", "1") and sum(map(int, ends)) == 4,
                    "action missing screening budget or repeated")
            actions[move] = dict(listed=int(member), screening=float(value), outcomes=list(map(int, ends)))
        elif kind == "N":
            require(not arms, "first moves already locked")
            arms = list(map(int, fields))
        elif kind == "P":
            world, table, stream, *results = map(int, fields)
            require(arms and len(results) == 2*len(arms), "missing paired outcome")
            ends = list(zip(results[::2], results[1::2], strict=True))
            require(all(score in SCORE and plies >= 1 for score, plies in ends), "invalid outcome")
            pairs.append(dict(world=world, table=table, stream=stream, results=ends))
        else:
            raise ValueError(f"unexpected record: {kind}")
    require(len(actions) == root["legal"] and sum(x["listed"] for x in actions.values()) == root["listed"],
            "incomplete legal/shortlist coverage")
    best = [min((move for move in actions if actions[move]["listed"] == member),
                key=lambda move: (-actions[move]["screening"], move)) for member in (1, 0)]
    require(arms[:3] == [root["fallback"], *best] and actions[arms[0]]["listed"], "first moves differ from screen")
    require(len(arms) == 3 + (root["legal"] - root["listed"] > 1), "random control count")
    if len(arms) == 4:
        require(arms[3] in actions and arms[3] != arms[2] and not actions[arms[3]]["listed"], "bad outside control")
    worlds = {p["world"] for p in pairs}
    require(len(screen) == len(set(screen)) == 4 and len(worlds) == 8 and not worlds.intersection(screen),
            "selection and confirmation worlds overlap or are incomplete")
    require({(p["world"], p["table"]) for p in pairs} == {(w, t) for w in worlds for t in (1, 2)}
            and len(pairs) == len({p["stream"] for p in pairs}) == 16, "confirmation streams are incomplete")
    return {**root, "screen_worlds": screen, "actions": actions, "arms": dict(zip(ARMS, arms)), "pairs": pairs}


def compare(pairs, against):
    lower, upper, speed = [], [], []
    for pair in pairs:
        a, ap = pair["results"][against]
        b, bp = pair["results"][2]
        lower.append((0 if b < 0 else SCORE[b]) - (1 if a < 0 else SCORE[a]))
        upper.append((1 if b < 0 else SCORE[b]) - (0 if a < 0 else SCORE[a]))
        if a == b == 2:
            speed.append(bp - ap)
    return dict(pairs=len(pairs), delta_bounds=[mean(lower), mean(upper)],
                both_win=len(speed), outside_win_plies_delta=median(speed) if speed else None)


def deadlines(tested):
    result = {}
    for index, name in enumerate(ARMS):
        cases = [(r["ply"]+p["results"][index][1], p["results"][index][0])
                 for r in tested if name in r["arms"] for p in r["pairs"]]
        result[name] = {h: dict(Counter(LABEL[outcome] if end <= h else "ongoing"
                                        for end, outcome in cases)) for h in (100, 140)}
    return result


def summarize(roots):
    tested = [root for root in roots if "pairs" in root]
    comparisons = {name: {} for name in ("frozen", "shortlist", "random_outside")}
    repeated_positive, reversals = [], []
    for root in tested:
        tables = [[p for p in root["pairs"] if p["table"] == table] for table in (1, 2)]
        if all(compare(table, against)["delta_bounds"][0] > 0 for table in tables for against in (0, 1)):
            repeated_positive.append([root["seed"], root["ply"], root["arms"]["outside"]])
        deltas = [compare(table, 0)["delta_bounds"] for table in tables]
        if any(lo > 0 for lo, hi in deltas) and any(hi < 0 for lo, hi in deltas):
            reversals.append(dict(seed=root["seed"], ply=root["ply"], deltas=deltas))
    totals = {name: dict(Counter(p["results"][i][0] for root in tested if name in root["arms"]
                                 for p in root["pairs"])) for i, name in enumerate(ARMS)}
    for name, by_table in comparisons.items():
        for table in (1, 2):
            available = [p for root in tested if name in root["arms"] for p in root["pairs"] if p["table"] == table]
            if available:
                by_table[table] = compare(available, ARMS.index(name))
    return dict(roots=len(roots), tested=len(tested), excluded=dict(Counter(r["excluded"] for r in roots
                                                                          if "excluded" in r)),
                screening_trials=sum(len(r["actions"])*4 for r in tested),
                confirmed_branches=sum(len(p["results"]) for r in tested for p in r["pairs"]),
                repeat_checks=2*len(tested), outcomes=totals, comparisons=comparisons,
                deadlines=deadlines(tested), positive_both_tables=repeated_positive, reversals=reversals)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RUN)
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    build(output)
    roster = model_roster()
    models = [roster[name] for name in ("candidate", "historical")]
    exe = output / "deadline_opening.exe"
    payload = " ".join(map(str, (START, COUNT, *(w for model in models for w in model["weights"]))))
    outputs, times = [], []
    for attempt in (1, 2):
        now = time.monotonic()
        run = subprocess.run([exe, RULES, "discover-public-actions"], input=payload, encoding="utf8",
                             capture_output=True, timeout=150, check=True, cwd=ROOT)
        (output / f"run-{attempt}.txt").write_text(run.stdout, encoding="utf8")
        times.append(round(time.monotonic()-now, 2))
        outputs.append(run.stdout)
    require(outputs[0] == outputs[1], "whole run is not reproducible")
    groups = groupby(outputs[0].splitlines(), key=lambda line: line.startswith("R "))
    roots = [parse_root([*header, *next(groups)[1]]) for is_header, header in groups if is_header]
    require([(r["seed"], r["ply"]) for r in roots] ==
            [(seed, ply) for seed in range(START, START+COUNT) for ply in (40, 88)], "missing roots")
    summary = summarize(roots)
    receipt = dict(engine=file_sha(exe), rules=file_sha(RULES), models=models, identical=True,
                   run_seconds=times, total_seconds=round(time.monotonic()-started, 2), summary=summary, roots=roots)
    (output / "results.json").write_text(json.dumps(receipt, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
