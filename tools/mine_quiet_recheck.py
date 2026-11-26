"""Recheck frozen quiet choices and attack alternatives with a new partner policy."""
from fractions import Fraction
import json
import shutil
import sys
import time

from mine_information_value import (
    EXE, REPORT as SOURCE, ROOT, RULES, collect, horizon, model_roster, read, score_delta,
    sha, shared_action,
)

TICKET = "confirm-mine-quiet-alternatives"
RUN = ROOT / "build" / (TICKET + "-v1")
REPORT = ROOT / "docs" / "REPORTS" / (TICKET + ".json")
SAMPLES = 24


def selection():
    rows = [row for row in read(SOURCE)["rows"]["ordinary"]
            if row["case"] and row["case"]["cells"]]
    quiet, controls = [], []
    for row in rows:
        a, b, informed, flip, _ = horizon(row, 0)
        point = row["case"]
        blind = shared_action(a, b, Fraction(1,2))
        attacks = [point["moves"].index([point[role],point["target"],0])
                   for role in ("engineer","ordinary")]
        if flip and informed[0] == informed[1] and point["moves"][informed[0]][2]:
            quiet.append((row, "shared_quiet", informed, blind, attacks))
        elif flip and informed[0] != informed[1]:
            controls.append((row, "different_choices", informed, blind, attacks))
    if [case[0]["seed"] for case in quiet] != [302047,302053,302063]:
        raise ValueError("preregistered quiet cases changed")
    result = list(quiet)
    used = {c[0]["seed"] for c in result}
    for variant in (0,2):
        matches = [c for c in controls if c[0]["variant"] == variant and c[0]["seed"] not in used][:2]
        if len(matches) != 2:
            raise ValueError("missing independent comparison cases")
        used.update(c[0]["seed"] for c in matches)
        result.extend(matches)
    return result


def replay(case, parent):
    source = case[0]
    return collect(RUN / "junqi.exe", RUN / "ruleset.json", parent,
                   source["seed"], source["seat"], source["variant"], mode="mine-quiet-recheck")


def compare(cell, candidate, old):
    deltas = [(0,0) if candidate == old else score_delta(row[candidate],row[old]) for row in cell]
    lows, highs = zip(*deltas, strict=True)
    return dict(low=sum(lows)/SAMPLES, high=sum(highs)/SAMPLES,
                gain=sum(d>0 for d in lows), loss=sum(d<0 for d in highs))


def summarize(case, row):
    source, role, informed, blind, attacks = case
    if any(row[k] != source[k] for k in ("seed","seat","variant","score")):
        raise ValueError("source trajectory changed")
    old, point = source["case"], row["case"]
    if any(point[k] != v for k,v in old.items() if k != "cells"):
        raise ValueError("fixed public first moves changed")
    cells = point["cells"]
    if len(cells)!=6 or any(len(cell)!=SAMPLES for cell in cells) or any(
            len(scores)!=len(point["moves"]) or any(s not in (-1,0,1,2) for s in scores)
            for cell in cells for scores in cell):
        raise ValueError("missing reply coverage or scoring")
    panels = []
    for i, cell in enumerate(cells):
        candidate = informed[i//3]
        panels.append(dict(world="mine" if i<3 else "nonmine", response=i%3,
                           versus_attacks=[compare(cell,candidate,j) for j in attacks],
                           versus_blind=compare(cell,candidate,blind),
                           completed=sum(-1 not in row for row in cell)))
    return dict(seed=row["seed"],seat=row["seat"],variant=row["variant"],role=role,
                candidate=informed,blind=blind,attacks=attacks,moves=point["moves"],panels=panels)


def frozen(parent):
    return dict(engine=sha(RUN/"junqi.exe"), rules=sha(RUN/"ruleset.json"),
                experiment=sha(SOURCE), parent=parent)


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("quiet recheck already exists; use audit")
    cases, parent = selection(), model_roster()["candidate"]
    RUN.mkdir()
    shutil.copyfile(EXE,RUN/"junqi.exe")
    shutil.copyfile(RULES,RUN/"ruleset.json")
    started = time.monotonic()
    rows = [replay(case,parent) for case in cases]
    results = [summarize(case,row) for case,row in zip(cases,rows,strict=True)]
    report = dict(frozen=frozen(parent), rows=rows, results=results, seconds=time.monotonic()-started)
    payload = (json.dumps(report,separators=(",",":"),allow_nan=False)+"\n").encode()
    if len(payload) > 96*1024:
        raise ValueError("quiet recheck exceeds byte cap")
    with REPORT.open("xb") as output:
        output.write(payload)
    print(f"Rechecked {len(cases)} independent source layouts with three frozen continuations")


def audit():
    report = read(REPORT)
    cases, parent = selection(), model_roster()["candidate"]
    if report["frozen"] != frozen(parent) or report["results"] != [
            summarize(case,row) for case,row in zip(cases,report["rows"],strict=True)]:
        raise ValueError("quiet recheck summary does not match")
    for case,row in zip(cases,report["rows"],strict=True):
        if replay(case,parent) != row:
            raise ValueError("quiet recheck did not replay")
    print("PASS frozen first moves, distinct source layouts, three continuations and replay")


if __name__ == "__main__":
    if not sys.argv[1:]:
        run()
    elif sys.argv[1:] == ["audit"]:
        audit()
    else:
        raise SystemExit("usage: mine_quiet_recheck.py [audit]")
