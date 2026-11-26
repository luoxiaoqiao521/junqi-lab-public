"""Frozen parent and children crossed with two partners on matched normal openings."""
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time

from evidence_model import load, sha
from evaluate_transfer import compare, totals
from feedback_training import RUN as GENERATIONS, frozen_roster
from learn_short import EXE, RULES
from partner_crossplay import COUNT, evaluate, interaction

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/"build/frozen-partner-credit-check-v1"
REPORT = ROOT/"docs/REPORTS/frozen-partner-credit-check.json"
START = 295000
PARTNERS = ("random", "independent")
ENEMIES = ("contact", "random")
CHILDREN = ("400000-0", "400000-4", "410000-0", "410000-4")
LEARNERS = ("candidate", *CHILDREN)
CELLS = [(learner, partner) for learner in LEARNERS for partner in PARTNERS]


def roster():
    profiles = frozen_roster()
    engine, rules = GENERATIONS/"junqi.exe", GENERATIONS/"ruleset.json"
    if sha(rules) != sha(RULES):
        raise ValueError("current rules differ from the training archive")
    for learner in CHILDREN:
        artifact = GENERATIONS/f"{learner}.json"
        model = load(artifact, engine, rules, "short-v1")
        if model["parent_sha256"] != profiles["candidate"]["artifact_sha256"]:
            raise ValueError("child does not descend from the frozen parent")
        profiles[learner] = dict(policy="short", weights=model["weights"], artifact_sha256=sha(artifact))
    return {name: profiles[name] for name in (*LEARNERS, *PARTNERS, *ENEMIES)}


def stable(rows):
    return [{key: value for key, value in row.items() if key != "learner_ns"} for row in rows]


def play(engine, rules, profiles, learner, partner):
    return stable(evaluate(engine, rules, profiles, learner, partner, ENEMIES[0], START, COUNT,
                           later_enemy=ENEMIES[1]))


def contrast(before, after):
    transitions = Counter(f"{a['score']}->{b['score']}" for a, b in zip(before, after, strict=True))
    return dict(score=compare(before, after), transitions=dict(transitions))


def summarize(rows):
    expected_keys = [(seed, seat) for seed in range(START, START+COUNT) for seat in range(4)]
    if list(rows) != [f"{learner}/{partner}" for learner, partner in CELLS]:
        raise ValueError("incomplete frozen partner grid")
    for block in rows.values():
        if ([(r["seed"], r["seat"]) for r in block] != expected_keys or
                any(r["score"] not in (0, .5, 1, None) for r in block)):
            raise ValueError("unexpected frozen cross-play rows")
    parent = [rows[f"candidate/{name}"] for name in PARTNERS]
    candidates = {}
    for learner in CHILDREN:
        child = [rows[f"{learner}/{name}"] for name in PARTNERS]
        candidates[learner] = dict(
            child_minus_parent={name: contrast(a, b) for name, a, b in zip(PARTNERS, parent, child, strict=True)},
            new_partner_minus_random=contrast(*child),
            interaction=interaction(parent[1], child[1], parent[0], child[0]))
    return dict(totals={key: totals(block) for key, block in rows.items()},
                parent_partner_effect=contrast(*parent), children=candidates)


def audit():
    data = json.loads(REPORT.read_text(encoding="utf8"))
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    profiles = roster()
    if data["frozen"] != dict(engine=sha(engine), rules=sha(rules), profiles=profiles, enemies=list(ENEMIES)):
        raise ValueError("frozen inputs changed")
    rows = data["rows"]
    if data["summary"] != summarize(rows):
        raise ValueError("paired score results changed")
    for learner, partner in (CELLS[0], CELLS[-1]):
        if play(engine, rules, profiles, learner, partner) != rows[f"{learner}/{partner}"]:
            raise ValueError("frozen cross-play replay changed")
    print("PASS all frozen cross-play cells, ancestry, paired score bounds and fixed replay")


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("frozen cross-play archive exists; use audit")
    profiles = roster()
    RUN.mkdir()
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    shutil.copyfile(EXE, engine)
    shutil.copyfile(RULES, rules)
    started = time.monotonic()
    rows = {f"{learner}/{partner}": play(engine, rules, profiles, learner, partner)
            for learner, partner in CELLS}
    data = dict(rows=rows, summary=summarize(rows), seconds=time.monotonic()-started,
                frozen=dict(engine=sha(engine), rules=sha(rules), profiles=profiles, enemies=list(ENEMIES)))
    content = (json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(content) > 1024**2:
        raise ValueError("frozen partner result exceeded its byte cap")
    with REPORT.open("xb") as output:
        output.write(content)
    print("Completed frozen single-seat cross-play on 64 independent layout blocks")


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
