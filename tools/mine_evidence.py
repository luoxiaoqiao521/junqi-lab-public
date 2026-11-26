"""Audit deployment and movement exclusions on frozen natural decisions."""
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time

from camp_control import collect
from learn_short import EXE, RULES
from partner_crossplay import model_roster, sha

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/"build/public-mine-evidence-v1"
REPORT = ROOT/"docs/REPORTS/public-mine-evidence.json"
START, COUNT = 288000, 64
VARIANTS = (0, 2)
ATTACKERS = ("engineer", "ordinary", "bomb")


def source_keys():
    return [(seed, (seed-START)%4, variant) for seed in range(START, START+COUNT) for variant in VARIANTS]


def summarize(rows):
    if [(row["seed"], row["seat"], row["variant"]) for row in rows] != source_keys():
        raise ValueError("unexpected mine test layout blocks")
    formations = {}
    cases = Counter()
    contacts = {attacker: Counter() for attacker in ATTACKERS}
    for row in rows:
        if (row["score"] not in (-1, 0, 1, 2) or
                not 0 <= row["changed_decisions"] <= row["decisions"] or
                len(row["formations"]) != 4 or any(x not in range(4) for x in row["formations"])):
            raise ValueError("invalid decision counts or formation bucket")
        if formations.setdefault(row["seed"], row["formations"]) != row["formations"]:
            raise ValueError("source policies use different layouts for the same seed")
        cases["decisions"] += row["decisions"]
        cases["changed_decisions"] += row["changed_decisions"]
        cases["layouts_with_changed_decisions", row["seed"]] += row["changed_decisions"] > 0
        for attacker, counts in zip(ATTACKERS, row["contacts"], strict=True):
            total, possible, moved, changed = counts
            if not (0 <= changed <= moved <= total and 0 <= possible <= total - moved):
                raise ValueError("contradictory public evidence mask statistics")
            contacts[attacker].update(total=total, possible=possible, moved=moved, support_changed=changed)
    return dict(decisions=cases["decisions"], changed_decisions=cases["changed_decisions"],
                layouts_with_changed_decisions=sum(cases["layouts_with_changed_decisions", s] > 0 for s in formations),
                formations=dict(Counter(str(v) for group in formations.values() for v in group)),
                contacts={key: dict(value) for key, value in contacts.items()})


def audit():
    report = json.loads(REPORT.read_text(encoding="utf8"))
    parent = model_roster()["candidate"]
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    if report["frozen"] != dict(engine=sha(engine), rules=sha(rules), parent=parent):
        raise ValueError("frozen mine evidence inputs changed")
    rows = report["sources"]
    if report["summary"] != summarize(rows):
        raise ValueError("mine evidence counts differ from raw rows")
    for row in (rows[0], rows[1], rows[-1]):
        if row != collect(engine, rules, parent, row["seed"], row["seat"], row["variant"],
                          mode="mine-evidence"):
            raise ValueError("fixed mine evidence replay did not reproduce")
    print("PASS full source grid, fixed layouts, compatible rank masks and replay")


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("mine evidence exists; use audit")
    parent = model_roster()["candidate"]
    RUN.mkdir()
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    shutil.copyfile(EXE, engine)
    shutil.copyfile(RULES, rules)
    started = time.monotonic()
    rows = [collect(engine, rules, parent, *key, mode="mine-evidence") for key in source_keys()]
    report = dict(sources=rows, summary=summarize(rows), seconds=time.monotonic()-started,
                  frozen=dict(engine=sha(engine), rules=sha(rules), parent=parent))
    with REPORT.open("x", encoding="utf8") as output:
        output.write(json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Checked public mine masks on {len(rows)} normal-opening source games")


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
