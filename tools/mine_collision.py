"""Audit an isolated public collision exclusion on normal opening trajectories."""
import json
from pathlib import Path
import shutil
import sys
import time

from camp_control import collect
from flag_interception import source_keys
from learn_short import EXE, RULES
from partner_crossplay import model_roster, sha

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/"build/public-mine-collision-evidence-v1"
REPORT = ROOT/"docs/REPORTS/public-mine-collision-evidence.json"
START, COUNT = 291000, 96
COUNTS = ("decisions", "exclusions", "opportunities", "attack_turns", "changed_turns")


def summarize(rows):
    if [(r["seed"], r["seat"], r["variant"]) for r in rows] != source_keys(START, COUNT):
        raise ValueError("mine-collision source layout grid differs")
    for row in rows:
        if (not 0 <= row["changed_turns"] <= row["attack_turns"] <= row["decisions"] or
                row["attack_turns"] > row["opportunities"] or row["exclusions"] < 0):
            raise ValueError("invalid exclusion or follow-up coverage")
    return dict(**{key: sum(r[key] for r in rows) for key in COUNTS},
                layouts_with_exclusion=len({r["seed"] for r in rows if r["exclusions"]}),
                layouts_with_changes=len({r["seed"] for r in rows if r["changed_turns"]}))


def audit():
    saved = json.loads(REPORT.read_text(encoding="utf8"))
    parent = model_roster()["candidate"]
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    rows = saved["rows"]
    if (saved["frozen"] != dict(engine=sha(engine), rules=sha(rules), parent=parent) or
            saved["summary"] != summarize(rows)):
        raise ValueError("mine collision inputs or results changed")
    witnessed = next((r for r in rows if r["exclusions"]), rows[0])
    for row in (witnessed, rows[-1]):
        if collect(engine, rules, parent, row["seed"], row["seat"], row["variant"],
                   mode="mine-collision-evidence") != row:
            raise ValueError("public collision replay differed")
    print("PASS isolated physical rule, unknown attacker, replacement and natural replays")


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("mine collision result exists; use audit")
    parent = model_roster()["candidate"]
    RUN.mkdir()
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    shutil.copyfile(EXE, engine)
    shutil.copyfile(RULES, rules)
    started = time.monotonic()
    rows = [collect(engine, rules, parent, *key, mode="mine-collision-evidence")
            for key in source_keys(START, COUNT)]
    report = dict(rows=rows, summary=summarize(rows), seconds=time.monotonic()-started,
                  frozen=dict(engine=sha(engine), rules=sha(rules), parent=parent))
    with REPORT.open("x", encoding="utf8") as output:
        output.write(json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Audited public mine-collision evidence across {len(rows)} sources")


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
