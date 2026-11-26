"""Test whether a more contact-seeking frozen enemy exposes team feedback."""
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time

from evaluate_transfer import compare, totals
from learn_short import EXE, RULES
from partner_crossplay import evaluate, model_roster, sha

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "build/feedback-coverage-v1"
REPORT = ROOT / "docs/REPORTS/feedback-coverage.json"
START, COUNT = 284000, 32
PARTNERS = ("random", "independent")
POSITIONS = ("before-partner", "after-partner")
ENEMIES = ("heuristic", "contact")
LEARNERS = ("heuristic", "candidate")
CONTEXTS = [(p, s, e) for p in PARTNERS for s in POSITIONS for e in ENEMIES]
TEAM_SCORES = (0, .5, 1, None)


def cell_summary(cell, expected, first_seats):
    rows = cell["rows"]
    if set(rows) != set(LEARNERS):
        raise ValueError("incomplete learner comparison")
    for group in rows.values():
        if ([(r["seed"], r["seat"]) for r in group] != expected or
                [r["first"] for r in group] != first_seats or
                any(r["score"] not in TEAM_SCORES for r in group)):
            raise ValueError("missing or invalid layout/seat rows")
    old, new = (rows[name] for name in LEARNERS)
    pairs = [(a, b) for a, b in zip(old, new, strict=True)
             if a["score"] is not None and b["score"] is not None]
    discordant = [(a, b) for a, b in pairs if a["score"] != b["score"]]
    transitions = Counter(f'{a["score"]}->{b["score"]}' for a, b in pairs)
    return dict(context=cell["context"], team={name: totals(rows[name]) for name in LEARNERS},
                comparison=compare(old, new), transitions=dict(transitions),
                unequal_pairs=len(discordant),
                unequal_layouts=len({a["seed"] for a, _ in discordant}),
                candidate_loss_layouts=len({r["seed"] for r in new if r["score"] == 0}))


def summarize(cells):
    if [tuple(c["context"]) for c in cells] != CONTEXTS:
        raise ValueError("unexpected or incomplete feedback context grid")
    expected = [(seed, seat) for seed in range(START, START+COUNT) for seat in range(4)]
    first_seats = [r["first"] for r in cells[0]["rows"]["heuristic"]]
    values = [cell_summary(cell, expected, first_seats) for cell in cells]
    for old, new in zip(values[::2], values[1::2], strict=True):
        new["added_unequal_layouts_vs_heuristic"] = new["unequal_layouts"] - old["unequal_layouts"]
    return values


def roster():
    return dict(model_roster(), contact=dict(policy="contact", weights=[0]*5))


def audit():
    report = json.loads(REPORT.read_text(encoding="utf8"))
    if report["frozen"] != dict(engine=sha(RUN/"junqi.exe"), rules=sha(RUN/"ruleset.json")):
        raise ValueError("feedback run artifacts changed")
    if report["roster"] != roster() or report["summary"] != summarize(report["cells"]):
        raise ValueError("feedback coverage report differs from raw rows")
    print("PASS feedback coverage, role/order pairing, artifact identity and all eight contexts")


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("coverage archive exists; use audit")
    profiles = roster()
    RUN.mkdir()
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    shutil.copyfile(EXE, engine)
    shutil.copyfile(RULES, rules)
    started = time.monotonic()
    cells = []
    for partner, position, enemy in CONTEXTS:
        before, after = (enemy, "random") if position == "before-partner" else ("random", enemy)
        rows = {learner: evaluate(engine, rules, profiles, learner, partner, before, START,
                                  COUNT, later_enemy=after) for learner in LEARNERS}
        cells.append(dict(context=[partner, position, enemy], rows=rows))
    report = dict(roster=profiles, cells=cells, summary=summarize(cells),
                  seconds=time.monotonic()-started,
                  frozen=dict(engine=sha(engine), rules=sha(rules)))
    data = (json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(data) > 1024**2:
        raise ValueError("coverage report byte budget exceeded")
    with REPORT.open("xb") as output:
        output.write(data)
    print(f"PASS {len(CONTEXTS)} feedback contexts in {report['seconds']:.2f}s")


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
