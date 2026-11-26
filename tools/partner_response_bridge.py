"""Test informative opponent mixes and the order of opponent responses."""
import json
from pathlib import Path
import shutil
import sys
import time

from evaluate_transfer import compare, totals
from learn_short import EXE, RULES
from partner_crossplay import evaluate, model_roster, sha

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "build/partner-response-bridge-v1"
REPORT = ROOT / "docs/REPORTS/partner-response-bridge.json"
START, COUNT = 282000, 48
OPPONENTS = (("random", "heuristic"), ("heuristic", "random"),
             ("random", "learned_enemy"), ("learned_enemy", "random"))
PARTNERS = ("random", "independent")
LEARNERS = ("candidate", "heuristic")


def summarize(cells):
    expected = {(a, b, p) for a, b in OPPONENTS for p in PARTNERS}
    actual = [(c["earlier"], c["later"], c["partner"]) for c in cells]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("opponent-order table is incomplete or duplicated")
    results = []
    layouts = [(seed, seat) for seed in range(START, START + COUNT) for seat in range(4)]
    for cell in cells:
        rows = cell["rows"]
        if set(rows) != set(LEARNERS) or any(
                [(r["seed"], r["seat"]) for r in group] != layouts for group in rows.values()):
            raise ValueError("evaluation rows differ from fixed layout blocks")
        candidate = totals(rows["candidate"])
        results.append(dict(earlier=cell["earlier"], later=cell["later"], partner=cell["partner"],
                            learner_totals={name: totals(group) for name, group in rows.items()},
                            candidate_non_draw=candidate["wins"] + candidate["losses"],
                            comparison=compare(rows["heuristic"], rows["candidate"])))
    return results


def audit():
    report = json.loads(REPORT.read_text(encoding="utf8"))
    if report["hashes"] != dict(engine=sha(RUN / "junqi.exe"), rules=sha(RUN / "ruleset.json")):
        raise ValueError("frozen opponent-order experiment changed")
    if report["roster"] != model_roster() or report["summary"] != summarize(report["cells"]):
        raise ValueError("opponent-order contrasts differ from raw results")
    print("PASS two enemy slots, complete ordered contexts, and paired outcomes")


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("opponent-order experiment exists; use audit")
    roster = model_roster()
    RUN.mkdir()
    engine, rules = RUN / "junqi.exe", RUN / "ruleset.json"
    shutil.copyfile(EXE, engine)
    shutil.copyfile(RULES, rules)
    started = time.monotonic()
    common = (engine, rules, roster, "candidate", "independent", "learned_enemy", START, 2)
    shared, separate = evaluate(*common), evaluate(*common, later_enemy="learned_enemy")
    for a, b in zip(shared, separate, strict=True):
        a.pop("learner_ns")
        b.pop("learner_ns")
        if a != b:
            raise ValueError("separate equal opponents differ from shared configuration")
    cells = []
    for earlier, later in OPPONENTS:
        for partner in PARTNERS:
            rows = {name: evaluate(engine, rules, roster, name, partner, earlier, START, COUNT,
                                   later_enemy=later) for name in LEARNERS}
            cells.append(dict(earlier=earlier, later=later, partner=partner, rows=rows))
            print(f"Completed {earlier}/{later}, partner {partner}", flush=True)
    report = dict(roster=roster, cells=cells, summary=summarize(cells),
                  hashes=dict(engine=sha(engine), rules=sha(rules)), seconds=time.monotonic()-started)
    data = (json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(data) > 4 * 1024**2:
        raise ValueError("opponent-order output budget")
    with REPORT.open("xb") as output:
        output.write(data)
    audit()


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
