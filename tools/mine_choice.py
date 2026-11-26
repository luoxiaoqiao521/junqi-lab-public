"""Run a targeted mine/non-mine attack contrast on an identical public trace."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sys
import time

from camp_control import collect
from camp_pairs import score_delta, scores_summary
from flag_interception import source_keys
from learn_short import EXE, RULES
from partner_crossplay import model_roster, sha

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/"build/public-mine-action-contrast-v1"
REPORT = ROOT/"docs/REPORTS/public-mine-action-contrast.json"
START, COUNT = 290000, 64
LABELS = ("mine/random", "mine/heuristic", "other/random", "other/heuristic")


def summarize(rows):
    if [(row["seed"], row["seat"], row["variant"]) for row in rows] != source_keys(START, COUNT):
        raise ValueError("wrong independent mine target source grid")
    totals = Counter()
    complete = set()
    immediate = Counter()
    reversals = Counter()
    comparisons = {name: defaultdict(list) for name in LABELS}
    for row in rows:
        point = row["case"]
        if point is None:
            continue
        totals["offers"] += 1
        cells = point["cells"]
        if not cells:
            if point["immediate"]:
                raise ValueError("partial targeted alternate")
            totals["no_compatible_alternate"] += 1
            continue
        if (len(cells) != 4 or len(point["immediate"]) != 2 or point["immediate"][0] != [1, 2] or
                any(result not in (1, 2, 3) for outcomes in point["immediate"] for result in outcomes)):
            raise ValueError("wrong targeted alternate or immediate outcome")
        totals["complete"] += 1
        complete.add(row["seed"])
        immediate[",".join(map(str, point["immediate"][1]))] += 1
        deltas = []
        for key, cell in zip(LABELS, cells, strict=True):
            if len(cell) != 8 or any(a not in (-1, 0, 1, 2) or b not in (-1, 0, 1, 2) for a, b in cell):
                raise ValueError("wrong terminal reward or budget")
            samples = [score_delta(a, b) for a, b in cell]
            comparisons[key][row["seed"]].extend(samples)
            lo, hi = (sum(x)/len(samples) for x in zip(*samples, strict=True))
            deltas.append(1 if lo > 0 else -1 if hi < 0 else 0)
        for response in (0, 1):
            if deltas[response] * deltas[response+2] == -1:
                reversals[str(response)] += 1
    return dict(totals=dict(totals), complete_layouts=len(complete), nonmine_collision=dict(immediate),
                point_sign_reversals=dict(reversals),
                team={key: scores_summary(layouts) for key, layouts in comparisons.items()})


def audit():
    report = json.loads(REPORT.read_text(encoding="utf8"))
    parent = model_roster()["candidate"]
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    if report["frozen"] != dict(engine=sha(engine), rules=sha(rules), parent=parent):
        raise ValueError("mine target artifacts changed")
    rows = report["sources"]
    if report["summary"] != summarize(rows):
        raise ValueError("targeted contrast results differ from the summary")
    known = next((row for row in rows if row["case"] and row["case"]["cells"]), rows[0])
    for row in (known, rows[-1]):
        if row != collect(engine, rules, parent, row["seed"], row["seat"], row["variant"], mode="mine-choice"):
            raise ValueError("fixed mine-target branch did not reproduce")
    print("PASS public action pair, targeted rank swap, terminal bounds and replay")


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("mine choice already exists; use audit")
    parent = model_roster()["candidate"]
    RUN.mkdir()
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    shutil.copyfile(EXE, engine)
    shutil.copyfile(RULES, rules)
    started = time.monotonic()
    rows = [collect(engine, rules, parent, *key, mode="mine-choice") for key in source_keys(START, COUNT)]
    report = dict(sources=rows, summary=summarize(rows), seconds=time.monotonic()-started,
                  frozen=dict(engine=sha(engine), rules=sha(rules), parent=parent))
    data = (json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(data) > 1024**2:
        raise ValueError("mine target result exceeds byte budget")
    with REPORT.open("xb") as output:
        output.write(data)
    print(f"Targeted mine comparison completed on {report['summary']['complete_layouts']} distinct layouts")


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
