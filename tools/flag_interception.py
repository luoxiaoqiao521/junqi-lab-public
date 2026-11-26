"""Bounded coverage and outcome probe for visible one-turn flag threats."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sys
import time

from camp_control import collect
from camp_pairs import score_delta, scores_summary
from learn_short import EXE, RULES
from partner_crossplay import model_roster, sha

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/"build/public-flag-interception-v1"
REPORT = ROOT/"docs/REPORTS/public-flag-interception.json"
START, COUNT = 287000, 48
VARIANTS = (0, 2)


def source_keys(start=START, count=COUNT):
    return [(seed, (seed-start)%4, variant) for seed in range(start, start+count) for variant in VARIANTS]


def summarize_group(rows, kind):
    matches = set()
    with_alternate = set()
    cells = defaultdict(lambda: defaultdict(list))
    events = defaultdict(Counter)
    games = 0
    for row in rows:
        points = [p for p in row["points"] if p["kind"] == kind]
        if not points:
            continue
        if len(points) != 1:
            raise ValueError("expected the first occurrence per flag owner")
        point, = points
        games += 1
        matches.add(row["seed"])
        worlds = point["worlds"]
        if worlds == 2:
            with_alternate.add(row["seed"])
        keys = [(cell["world"], cell["response"]) for cell in point["cells"]]
        if (worlds not in (1, 2) or keys != [(w, r) for w in range(worlds) for r in (0, 1)]
                or point["intercept"][1] != point["threat"] or point["same_five"]):
            raise ValueError("invalid interception context or feature contrast")
        for cell in point["cells"]:
            key = f'{cell["world"]}/{cell["response"]}'
            if len(cell["pairs"]) != 8:
                raise ValueError("wrong counterfactual budget")
            for a, b in cell["pairs"]:
                cells[key][row["seed"]].append(score_delta(a[0], b[0]))
                for name, branch in (("intercept", a), ("advance", b)):
                    if (len(branch) != 4 or branch[0] not in (-1, 0, 1, 2) or
                            any(x not in (0, 1) for x in branch[1:]) or
                            (cell["response"] == 0 and branch[3])):
                        raise ValueError("invalid branch outcome")
                    counter = events[f"{key}/{name}"]
                    counter["games"] += 1
                    counter["flag_lost"] += branch[1]
                    counter["enemy_saw_flag"] += branch[2]
                    counter["priority_applied"] += branch[3]
                    counter[f"score_{branch[0]}"] += 1
    return dict(games=games, layouts=len(matches), compatible_layouts=len(with_alternate),
                comparisons={key: scores_summary(layouts) for key, layouts in cells.items()},
                events={key: dict(counter) for key, counter in events.items()})


def summarize(rows, start=START, count=COUNT):
    if [(r["seed"], r["seat"], r["variant"]) for r in rows] != source_keys(start, count):
        raise ValueError("invalid flag threat source grid")
    counts = {}
    for variant in VARIANTS:
        games = [r for r in rows if r["variant"] == variant]
        if any(r["source_score"] not in (-1, 0, 1, 2) or
               any(p["kind"] not in (0, 1) for p in r["points"]) for r in games):
            raise ValueError("invalid source score or category")
        counts[str(variant)] = dict(no_match=sum(not r["points"] for r in games),
                                   outcomes=dict(Counter(str(r["source_score"]) for r in games)))
    return dict(sources=counts, groups=[summarize_group(rows, kind) for kind in (0, 1)])


def audit():
    data = json.loads(REPORT.read_text(encoding="utf8"))
    parent = model_roster()["candidate"]
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    if data["frozen"] != dict(engine=sha(engine), rules=sha(rules), parent=parent):
        raise ValueError("flag interception artifacts changed")
    rows = data["sources"]
    if data["summary"] != summarize(rows):
        raise ValueError("flag interception results no longer match raw continuations")
    row = next((row for row in rows if row["points"]), rows[0])
    for source in (row, rows[-1]):
        if source != collect(engine, rules, parent, source["seed"], source["seat"], source["variant"],
                             mode="flag-interception"):
            raise ValueError("flag interception replay differs")
    print("PASS full source grid, branch identities, score bounds and fixed replay")


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("flag interception archive exists; use audit")
    parent = model_roster()["candidate"]
    RUN.mkdir()
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    shutil.copyfile(EXE, engine)
    shutil.copyfile(RULES, rules)
    started = time.monotonic()
    rows = []
    for seed, seat, variant in source_keys():
        rows.append(collect(engine, rules, parent, seed, seat, variant, mode="flag-interception"))
    data = dict(sources=rows, summary=summarize(rows), seconds=time.monotonic()-started,
                frozen=dict(engine=sha(engine), rules=sha(rules), parent=parent))
    content = (json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(content) > 1024**2:
        raise ValueError("flag interception data exceeded its byte cap")
    with REPORT.open("xb") as output:
        output.write(content)
    print("Completed natural flag-threat coverage and paired continuations")


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
