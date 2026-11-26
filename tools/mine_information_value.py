"""Measure the offline value of one hidden bit with a fixed candidate set."""
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import combinations
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
TICKET = "compare-natural-decision-coverage"
RUN = ROOT / "build" / (TICKET + "-v1")
REPORT = ROOT / "docs" / "REPORTS" / (TICKET + ".json")
TARGETED = ROOT / "docs" / "REPORTS" / "public-mine-action-contrast.json"
MODE = "mine-information-value"
PRIORS = (25, 50, 75)
SCREEN, CONFIRM = 8, 12


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def sources():
    selected = []
    used = set()
    for row in read(TARGETED)["sources"]:
        if row["seed"] in used or not row["case"] or not row["case"]["cells"]:
            continue
        mine, _, other, _ = row["case"]["cells"]
        if (sum(a - b for a, b in mine) * sum(a - b for a, b in other)) < 0:
            used.add(row["seed"])
            selected.append((row["seed"], row["seat"], row["variant"]))
    return dict(ordinary=source_keys(302000, 64), targeted=selected[:8])


def screen_totals(cell):
    return [sum(max(value, 0) for value in values) for values in zip(*cell[:SCREEN], strict=True)]


def winner(scores):
    return max(range(len(scores)), key=scores.__getitem__)


def shared_action(mine, other, p):
    return winner([p * a + (1-p) * b for a, b in zip(mine, other, strict=True)])


def switch_points(mine, other):
    points = {Fraction(0), Fraction(1)}
    for a, b in combinations(range(len(mine)), 2):
        intercept = other[a] - other[b]
        slope = mine[a] - mine[b] - intercept
        if slope and 0 < (cross := Fraction(-intercept, slope)) < 1:
            points.add(cross)
    levels = sorted(points)
    arms = [shared_action(mine, other, (a+b)/2) for a, b in zip(levels, levels[1:])]
    return [[str(levels[i]), x, y] for i, (x, y) in enumerate(zip(arms, arms[1:]), 1) if x != y]


def validate_offer(point):
    moves, target = point["moves"], point["target"]
    if (not 2 <= len(moves) <= min(5, point["legal"]) or
            len({tuple(move[:2]) for move in moves}) != len(moves) or
            any(not (0 <= a < 129 and 0 <= b < 129 and a != b and quiet in (0, 1))
                for a, b, quiet in moves)):
        raise ValueError("invalid public candidate set")
    for role in ("engineer", "ordinary"):
        if [point[role], target, 0] not in moves:
            raise ValueError("missing declared attack")
    cells = point["cells"]
    if cells and (len(cells) != 4 or any(len(cell) != SCREEN+CONFIRM for cell in cells) or
                  any(len(row) != len(moves) or any(v not in (-1, 0, 1, 2) for v in row)
                      for cell in cells for row in cell)):
        raise ValueError("incomplete branch or invalid team outcome")


def horizon(row, response):
    point = row["case"]
    mine, other = point["cells"][response], point["cells"][response+2]
    a, b = screen_totals(mine), screen_totals(other)
    informed = [winner(a), winner(b)]
    attack = [point["moves"].index([point[role], point["target"], 0])
              for role in ("engineer", "ordinary")]
    reversal = (a[attack[0]] - a[attack[1]]) * (b[attack[0]] - b[attack[1]]) < 0
    return a, b, informed, reversal, [mine[SCREEN:], other[SCREEN:]]


def evaluate(branches, informed, blind, prior):
    bounds = [0.0, 0.0]
    transitions = Counter()
    for basis, ready, cell in zip((prior, 100-prior), informed, branches, strict=True):
        for outcomes in cell:
            a, b = outcomes[ready], outcomes[blind]
            ends = (0, 0) if ready == blind else score_delta(a, b)
            for i, end in enumerate(ends):
                bounds[i] += basis * end / (100 * CONFIRM)
            transitions["unresolved" if -1 in (a,b) and ready != blind else
                        "gain" if a>b else "loss" if a<b else "same"] += basis
    return bounds, transitions


def summarize(rows_by_group):
    result = {}
    for group, declared in sources().items():
        rows = rows_by_group[group]
        if [(r["seed"],r["seat"],r["variant"]) for r in rows] != declared:
            raise ValueError("source grid changed")
        coverage, panels = {}, {}
        for variant in (0, 2):
            context = [r for r in rows if r["variant"] == variant]
            offers = [r for r in context if r["case"] is not None]
            for row in offers:
                validate_offer(row["case"])
            complete = [r for r in offers if r["case"]["cells"]]
            coverage[str(variant)] = dict(games=len(context), offers=len(offers), complete=len(complete),
                                          offer_layouts=len({r["seed"] for r in offers}),
                                          complete_layouts=len({r["seed"] for r in complete}),
                                          quiet=sum(any(m[2] for m in r["case"]["moves"]) for r in complete))
            for response in (0, 1):
                panels[f"{variant}/{response}"] = summarize_panel(complete,response)
        result[group] = dict(coverage=coverage, panels=panels)
    return result


def summarize_panel(rows,response):
    layouts = {p:defaultdict(list) for p in PRIORS}
    changed = Counter()
    transitions = {p:Counter() for p in PRIORS}
    reversal = suppressed = 0
    switches = []
    for row in rows:
        a,b,informed,flipped,branches = horizon(row,response)
        reversal += flipped
        suppressed += flipped and informed[0] == informed[1]
        cuts = switch_points(a,b)
        if cuts:
            switches.append([row["seed"],row["seat"],cuts])
        for p in PRIORS:
            blind = shared_action(a,b,Fraction(p,100))
            bounds,tr = evaluate(branches,informed,blind,p)
            changed[p] += int(any(arm != blind for arm in informed))
            transitions[p].update(tr)
            layouts[p][row["seed"]].append(bounds)
    return dict(attack_reversals=reversal, same_best_despite_reversal=suppressed,
                screen_switches=switches,
                priors={str(p):dict(changed_cases=changed[p], result=scores_summary(layouts[p]),
                                    weighted_pair_outcomes=dict(transitions[p])) for p in PRIORS})


def check_reference():
    # With a shared quiet best, an attack-only cross-over has no decision value.
    cells = [[1,0,2]] * SCREEN, [[0,1,2]] * SCREEN
    a,b = map(screen_totals,cells)
    if switch_points(a,b) or winner(a) != 2 or winner(b) != 2:
        raise ValueError("known quiet dominance was missed")


def frozen(parent):
    return dict(engine=sha(RUN/"junqi.exe"), rules=sha(RUN/"ruleset.json"),
                original_mine_study=sha(TARGETED), parent=parent)


def replay(parent, key):
    return collect(RUN/"junqi.exe", RUN/"ruleset.json", parent, *key, mode=MODE)


def run():
    check_reference()
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("information value study already exists; use audit")
    parent = model_roster()["candidate"]
    RUN.mkdir()
    shutil.copyfile(EXE,RUN/"junqi.exe")
    shutil.copyfile(RULES,RUN/"ruleset.json")
    started = time.monotonic()
    rows = {label:[replay(parent,key) for key in keys] for label,keys in sources().items()}
    report = dict(frozen=frozen(parent), rows=rows, summary=summarize(rows),
                  seconds=time.monotonic()-started)
    payload = (json.dumps(report,separators=(",",":"),allow_nan=False)+"\n").encode()
    if len(payload) > 1024**2:
        raise ValueError("experiment output exceeded byte cap")
    with REPORT.open("xb") as output:
        output.write(payload)
    print("Recorded natural opportunities and independently tested frozen decisions")


def audit():
    check_reference()
    data = read(REPORT)
    parent = model_roster()["candidate"]
    if data["frozen"] != frozen(parent) or data["summary"] != summarize(data["rows"]):
        raise ValueError("evidence or split-seed summary changed")
    for rows in data["rows"].values():
        if not rows:
            continue
        complete = next((r for r in rows if r["case"] and r["case"]["cells"]),rows[0])
        for row in (complete,rows[-1]):
            if replay(parent,tuple(row[k] for k in ("seed","seat","variant"))) != row:
                raise ValueError("fixed continuation did not replay")
    print("PASS source grid, quiet dominance, prescreened choice and independent seed replay")


if __name__ == "__main__":
    if not sys.argv[1:]:
        run()
    elif sys.argv[1:] == ["audit"]:
        audit()
    else:
        raise SystemExit("usage: mine_information_value.py [audit]")
