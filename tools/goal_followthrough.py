"""Audit first-step information and a distinct continuation intervention."""
from collections import Counter, defaultdict
from fractions import Fraction
import json
import shutil
import sys
import time

from early_route import EXE, ROOT, RULES, collect, model_roster, score_delta, sha, source_keys, scores_summary

TICKET = "measure-goal-directed-followthrough"
RUN = ROOT / "build" / (TICKET + "-v1")
REPORT = ROOT / "docs" / "REPORTS" / (TICKET + ".json")
GRID = source_keys(305000, 96)
SCREEN, CONFIRM = 6, 8
SCORES = {-1: "U", 0: "L", 1: "D", 2: "W"}
USAGES = ("parent", "dark", "ally", "full")


def select_actions(scores, groups, weights):
    def choose(worlds):
        return max(range(len(scores[0])),
                   key=lambda a: sum(weights[w] * scores[w][a] for w in worlds))

    blind = choose(range(len(groups)))
    by_group = {g: choose([i for i, label in enumerate(groups) if label == g]) for g in set(groups)}
    return [[0] * len(groups), [blind] * len(groups), [by_group[g] for g in groups],
            [choose([w]) for w in range(len(groups))]]


def fixture():
    # A common blind action differs from the action possible with perfect identity.
    values = [[1, 0], [0, 1], [1, 0], [0, 1]]
    values = [[v[1], v[0]] for v in values]
    weights = [Fraction(i, 10) for i in (4, 1, 2, 3)]
    arms = select_actions(values, [0, 0, 1, 1], weights)
    totals = [sum(w * row[a] for w, row, a in zip(weights, values, plan, strict=True))
              for plan in arms]
    if totals != [Fraction(i, 10) for i in (4, 6, 7, 10)]:
        raise ValueError("incorrect information nesting")
    if pinned_delta([-1, -1], 0, 0) != (0, 0) or pinned_delta([-1, 2], 1, 0) != (0, 1):
        raise ValueError("incorrect external-cutoff bounds")
    if interaction([1, 1, [2, 1], [2, 1]]) != (0, 0):
        raise ValueError("common continuation gain assigned to guard")
    return [str(t) for t in totals]


def pinned_delta(row, new, old):
    return (0, 0) if new == old else score_delta(row[new], row[old])


def interaction(row):
    before = score_delta(row[1], row[0])
    after = score_delta(row[-1][0], row[-2][0])
    if not (row[-1][1] or row[-2][1]):
        return (0, 0)
    return after[0] - before[1], after[1] - before[0]


def verify_point(point):
    moves, cells = point["moves"], point["cells"]
    if (not 2 <= len(moves) <= min(5, point["legal"]) or
            len({tuple(move) for move in moves}) != len(moves) or len(cells) not in (2, 4, 6, 8)):
        raise ValueError("invalid frozen candidate set or incomplete world")
    groups = []
    for i in range(0, len(cells), 2):
        a, b = cells[i:i+2]
        if (a["response"], b["response"]) != (0, 2) or a["group"] != b["group"]:
            raise ValueError("missing response or mismatched ally information")
        groups.append(a["group"])
        for cell in (a, b):
            for row in cell["samples"]:
                if (len(row) != len(moves)+2 or any(v not in SCORES for v in row[:-2]) or
                        any(s not in SCORES or changed not in (0, 1) or (not changed and s != row[j])
                            for j, (s, changed) in enumerate(row[-2:]))):
                    raise ValueError("invalid action or changed inactive intervention")
            if len(cell["samples"]) != SCREEN+CONFIRM:
                raise ValueError("incomplete screening or confirmation seed range")
    if groups not in ([0], [0, 0], [0, 1], [0, 0, 1], [0, 1, 1], [0, 0, 1, 1]):
        raise ValueError("invalid visible grouping")
    return groups


def measure(point, response, seed, report, deltas):
    groups = verify_point(point)
    cells = [cell["samples"] for cell in point["cells"] if cell["response"] == response]
    wins = [[sum(max(0, row[a]) for row in cell[:SCREEN])
             for a in range(len(point["moves"]))] for cell in cells]
    choices = select_actions(wins, groups, [1] * len(groups))
    report["points"] += 1
    report["equal_five"] += point["same_features"]
    report["ally_partition"] += int(1 in groups)
    report["full_partition"] += int(len(groups) > len(set(groups)))
    report["candidate_count"] += len(point["moves"])
    report["confirmation_samples"] += CONFIRM * len(groups)
    for w, cell in enumerate(cells):
        for row in cell[SCREEN:]:
            for name, choice in zip(USAGES, choices, strict=True):
                report[f"{name}/{SCORES[row[choice[w]]]}"] += 1
            for label, new, old in (("dark_vs_parent", 1, 0), ("ally_vs_dark", 2, 1),
                                    ("full_vs_ally", 3, 2)):
                # Missing partitions are missing coverage, rather than zero-value evidence.
                if (new == 2 and 1 not in groups) or (new == 3 and len(groups) == len(set(groups))):
                    continue
                a, b = choices[new][w], choices[old][w]
                report[f"{label}/switched"] += int(a != b)
                deltas[label][seed].append(pinned_delta(row, a, b))
            report["later_parent_changed"] += row[-2][1]
            report["later_guard_changed"] += row[-1][1]
            deltas["guard_vs_parent"][seed].append(score_delta(row[1], row[0]))
            deltas["follow_interaction"][seed].append(interaction(row))


def summarize(rows):
    if [tuple(r[k] for k in ("seed", "seat", "variant")) for r in rows] != GRID:
        raise ValueError("unexpected ordinary source grid")
    coverage = defaultdict(Counter)
    counts = defaultdict(Counter)
    deltas = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        base = coverage[row["variant"]]
        base.update(games=1, decisions=row["decisions"], ordinary_offers=row["offers"][0],
                    vacancy_offers=row["offers"][1])
        if sorted(p["vacancy"] for p in row["points"]) != [i for i, v in enumerate(row["offers"]) if v]:
            raise ValueError("malformed points or duplicate source coverage")
        for point in row["points"]:
            base[f"layout_with_point/{point['vacancy']}"] += 1
            for response in (0, 2):
                label = f"{row['variant']}/{point['vacancy']}/{response}"
                measure(point, response, row["seed"], counts[label], deltas[label])
    return dict(coverage={str(k): dict(v) for k, v in coverage.items()},
                panels={k: dict(c) | {"team": {name: scores_summary(v) for name, v in deltas[k].items()}}
                        for k, c in counts.items()})


def frozen(parent):
    return dict(engine=sha(RUN/"junqi.exe"), rules=sha(RUN/"ruleset.json"), parent=parent)


def replay(parent, key):
    return collect(RUN/"junqi.exe", RUN/"ruleset.json", parent, *key, mode="goal-followthrough")


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("goal-followthrough exists; use audit")
    parent = model_roster()["candidate"]
    RUN.mkdir()
    shutil.copyfile(EXE, RUN/"junqi.exe")
    shutil.copyfile(RULES, RUN/"ruleset.json")
    calibration = fixture()
    started = time.monotonic()
    rows = [replay(parent, key) for key in GRID]
    report = dict(frozen=frozen(parent), fixture=calibration, rows=rows, summary=summarize(rows),
                  seconds=time.monotonic()-started)
    output = (json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(output) > 4 * 1024**2:
        raise ValueError("goal-followthrough report exceeded byte cap")
    with REPORT.open("xb") as file:
        file.write(output)
    print(f"Measured {len(rows)} ordinary source games and checked the exact information fixture")


def audit():
    result = json.loads(REPORT.read_text(encoding="utf8"))
    parent = model_roster()["candidate"]
    rows = result["rows"]
    if (result["fixture"] != fixture() or result["frozen"] != frozen(parent) or
            result["summary"] != summarize(rows)):
        raise ValueError("frozen results did not reproduce")
    for variant in (0, 2):
        selected = next((r for r in rows if r["variant"] == variant and r["points"]), rows[variant//2])
        if replay(parent, tuple(selected[k] for k in ("seed", "seat", "variant"))) != selected:
            raise ValueError("deterministic audit did not reproduce a source")
    print("PASS information fixture, public-prefix invariance, paired continuation and replay")


if __name__ == "__main__":
    if not sys.argv[1:]:
        run()
    elif sys.argv[1:] == ["audit"]:
        audit()
    else:
        raise SystemExit("usage: goal_followthrough.py [audit]")
