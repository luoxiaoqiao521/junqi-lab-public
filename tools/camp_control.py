"""Freeze normal-opening source coverage for the public camp comparison."""
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from learn_short import EXE, RULES
from partner_crossplay import model_roster, sha

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/"build/public-camp-control-window-v1"
REPORT = ROOT/"docs/REPORTS/public-camp-control-window.json"
DEVELOPMENT = 286600
CONFIRMATION = 286800
LAYOUTS = 64
VARIANTS = (0, 2)
MIN_LAYOUTS = 8


def source_keys(start):
    return [(seed, (seed-start) % 4, variant) for seed in range(start, start+LAYOUTS)
            for variant in VARIANTS]


def collect(engine, rules, parent, seed, seat, variant, mode="camp-control-scan", extra=()):
    result = subprocess.run(
        [str(engine), mode, str(rules), *map(str, (seed, seat, variant, *parent["weights"], *extra))],
        capture_output=True, check=True, text=True, timeout=20, creationflags=0x40 | 0x08000000)
    return json.loads(result.stdout)


def validate_point(point, kind, seat):
    seal, press = point["seal"], point["press"]
    if (len(seal) != 2 or len(press) != 2 or seal[0] != press[0] or seal[1] == press[1] or
            not all(0 <= node < 129 for node in (*seal, *press, point["target"])) or
            point["target_seat"] not in range(4) or point["target_seat"] % 2 == seat % 2 or
            (point["backup"] != -1) != bool(kind) or point["ply"] < 0):
        raise ValueError("invalid public camp contrast")


def select_sources(rows, kind, limit=16):
    selected = {}
    for row in rows:
        point = row["points"][kind]
        if point is None:
            continue
        previous = selected.get(row["seed"])
        if previous is None or point["ply"] < previous["points"][kind]["ply"]:
            selected[row["seed"]] = row
    return list(selected.values())[:limit]


def summarize(rows, start=DEVELOPMENT):
    if [(r["seed"], r["seat"], r["variant"]) for r in rows] != source_keys(start):
        raise ValueError("incomplete, duplicated, or reordered source layout grid")
    layouts = [set(), set()]
    summaries = {}
    for variant in VARIANTS:
        games = [row for row in rows if row["variant"] == variant]
        counts = Counter()
        for row in games:
            if row["score"] not in (-1, 0, 1, 2) or len(row["points"]) != 2 or row["decisions"] < 0:
                raise ValueError("invalid scan source result")
            counts["no_match"] += all(point is None for point in row["points"])
            counts["decisions"] += row["decisions"]
            for kind, point in enumerate(row["points"]):
                if point is None:
                    continue
                validate_point(point, kind, row["seat"])
                layouts[kind].add(row["seed"])
                counts[f"group_{kind}"] += 1
                counts[f"same_five_{kind}"] += point["same_five"]
                counts[f"seen_moved_{kind}"] += point["seen_moved"]
        summaries[str(variant)] = dict(games=len(games), counts=dict(counts),
                                       outcomes=dict(Counter(str(row["score"]) for row in games)))
    coverage = [len(group) for group in layouts]
    return dict(source_variants=summaries, usable_layouts=coverage,
                enough_for_paired_screen=all(value >= MIN_LAYOUTS for value in coverage))


def audit():
    report = json.loads(REPORT.read_text(encoding="utf8"))
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    parent = model_roster()["candidate"]
    if report["frozen"] != dict(engine=sha(engine), rules=sha(rules), parent=parent):
        raise ValueError("camp scan inputs changed")
    rows = report["development"]
    if report["summary"] != summarize(rows) or report["confirmation_started"]:
        raise ValueError("coverage no longer agrees with source rows or unopened confirmation")
    for row in (rows[0], rows[1], rows[-2], rows[-1]):
        if row != collect(engine, rules, parent, row["seed"], row["seat"], row["variant"]):
            raise ValueError("fixed public source rows did not reproduce")
    print("PASS source grid, public pair order and unopened confirmation cohort")


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("camp scan archive exists; use audit")
    parent = model_roster()["candidate"]
    RUN.mkdir()
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    shutil.copyfile(EXE, engine)
    shutil.copyfile(RULES, rules)
    started = time.monotonic()
    rows = [collect(engine, rules, parent, *key) for key in source_keys(DEVELOPMENT)]
    summary = summarize(rows)
    report = dict(development=rows, summary=summary, confirmation_started=False,
                  seconds=time.monotonic()-started,
                  frozen=dict(engine=sha(engine), rules=sha(rules), parent=parent))
    with REPORT.open("x", encoding="utf8") as output:
        output.write(json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Scanned {len(rows)} games; distinct layouts by public condition {summary['usable_layouts']}")


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
