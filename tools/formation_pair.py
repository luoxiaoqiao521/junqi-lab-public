"""Compare a one-swap flag-covering layout intervention."""
from collections import Counter
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
RUN = ROOT/"build/opening-formation-one-swap-v1"
REPORT = ROOT/"docs/REPORTS/opening-formation-one-swap.json"
START, COUNT = 293000, 64


def summarize(rows):
    if [(r["seed"], r["seat"], r["variant"]) for r in rows] != source_keys(START, COUNT):
        raise ValueError("unexpected formation comparison layout grid")
    groups = {}
    for variant in (0, 2):
        scores = {}
        metrics = {"closed": Counter(), "opening": Counter()}
        for row in rows:
            if row["variant"] != variant or row["pair"] is None:
                continue
            a, b = row["pair"]
            scores[row["seed"]] = [score_delta(a[0], b[0])]
            for key, sample in zip(metrics, (a, b), strict=True):
                outcome, ready, first, taken = sample
                if outcome not in (-1, 0, 1, 2) or not -1 <= ready <= 25 or first < -1 or taken not in (0, 1):
                    raise ValueError("invalid formation endpoint")
                metrics[key][f"score_{outcome}"] += 1
                metrics[key]["explicit_flag_taken"] += taken
                metrics[key]["reached_center"] += first > 0
                metrics[key]["nominal_first_ready"] += max(0, ready)
                metrics[key]["has_opening_turn"] += ready >= 0
                metrics[key]["first_arrival_ply_sum"] += max(0, first)
        groups[str(variant)] = dict(skipped=COUNT-len(scores), events={k: dict(v) for k, v in metrics.items()},
                                    closed_minus_open=scores_summary(scores))
    return groups


def audit():
    report = json.loads(REPORT.read_text(encoding="utf8"))
    parent = model_roster()["candidate"]
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    if (report["frozen"] != dict(engine=sha(engine), rules=sha(rules), parent=parent) or
            report["summary"] != summarize(report["rows"])):
        raise ValueError("formation inputs or paired outcomes changed")
    rows = report["rows"]
    for row in (rows[0], rows[-1]):
        if row != collect(engine, rules, parent, row["seed"], row["seat"], row["variant"],
                          mode="formation-one-swap"):
            raise ValueError("fixed formation replay failed")
    print("PASS legal formation pair, no observation leak, layout blocks and terminal outcomes")


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("formation trial already exists; use audit")
    parent = model_roster()["candidate"]
    RUN.mkdir()
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    shutil.copyfile(EXE, engine)
    shutil.copyfile(RULES, rules)
    started = time.monotonic()
    rows = [collect(engine, rules, parent, *key, mode="formation-one-swap")
            for key in source_keys(START, COUNT)]
    report = dict(rows=rows, summary=summarize(rows), seconds=time.monotonic()-started,
                  frozen=dict(engine=sha(engine), rules=sha(rules), parent=parent))
    with REPORT.open("x", encoding="utf8") as output:
        output.write(json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n")
    print(f"Paired {len(rows)} layout and opponent blocks with one own swap")


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
