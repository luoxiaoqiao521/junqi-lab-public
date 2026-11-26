"""Distinguish projected local arrival, real turn-order arrival and team score."""
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
RUN = ROOT/"build/public-regional-arrival-v1"
REPORT = ROOT/"docs/REPORTS/public-regional-arrival.json"
START, COUNT = 292000, 48


def validate(case, seat):
    if (case["arrival"][0] != case["closer"][0] or case["arrival"][1] == case["closer"][1] or
            case["worlds"] not in (1, 2) or len(case["cells"]) != case["worlds"]*2):
        raise ValueError("mismatched regional arrivals or responses")
    for key in ("ready_arrival", "ready_closer"):
        counts = case[key]
        if len(counts) != 4 or any(not 0 <= a <= b <= 25 for a, b in counts) or counts[seat][0] != counts[seat][1]:
            raise ValueError("invalid known-seat count or remote uncertainty bounds")


def summarize(rows):
    if [(r["seed"], r["seat"], r["variant"]) for r in rows] != source_keys(START, COUNT):
        raise ValueError("unexpected regional layout grid")
    coverage = Counter()
    matches = set()
    compatible = set()
    team = defaultdict(lambda: defaultdict(list))
    arrival = defaultdict(Counter)
    possible = Counter()
    for row in rows:
        hit = row["case"]
        if not hit:
            continue
        validate(hit, row["seat"])
        matches.add(row["seed"])
        coverage[str(row["variant"])] += 1
        if hit["worlds"] == 2:
            compatible.add(row["seed"])
        for arm in ("arrival", "closer"):
            counts = hit[f"ready_{arm}"]
            if any(low != high for low, high in counts):
                possible[arm] += 1
        for index, cell in enumerate(hit["cells"]):
            if len(cell) != 8:
                raise ValueError("regional seed-pair budget changed")
            key = f"{index//2}/{index%2}"
            for a, a_turn, a_arrived, b, b_turn, b_arrived in cell:
                if not (0 <= a_arrived <= a_turn <= 1 and 0 <= b_arrived <= b_turn <= 1 and
                        a in (-1, 0, 1, 2) and b in (-1, 0, 1, 2)):
                    raise ValueError("invalid arrival or terminal outcome")
                team[key][row["seed"]].append(score_delta(a, b))
                arrival[key].update(pairs=1, arrival_turn=a_turn, closer_turn=b_turn,
                                    arrival_entered=a_arrived, closer_entered=b_arrived)
    return dict(source_coverage=dict(coverage), layouts=len(matches), compatible_layouts=len(compatible),
                nonexact_remote_coverage=dict(possible), arrival={k: dict(v) for k, v in arrival.items()},
                team={k: scores_summary(v) for k, v in team.items()})


def audit():
    saved = json.loads(REPORT.read_text(encoding="utf8"))
    parent = model_roster()["candidate"]
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    if (saved["frozen"] != dict(engine=sha(engine), rules=sha(rules), parent=parent) or
            saved["summary"] != summarize(saved["rows"])):
        raise ValueError("regional artifact or coverage summary changed")
    rows = saved["rows"]
    known = next((row for row in rows if row["case"]), rows[0])
    for row in (known, rows[-1]):
        if row != collect(engine, rules, parent, row["seed"], row["seat"], row["variant"],
                          mode="regional-arrival"):
            raise ValueError("frozen regional replay did not reproduce")
    print("PASS region bounds, legal-arrival receipts, scores and replay")


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("regional arrival data exists; use audit")
    parent = model_roster()["candidate"]
    RUN.mkdir()
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    shutil.copyfile(EXE, engine)
    shutil.copyfile(RULES, rules)
    started = time.monotonic()
    rows = [collect(engine, rules, parent, *key, mode="regional-arrival") for key in source_keys(START, COUNT)]
    report = dict(rows=rows, summary=summarize(rows), seconds=time.monotonic()-started,
                  frozen=dict(engine=sha(engine), rules=sha(rules), parent=parent))
    data = (json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(data) > 2*1024**2:
        raise ValueError("regional arrival result exceeds byte budget")
    with REPORT.open("xb") as output:
        output.write(data)
    print(f"Regional arrivals compared in {report['summary']['layouts']} independent layout blocks")


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
