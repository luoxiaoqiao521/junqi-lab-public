"""Paired normal-opening test of one publicly provable flag interception."""
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import time

from camp_control import collect
from evaluate_transfer import compare, totals
from learn_short import EXE, RULES
from partner_crossplay import model_roster, sha

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/"build/public-flag-guard-transfer-v1"
REPORT = ROOT/"docs/REPORTS/public-flag-guard-transfer.json"
START, COUNT = 298000, 128
MODE = "flag-guard-transfer"


def keys():
    return [(seed, seat, variant) for seed in range(START, START+COUNT)
            for seat in range(4) for variant in (0, 1)]


def branch(row, ar):
    result, window, change, plies, decisions = ar
    if (result not in (-1, 0, 1, 2) or change not in (0, 1)
            or not -1 <= window < plies or (change and window < 0)
            or not 0 <= decisions <= plies):
        raise ValueError("invalid guard intervention branch")
    return dict(seed=row["seed"], seat=row["seat"], first=row["first"],
                score=None if result == -1 else result/2, opportunities=int(window >= 0),
                decisions=decisions)


def summarize(rows):
    if [(r["seed"], r["seat"], r["variant"]) for r in rows] != keys():
        raise ValueError("invalid guard transfer source grid")
    summaries = {}
    for variant in (0, 1):
        control, candidate, touched = [], [], set()
        changes = 0
        transitions = Counter()
        for row in (record for record in rows if record["variant"] == variant):
            a, b = row["rows"]
            if a[1] != b[1] or a[2] or (not b[2] and a != b):
                raise ValueError("arm changed without the single declared intervention")
            if b[2]:
                touched.add(row["seed"])
                changes += 1
                transitions[f"{a[0]}->{b[0]}"] += 1
            control.append(branch(row, a))
            candidate.append(branch(row, b))
        summaries[str(variant)] = dict(control=totals(control), candidate=totals(candidate),
                                       changed_games=changes, changed_layouts=len(touched),
                                       comparison=compare(control, candidate), transitions=dict(transitions))
    primary, common = summaries["1"], summaries["0"]
    gain = primary["comparison"]
    controls = common["comparison"]
    gate = (min(primary["changed_layouts"], common["changed_layouts"]) >= 8
            and gain["difference_bounds"][0] >= .01 and gain["exploratory_block_interval"][0] > 0
            and controls["exploratory_block_interval"][0] >= -.005)
    return dict(contexts=summaries, next="replicate-urgent-guard" if gate else "rank-independent-experiment-families",
                eligible_for_model_promotion=False)


def frozen():
    return dict(engine=sha(RUN/"junqi.exe"), rules=sha(RUN/"ruleset.json"), parent=model_roster()["candidate"])


def replay(seed, seat, variant, parent):
    return collect(RUN/"junqi.exe", RUN/"ruleset.json", parent, seed, seat, variant, mode=MODE)


def audit():
    data = json.loads(REPORT.read_text(encoding="utf8"))
    inputs = frozen()
    rows = data["sources"]
    if data["frozen"] != inputs or data["summary"] != summarize(rows):
        raise ValueError("guard comparison differed on audit")
    row = next((r for r in rows if r["rows"][1][2]), rows[0])
    for source in (row, rows[-1]):
        if replay(source["seed"], source["seat"], source["variant"], inputs["parent"]) != source:
            raise ValueError("guard intervention replay changed")
    print("PASS single intervention, identical control trajectories and held-out layout blocks")


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("guard transfer archive exists; use audit")
    RUN.mkdir()
    shutil.copyfile(EXE, RUN/"junqi.exe")
    shutil.copyfile(RULES, RUN/"ruleset.json")
    inputs = frozen()
    start = time.monotonic()
    rows = [replay(seed, seat, variant, inputs["parent"]) for seed, seat, variant in keys()]
    data = dict(sources=rows, summary=summarize(rows), seconds=time.monotonic()-start, frozen=inputs)
    content = (json.dumps(data, separators=(",", ":"), allow_nan=False)+"\n").encode()
    if len(content) > 1024**2:
        raise ValueError("guard evaluation exceeded its byte cap")
    with REPORT.open("xb") as output:
        output.write(content)
    print("Completed normal-opening single-seat guard evaluation")


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
