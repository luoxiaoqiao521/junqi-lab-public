"""Compare early and later flag threats with the same action feature differences."""
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
RUN = ROOT/"build/public-flag-urgency-contrast-v1"
REPORT = ROOT/"docs/REPORTS/public-flag-urgency-contrast.json"
START, COUNT = 289000, 128
MAX_MATCHES = 24


def paired_selection(rows):
    if [(r["seed"], r["seat"], r["variant"]) for r in rows] != source_keys(START, COUNT):
        raise ValueError("wrong urgency source population")
    buckets = [defaultdict(list), defaultdict(list)]
    for row in rows:
        kinds = set()
        for point in row["points"]:
            kind = point["group"]
            if kind not in (0, 1) or kind in kinds or len(point["delta"]) != 5:
                raise ValueError("invalid public timing offer")
            kinds.add(kind)
            buckets[kind][(row["variant"], *point["delta"])].append((row, point))
    selected = []
    used = [set(), set()]
    for key in sorted(buckets[0].keys() & buckets[1].keys()):
        for early in buckets[0][key]:
            if early[0]["seed"] in used[0]:
                continue
            late = next((case for case in buckets[1][key] if case[0]["seed"] not in used[1]), None)
            if late is None:
                break
            selected.append((early, late))
            used[0].add(early[0]["seed"])
            used[1].add(late[0]["seed"])
            if len(selected) == MAX_MATCHES:
                break
        if len(selected) == MAX_MATCHES:
            break
    return selected, [len({case[0]["seed"] for group in b.values() for case in group}) for b in buckets]


def add_case(branch, selection, comparisons, events):
    source, point = selection
    worlds = branch["worlds"]
    if worlds not in (1, 2) or len(branch["cells"]) != worlds*2:
        raise ValueError("incomplete matched branch settings")
    if {k: v for k, v in branch.items() if k not in ("cells", "worlds")} != point:
        raise ValueError("matched action pair changed after selection")
    for index, cell in enumerate(branch["cells"]):
        key = f"{index//2}/{index%2}"
        if len(cell) != 8:
            raise ValueError("wrong seed-pair count")
        for row in cell:
            a, b = row[:5], row[5:]
            if (len(row) != 10 or not all(record[0] in (-1, 0, 1, 2) and
                    all(x in (0, 1) for x in record[1:]) and (index%2 or not record[2])
                    for record in (a, b))):
                raise ValueError("invalid score or public-rotation counters")
            comparisons[key][source["seed"]].append(score_delta(a[0], b[0]))
            for label, record in (("intercept", a), ("advance", b)):
                counter = events[f"{key}/{label}"]
                counter.update(rollouts=1, lost=record[1], pressure=record[2],
                               ally_first=record[3], enemy_turn=record[4])
                counter[f"score_{record[0]}"] += 1


def summarize(rows, cases):
    selected, coverage = paired_selection(rows)
    outputs = []
    for group in (0, 1):
        comparisons = defaultdict(lambda: defaultdict(list))
        events = defaultdict(Counter)
        for match, recorded in zip(selected, cases, strict=True):
            source, _ = match[group]
            needed = recorded[group]
            if any(needed[k] != source[k] for k in ("seed", "seat", "variant")) or len(needed["points"]) != 1:
                raise ValueError("matched source differs from frozen scan")
            add_case(needed["points"][0], match[group], comparisons, events)
        outputs.append(dict(differences={k: scores_summary(v) for k, v in comparisons.items()},
                            events={k: dict(v) for k, v in events.items()}))
    return dict(available_layouts=coverage, matched_pairs=len(selected), groups=outputs)


def audit():
    data = json.loads(REPORT.read_text(encoding="utf8"))
    parent = model_roster()["candidate"]
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    if data["frozen"] != dict(engine=sha(engine), rules=sha(rules), parent=parent):
        raise ValueError("frozen urgency artifacts changed")
    if data["summary"] != summarize(data["sources"], data["cases"]):
        raise ValueError("raw matched urgency results no longer agree with the summary")
    if data["cases"]:
        for group, row in enumerate(data["cases"][0]):
            actual = collect(engine, rules, parent, row["seed"], row["seat"], row["variant"],
                             mode="flag-urgency-pair", extra=(group,))
            if actual != row:
                raise ValueError("known matched source failed replay")
    print("PASS independent source keys, exact feature matches, score bounds and replay")


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("urgency experiment already exists; use audit")
    parent = model_roster()["candidate"]
    RUN.mkdir()
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    shutil.copyfile(EXE, engine)
    shutil.copyfile(RULES, rules)
    started = time.monotonic()
    rows = [collect(engine, rules, parent, *key, mode="flag-urgency-scan") for key in source_keys(START, COUNT)]
    selected, coverage = paired_selection(rows)
    print(f"Independent source coverage {coverage}; exact matched comparison sets {len(selected)}", flush=True)
    cases = [
        [collect(engine, rules, parent, row["seed"], row["seat"], row["variant"],
                 mode="flag-urgency-pair", extra=(group,)) for group, (row, _) in enumerate(pair)]
        for pair in selected
    ]
    report = dict(sources=rows, cases=cases, summary=summarize(rows, cases),
                  seconds=time.monotonic()-started, frozen=dict(engine=sha(engine), rules=sha(rules), parent=parent))
    data = (json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(data) > 2*1024**2:
        raise ValueError("urgency output byte budget exceeded")
    with REPORT.open("xb") as output:
        output.write(data)


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
