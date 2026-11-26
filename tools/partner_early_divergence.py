"""Locate the first natural policy split on matched new opening layouts."""
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from learn_short import EXE, RULES
from partner_crossplay import evaluate, model_roster, sha

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "build/partner-early-divergence-v1"
REPORT = ROOT / "docs/REPORTS/partner-early-divergence.json"
START, COUNT = 283000, 24
CONTEXTS = [(partner, before, after) for partner in ("random", "independent")
            for before, after in (("heuristic", "random"), ("random", "heuristic"))]
SCORE = {None: -1, 0: 0, .5: 1, 1: 2}


def summarize(cells):
    if [tuple(cell["context"]) for cell in cells] != CONTEXTS:
        raise ValueError("missing, duplicate or reordered response context")
    result = []
    expected = [(seed, seat) for seed in range(START, START + COUNT) for seat in range(4)]
    first_seats = [row["first"] for row in cells[0]["rows"]]
    for cell in cells:
        rows = cell["rows"]
        if ([(r["seed"], r["seat"]) for r in rows] != expected or
                [r["first"] for r in rows] != first_seats):
            raise ValueError("missing, duplicate or unpaired layout block")
        split_turns = Counter()
        transitions = Counter()
        asymmetric_rail = partner_occupation = 0
        for row in rows:
            old, new = row["branches"]
            if any(type(branch["score"]) is not int or branch["score"] not in (-1, 0, 1, 2)
                   for branch in (old, new)):
                raise ValueError("invalid team result")
            transitions[f'{old["score"]}->{new["score"]}'] += 1
            split = row["split"]
            if not split:
                if old != new:
                    raise ValueError("different branches without a split")
                continue
            if (len(split["moves"]) != 2 or split["moves"][0] == split["moves"][1] or
                    not 1 <= split["turn"] <= split["ply"] + 1 or
                    split["ply"] >= min(old["plies"], new["plies"])):
                raise ValueError("invalid first divergence")
            split_turns[str(split["turn"])] += 1
            markers = split["vacates_partner_rail"]
            if markers in ([1, 0], [0, 1]):
                asymmetric_rail += 1
                offset = markers.index(1)
                reply = row["branches"][offset]["partner"]
                partner_occupation += reply is not None and reply[1] == split["moves"][offset][0]
        result.append(dict(context=cell["context"], transitions=dict(transitions), turns=dict(split_turns),
                           rail_proxy_asymmetric=asymmetric_rail, partner_occupation=partner_occupation))
    return result


def collect(engine, rules, roster, context):
    partner, before, after = context
    profiles = [roster[name] for name in ("candidate", partner, before, after)]
    command = [str(engine), "cross-play-divergence", str(rules), str(START), str(COUNT)]
    command.extend(profile["policy"] for profile in profiles)
    command.extend(str(value) for profile in profiles for value in profile["weights"])
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=50,
                            creationflags=0x40 | 0x08000000)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    for index, name in enumerate(("heuristic", "candidate")):
        comparison = evaluate(engine, rules, roster, name, partner, before, START, COUNT, later_enemy=after)
        for split, ordinary in zip(rows, comparison, strict=True):
            branch = split["branches"][index]
            if ([split[k] for k in ("seed", "seat", "first")] !=
                    [ordinary[k] for k in ("seed", "seat", "first")] or
                    (branch["plies"], branch["score"]) != (ordinary["plies"], SCORE[ordinary["score"]])):
                raise ValueError("natural split did not reproduce ordinary cross-play")
    return dict(context=list(context), rows=rows)


def audit():
    report = json.loads(REPORT.read_text(encoding="utf8"))
    engine, rules = RUN / "junqi.exe", RUN / "ruleset.json"
    if report["hashes"] != dict(engine=sha(engine), rules=sha(rules)):
        raise ValueError("frozen run artifacts changed")
    if report["roster"] != model_roster() or report["summary"] != summarize(report["cells"]):
        raise ValueError("opening split report differs from raw rows")
    # Exercise the omissions that older screens silently accepted.
    cells = report["cells"]
    for broken in (cells[:-1], cells[:-1] + [cells[0]]):
        try:
            summarize(broken)
        except ValueError:
            continue
        raise ValueError("incomplete or duplicate contexts passed audit")
    print("PASS paired first splits, frozen artifact identity, missing/duplicate context rejection")


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("opening split archive exists; use audit")
    roster = model_roster()
    RUN.mkdir()
    engine, rules = RUN / "junqi.exe", RUN / "ruleset.json"
    shutil.copyfile(EXE, engine)
    shutil.copyfile(RULES, rules)
    begin = time.monotonic()
    cells = [collect(engine, rules, roster, context) for context in CONTEXTS]
    report = dict(roster=roster, cells=cells, summary=summarize(cells),
                  hashes=dict(engine=sha(engine), rules=sha(rules)), seconds=time.monotonic()-begin)
    contents = (json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(contents) > 1024**2:
        raise ValueError("opening split output budget exceeded")
    with REPORT.open("xb") as output:
        output.write(contents)
    print("PASS natural branches reproduced ordinary cross-play in every context")


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
