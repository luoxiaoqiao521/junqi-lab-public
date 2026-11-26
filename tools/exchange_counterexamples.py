"""Count and replay public one-HQ flag opportunities without changing the policy."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sys
import time

from camp_control import collect
from camp_pairs import score_delta, scores_summary
from exchange_counterplay import PROFILES, SCORES, record_branch
from flag_interception import source_keys
from learn_short import EXE, RULES
from partner_crossplay import model_roster, sha

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/"build/sample-exchange-counterexamples-v1"
REPORT = ROOT/"docs/REPORTS/sample-exchange-counterexamples.json"
ROUTE = ROOT/"docs/REPORTS/public-exchange-next.json"
EVIDENCE = ROOT/"docs/REPORTS/public-exchange-response-confirm.json"
START, COUNT = 297000, 384
MODE = "exchange-counterexamples"
FIELDS = ("decisions", "inference_turns", "attack_turns", "win_turns", "ambiguous_hq_turns",
          "competing_defense_turns", "competing_win_turns")


def pair_summary(row, totals, deltas):
    case, seed = row["case"], row["seed"]
    pair = case["pair"]
    worlds = pair["worlds"]
    if (case["immediate_win"] not in (0, 1) or worlds not in (1, 2)
            or len(pair["cells"]) != 3*worlds or pair["guard"] == pair["capture"]
            or not 0 <= pair["guard_share"] <= 10000):
        raise ValueError("invalid public flag and defense pair")
    totals["paired_source_games"] += 1
    totals["immediate_win_in_first_pair"] += case["immediate_win"]
    for i, cell in enumerate(pair["cells"]):
        if len(cell) != 8:
            raise ValueError("incomplete counterexample cell")
        profile = i%3
        key = f"{i//3}/{PROFILES[profile]}"
        for attack, guard in cell:
            for arm in (attack, guard):
                record_branch(arm, profile, pair["ply"], Counter(), defaultdict(list), seed)
            if case["immediate_win"] and (attack[0] != 2 or attack[5] != 0):
                raise ValueError("a public winning flag attack differed across branches")
            deltas[key][seed].append(score_delta(attack[0], guard[0]))


def summarize(rows):
    if [(r["seed"], r["seat"], r["variant"]) for r in rows] != source_keys(START, COUNT):
        raise ValueError("invalid counterexample source grid")
    contexts = defaultdict(Counter)
    opportunity_layouts, win_layouts, pair_layouts = set(), set(), set()
    deltas = defaultdict(lambda: defaultdict(list))
    for row in rows:
        totals = contexts[str(row["variant"])]
        counts = row["counts"]
        n, info, attack, win, ambiguous, paired, winning_pairs = counts
        share = row["first_win_share"]
        if (row["source_score"] not in SCORES or not 0 <= winning_pairs <= min(paired, win)
                or not 0 <= paired <= attack <= info <= n or not 0 <= win <= attack
                or not 0 <= ambiguous <= n or bool(row["case"]) != bool(paired)
                or (row["case"] and row["case"]["immediate_win"] > winning_pairs)
                or (win and not 0 <= share <= 10000) or (not win and share != -1)):
            raise ValueError("invalid natural flag opportunity counts")
        totals.update(dict(zip(FIELDS, counts, strict=True)))
        totals["games"] += 1
        totals["games_with_attack"] += int(bool(attack))
        totals["games_with_win"] += int(bool(win))
        if attack:
            opportunity_layouts.add(row["seed"])
        if win:
            win_layouts.add(row["seed"])
            totals["first_win_share_total"] += share
        if row["case"]:
            pair_layouts.add(row["seed"])
            pair_summary(row, totals, deltas)
    return dict(contexts={name: dict(value) for name, value in contexts.items()},
                attack_layouts=len(opportunity_layouts), win_layouts=len(win_layouts),
                pair_layouts=len(pair_layouts), attack_minus_guard={k: scores_summary(v) for k, v in deltas.items()})


def frozen():
    route = json.loads(ROUTE.read_text(encoding="utf8"))
    if route["decision"]["next"] != "sample-exchange-counterexamples" or route["report_sha256"] != sha(EVIDENCE):
        raise ValueError("research route does not match the audited prior result")
    return dict(engine=sha(RUN/"junqi.exe"), rules=sha(RUN/"ruleset.json"),
                parent=model_roster()["candidate"], route=sha(ROUTE))


def replay(row, parent):
    return collect(RUN/"junqi.exe", RUN/"ruleset.json", parent, row["seed"], row["seat"],
                   row["variant"], mode=MODE)


def audit():
    data = json.loads(REPORT.read_text(encoding="utf8"))
    rows = data["sources"]
    if data["frozen"] != frozen() or data["summary"] != summarize(rows):
        raise ValueError("counterexample inputs or summary changed")
    attack = next((row for row in rows if row["counts"][2]), rows[0])
    paired = next((row for row in rows if row["case"]), rows[-1])
    for row in (attack, paired, rows[-1]):
        if replay(row, data["frozen"]["parent"]) != row:
            raise ValueError("counterexample replay changed")
    print("PASS frozen route, full source grid, public inference and paired replay")


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("counterexample archive exists; use audit")
    RUN.mkdir()
    shutil.copyfile(EXE, RUN/"junqi.exe")
    shutil.copyfile(RULES, RUN/"ruleset.json")
    inputs = frozen()
    started = time.monotonic()
    rows = [replay(dict(seed=seed, seat=seat, variant=variant), inputs["parent"])
            for seed, seat, variant in source_keys(START, COUNT)]
    data = dict(sources=rows, summary=summarize(rows), seconds=time.monotonic()-started, frozen=inputs)
    content = (json.dumps(data, separators=(",", ":"), allow_nan=False)+"\n").encode()
    if len(content) > 1024**2:
        raise ValueError("counterexample archive exceeded its byte cap")
    with REPORT.open("xb") as output:
        output.write(content)
    print("Completed natural public flag inference and competing-defense coverage")


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
