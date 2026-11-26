"""Paired flag-defense decisions, with one frozen response change at a time."""
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
RUN = ROOT/"build/public-exchange-counterplay-v1"
REPORT = ROOT/"docs/REPORTS/public-exchange-counterplay.json"
START, COUNT, SAMPLES = 294000, 192, 8
SCORES = (-1, 0, 1, 2)
PROFILES = ("ordinary", "enemy_pressure", "enemy_pressure_ally_defense")


def record_branch(arm, profile, ply, events, next_deltas, seed):
    (score, lost_flag, enemy_used, ally_used, ally_hit_threat, next_turn, next_pair,
     next_guard_share, next_capture, next_guard, won_contact, changed,
     enemy_window, ally_window, next_window) = arm
    flags = (lost_flag, enemy_used, ally_used, ally_hit_threat, next_turn, next_pair, won_contact, changed)
    if (score not in SCORES or any(value not in (0, 1) for value in flags)
            or (enemy_used and profile == 0) or (ally_used and profile != 2)
            or ally_hit_threat > ally_used or changed > next_pair or next_pair > next_turn
            or next_turn != (next_window != -1)
            or any(value != -1 and not ply < value <= ply+4
                   for value in (enemy_window, ally_window, next_window))
            or (enemy_window != -1 and next_turn and enemy_window >= next_window)
            or (ally_window != -1 and next_turn and ally_window >= next_window)
            or (enemy_window != -1 and ally_window != -1 and ally_window >= enemy_window)
            or (ally_used and ally_window == -1)):
        raise ValueError("invalid response or decision window")
    events["games"] += 1
    events[f"score_{score}"] += 1
    for key, count in zip(
            ("flag_lost", "enemy_used", "ally_used", "ally_hit_threat",
             "next_turn", "next_pair", "won_contact", "changed_threat",
             "enemy_window", "ally_window", "no_rescue_flag_lost"),
            (*flags, enemy_window != -1, ally_window != -1, lost_flag and not ally_hit_threat),
            strict=True):
        events[key] += int(count)
    if not next_pair:
        if (next_guard_share, next_capture, next_guard) != (-1, -1, -1):
            raise ValueError("found an unrecorded second decision")
        return
    if not 0 <= next_guard_share <= 10000 or next_capture not in SCORES or next_guard not in SCORES:
        raise ValueError("invalid second decision")
    delta = score_delta(next_guard, next_capture)
    next_deltas[seed].append(delta)
    events["next_guard_ahead"] += int(delta[0] > 0)
    events["next_capture_ahead"] += int(delta[1] < 0)
    events["next_prefers_guard"] += int(next_guard_share > 5000)
    events["next_prefers_guard_when_ahead"] += int(delta[0] > 0 and next_guard_share > 5000)
    events["next_prefers_capture_when_ahead"] += int(delta[1] < 0 and next_guard_share < 5000)
    events["changed_guard_ahead"] += int(changed and delta[0] > 0)
    events["changed_capture_ahead"] += int(changed and delta[1] < 0)
    events["next_capture_ahead_without_ally"] += int(not ally_hit_threat and delta[1] < 0)


def summarize(rows, start=START, count=COUNT, extended=False):
    if [(r["seed"], r["seat"], r["variant"]) for r in rows] != source_keys(start, count):
        raise ValueError("invalid exchange source grid")
    sources = defaultdict(Counter)
    cases, compatible = set(), set()
    comparisons, interactions, next_deltas = (defaultdict(lambda: defaultdict(list)) for _ in range(3))
    events = defaultdict(Counter)
    signals = Counter()
    for row in rows:
        seed, source = row["seed"], sources[str(row["variant"])]
        if row["source_score"] not in SCORES:
            raise ValueError("invalid source result")
        source[f"score_{row['source_score']}"] += 1
        case = row["case"]
        source["offer" if case else "no_offer"] += 1
        if not case:
            continue
        cases.add(seed)
        worlds, cells, ply = case["worlds"], case["cells"], case["ply"]
        if (worlds not in (1, 2) or len(cells) != len(PROFILES)*worlds or
                any(len(cell) != SAMPLES for cell in cells) or case["capture"] == case["guard"] or
                not 0 <= case["guard_share"] <= 10000 or not 0 <= ply):
            raise ValueError("incomplete paired branch grid")
        if worlds == 2:
            compatible.add(seed)
        case_deltas = []
        for w in range(worlds):
            world_deltas = []
            for p, name in enumerate(PROFILES):
                key = f"{w}/{name}"
                deltas = []
                for capture, guard in cells[w*len(PROFILES)+p]:
                    delta = score_delta(guard[0], capture[0])
                    deltas.append(delta)
                    for label, arm in (("capture", capture), ("guard", guard)):
                        arm_key = f"{key}/{label}"
                        record_branch(arm[:15] if extended else arm, p, ply, events[arm_key],
                                      next_deltas[arm_key], seed)
                comparisons[key][seed].extend(deltas)
                world_deltas.append(deltas)
            ordinary, pressure, assistance = world_deltas
            case_deltas.extend(pressure)
            for name, other in (("effect_of_enemy", ordinary), ("effect_of_ally", assistance)):
                for a, b in zip(pressure, other, strict=True):
                    interactions[f"{w}/{name}"][seed].append((a[0]-b[1], a[1]-b[0]))
        guard_ahead = sum(d[0] for d in case_deltas) > 0
        capture_ahead = sum(d[1] for d in case_deltas) < 0
        signals["cases_guard_ahead_under_pressure"] += int(guard_ahead)
        signals["cases_capture_ahead_under_pressure"] += int(capture_ahead)
        signals["cases_prefers_capture_when_guard_ahead"] += int(guard_ahead and case["guard_share"] < 5000)
        signals["cases_prefers_guard_when_capture_ahead"] += int(capture_ahead and case["guard_share"] > 5000)
    return dict(sources={k: dict(v) for k, v in sources.items()}, layouts=len(cases),
                compatible_layouts=len(compatible), signals=dict(signals),
                guard_minus_capture={k: scores_summary(v) for k, v in comparisons.items()},
                pressure_minus_other={k: scores_summary(v) for k, v in interactions.items()},
                next_guard_minus_capture={k: scores_summary(v) for k, v in next_deltas.items()},
                events={k: dict(v) for k, v in events.items()})


def audit():
    data = json.loads(REPORT.read_text(encoding="utf8"))
    parent = model_roster()["candidate"]
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    if data["frozen"] != dict(engine=sha(engine), rules=sha(rules), parent=parent):
        raise ValueError("frozen exchange inputs changed")
    rows = data["sources"]
    if data["summary"] != summarize(rows):
        raise ValueError("exchange summary differed")
    first = next((row for row in rows if row["case"]), rows[0])
    for row in (first, rows[-1]):
        replay = collect(engine, rules, parent, row["seed"], row["seat"], row["variant"],
                         mode="exchange-counterplay")
        if row != replay:
            raise ValueError("fixed exchange replay differed")
    print("PASS source and branch coverage, legal-response timing, score bounds and fixed replay")


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("exchange archive exists; use audit")
    parent = model_roster()["candidate"]
    RUN.mkdir()
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    shutil.copyfile(EXE, engine)
    shutil.copyfile(RULES, rules)
    started = time.monotonic()
    rows = [collect(engine, rules, parent, seed, seat, variant, mode="exchange-counterplay")
            for seed, seat, variant in source_keys(START, COUNT)]
    data = dict(sources=rows, summary=summarize(rows), seconds=time.monotonic()-started,
                frozen=dict(engine=sha(engine), rules=sha(rules), parent=parent))
    content = (json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(content) > 1024**2:
        raise ValueError("exchange result exceeded its byte cap")
    with REPORT.open("xb") as output:
        output.write(content)
    print("Completed paired capture and defense decisions under isolated responses")


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
