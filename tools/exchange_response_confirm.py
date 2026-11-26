"""Check whether public features can explain repeatable exchange choices."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sys
import time

from camp_control import collect
from exchange_counterplay import REPORT as PREVIOUS, summarize as exchange_summary
from flag_interception import source_keys
from learn_short import EXE, RULES
from partner_crossplay import model_roster, sha

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/"build/public-exchange-response-confirm-v1"
REPORT = ROOT/"docs/REPORTS/public-exchange-response-confirm.json"
START, COUNT = 296000, 256
MODE = "exchange-response-confirm"
CUT = START+COUNT//2


def repeatable_direction(case):
    """Require sign agreement across both worlds and two disjoint seed halves."""
    if case["worlds"] != 2:
        return 0
    signs = []
    for cell in (case["cells"][1], case["cells"][4]):
        for half in (cell[:4], cell[4:]):
            if any(-1 in (a[0], b[0]) for a, b in half):
                return 0
            net = sum(b[0]-a[0] for a, b in half)
            signs.append((net > 0)-(net < 0))
    return signs[0] if len(set(signs)) == 1 else 0


def natural_and_revisit(rows):
    reach, changed_layouts, changed_pair_layouts = Counter(), set(), set()
    for row in rows:
        case = row["case"]
        if case is None:
            continue
        if (len(case["feature_delta"]) != 5 or any(type(x) is not int for x in case["feature_delta"])
                or any(case[key] not in (0, 1) for key in ("guaranteed_interception", "visible_flag_attack"))):
            raise ValueError("invalid observation-derived features")
        reach["visible_flag_attacks"] += case["visible_flag_attack"]
        for index, cell in enumerate(case["cells"]):
            for capture, guard in cell:
                for arm in (capture, guard):
                    if len(arm) != 17 or any(x not in (0, 1) for x in arm[15:]) or arm[16] > arm[5]:
                        raise ValueError("invalid public revisit fields")
                if index != 1:
                    continue
                # Only the original world and enemy pressure, without prioritized ally help.
                reach["capture_paths"] += 1
                reach["ally_played_elsewhere"] += capture[15]
                reach["reached"] += capture[5]
                reach["pressure_changed"] += capture[16]
                reach["changed_and_pair"] += int(capture[16] and capture[6])
                if capture[16]:
                    changed_layouts.add(row["seed"])
                if capture[16] and capture[6]:
                    changed_pair_layouts.add(row["seed"])
    return dict(reach=reach, changed_layouts=len(changed_layouts), changed_pair_layouts=len(changed_pair_layouts))


def representation(rows):
    halves = (defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set)))
    layouts = (defaultdict(set), defaultdict(set))
    missed = (defaultdict(set), defaultdict(set))
    candidates = []
    for row in rows:
        case = row["case"]
        sign = repeatable_direction(case) if case else 0
        if not sign:
            continue
        seed, half = row["seed"], int(row["seed"] >= CUT)
        key, extra = tuple(case["feature_delta"]), case["guaranteed_interception"]
        halves[half][key][extra, sign].add(seed)
        layouts[half][sign].add(seed)
        if (case["guard_share"]-5000)*sign < 0:
            missed[half][sign].add(seed)
        candidates.append(dict(seed=seed, variant=row["variant"], ply=case["ply"],
                               feature_delta=key, guaranteed=extra, direction=sign))
    aliases, resolved = [], []
    for key in sorted(set().union(*halves)):
        left, right = (groups.get(key, {}) for groups in halves)
        positive = set().union(*(v for groups in (left, right) for (_, s), v in groups.items() if s == 1))
        negative = set().union(*(v for groups in (left, right) for (_, s), v in groups.items() if s == -1))
        if not any(a != b for a in positive for b in negative):
            continue
        aliases.append(dict(delta=key, guard_layouts=len(positive), capture_layouts=len(negative)))
        if (set(left) == set(right) == {(1, 1), (0, -1)} and
                all(len(seeds) >= 2 for groups in (left, right) for seeds in groups.values())):
            resolved.append(key)
    # Keep JSON representations equal on audit, and both halves independent by seed.
    for item in (*candidates, *aliases):
        key = "feature_delta" if "feature_delta" in item else "delta"
        item[key] = list(item[key])
    counts = lambda data: [{str(sign): len(part[sign]) for sign in (1, -1)} for part in data]
    return dict(direction_layouts=counts(layouts), missed_layouts=counts(missed),
                five_feature_conflicts=aliases, repeated_interception_discriminators=[list(k) for k in resolved],
                sources=candidates)


def summarize(rows):
    return dict(exchange=exchange_summary(rows, START, COUNT, extended=True),
                revisits=natural_and_revisit(rows), representation=representation(rows))


def frozen():
    return dict(engine=sha(RUN/"junqi.exe"), rules=sha(RUN/"ruleset.json"), parent=model_roster()["candidate"])


def replay(row, parent, mode):
    return collect(RUN/"junqi.exe", RUN/"ruleset.json", parent, row["seed"], row["seat"],
                   row["variant"], mode=mode)


def audit():
    data = json.loads(REPORT.read_text(encoding="utf8"))
    rows = data["sources"]
    if data["frozen"] != frozen() or data["summary"] != summarize(rows):
        raise ValueError("frozen inputs or evidence summary changed")
    first = next((row for row in rows if row["case"]), rows[0])
    old = json.loads(PREVIOUS.read_text(encoding="utf8"))["sources"]
    old_hit = next(row for row in old if row["case"])
    for row, mode in ((first, MODE), (rows[-1], MODE), (old_hit, "exchange-counterplay")):
        if row != replay(row, data["frozen"]["parent"], mode):
            raise ValueError("fixed replay differed")
    print("PASS fresh source grid, public features, independent halves and legacy replay")


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("confirmation archive exists; use audit")
    RUN.mkdir()
    shutil.copyfile(EXE, RUN/"junqi.exe")
    shutil.copyfile(RULES, RUN/"ruleset.json")
    inputs = frozen()
    started = time.monotonic()
    rows = [collect(RUN/"junqi.exe", RUN/"ruleset.json", inputs["parent"], seed, seat, variant, mode=MODE)
            for seed, seat, variant in source_keys(START, COUNT)]
    summary = summarize(rows)
    data = dict(sources=rows, summary=summary, seconds=time.monotonic()-started, frozen=inputs)
    content = (json.dumps(data, separators=(",", ":"), allow_nan=False)+"\n").encode()
    if len(content) > 1024**2:
        raise ValueError("exchange confirmation exceeded its byte cap")
    with REPORT.open("xb") as output:
        output.write(content)
    print("Completed fixed-budget exchange response replication")


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
