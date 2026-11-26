"""Paired team-stop and revisit outcomes on the frozen camp-control sources."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import sys
import time

from camp_control import REPORT as SCAN, MIN_LAYOUTS, collect, select_sources, summarize as summarize_scan
from learn_short import EXE, RULES, np
from partner_crossplay import model_roster, sha

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/"build/public-camp-paired-outcomes-v1"
REPORT = ROOT/"docs/REPORTS/public-camp-paired-outcomes.json"
SCORES = (-1, 0, 1, 2)
NAMES = ("score", "next_group", "escape", "flag_exposure", "ally_turn", "ally_contact",
         "target_turn", "pursue_score", "switch_score")
CONTEXTS = (0, 2)
STREAMS = (0, 1)
SAMPLES = 8


def frozen_sources():
    scan = json.loads(SCAN.read_text(encoding="utf8"))
    if scan["summary"] != summarize_scan(scan["development"]) or scan["confirmation_started"]:
        raise ValueError("source receipt was not approved")
    if not scan["summary"]["enough_for_paired_screen"]:
        raise ValueError("insufficient natural layouts for continuation")
    return [(kind, row) for kind in (0, 1) for row in select_sources(scan["development"], kind)]


def score_delta(a, b):
    return ((a if a != -1 else 0)/2 - (b if b != -1 else 2)/2,
            (a if a != -1 else 2)/2 - (b if b != -1 else 0)/2)


def scores_summary(layouts):
    if not layouts:
        return dict(layouts=0, score_bounds=None, exploratory_interval=None)
    per_layout = np.array([np.mean(rows, axis=0) for rows in layouts.values()])
    samples = np.random.default_rng(84012).integers(0, len(per_layout), (4096, len(per_layout)))
    resample = per_layout[samples].mean(axis=1)
    return dict(layouts=len(per_layout), score_bounds=per_layout.mean(axis=0).tolist(),
                exploratory_interval=[float(np.quantile(resample[:, 0], .025)),
                                      float(np.quantile(resample[:, 1], .975))])


def validate_case(case, kind, source):
    if (case["kind"] != kind or
            any(case[key] != source[key] for key in ("seed", "seat", "variant")) or
            case["point"] != source["points"][kind]):
        raise ValueError("continuation altered the frozen public case")
    worlds = case["worlds"]
    if worlds not in (1, 2):
        raise ValueError("invalid number of information-compatible worlds")
    keys = [(c["world"], c["response"], c["stream"]) for c in case["cells"]]
    expected = [(world, response, stream) for world in range(worlds)
                for response in CONTEXTS for stream in STREAMS]
    if keys != expected or any(len(c["pairs"]) != SAMPLES for c in case["cells"]):
        raise ValueError("missing world, response or stream in paired continuation")
    for cell in case["cells"]:
        for a, b in cell["pairs"]:
            for record in (a, b):
                if (len(record) != len(NAMES) or record[0] not in SCORES or record[1] not in range(4)
                        or not all(flag in (0, 1) for flag in record[2:7])
                        or record[7] not in SCORES or record[8] not in SCORES
                        or (record[1] < 2 and record[7:] != [-1, -1])):
                    raise ValueError("invalid first or second decision receipt")


def aggregate_group(cases):
    first = defaultdict(lambda: defaultdict(list))
    second = defaultdict(lambda: defaultdict(list))
    events = defaultdict(Counter)
    for case in cases:
        for cell in case["cells"]:
            key = f'{cell["world"]}/{cell["response"]}/{cell["stream"]}'
            for a, b in cell["pairs"]:
                first[key][case["seed"]].append(score_delta(a[0], b[0]))
                for label, row in (("seal", a), ("press", b)):
                    event = events[label]
                    event["rollouts"] += 1
                    event[f"first_score_{row[0]}"] += 1
                    event[f"next_group_{row[1]}"] += 1
                    event.update({name: count for name, count in zip(NAMES[2:7], row[2:7], strict=True)})
                    if row[1] >= 2:
                        second[f"{label}/{row[1]}"][case["seed"]].append(score_delta(row[8], row[7]))
    return dict(cases=len(cases), compatible_layouts=sum(case["worlds"] == 2 for case in cases),
                first={key: scores_summary(layouts) for key, layouts in first.items()},
                second={key: scores_summary(layouts) for key, layouts in second.items()},
                events={key: dict(values) for key, values in events.items()})


def direction(interval):
    if interval[0] > 0:
        return 1
    if interval[1] < 0:
        return -1
    return 0


def eligible_for_confirmation(groups):
    if any(g["compatible_layouts"] < MIN_LAYOUTS for g in groups):
        return False
    # The developer sample is a screen. Independent layouts remain untouched
    # unless the declared primary both shows and repeats its conditional flip.
    directions = [{direction(g["first"][f"0/0/{stream}"]["exploratory_interval"])
                   for stream in STREAMS} for g in groups]
    if directions not in ([{1}, {-1}], [{-1}, {1}]):
        return False
    for group, primary in zip(groups, directions, strict=True):
        desired = next(iter(primary))
        if any(direction(c["exploratory_interval"]) == -desired for c in group["first"].values()):
            return False
    return True


def summarize(cases, selections):
    for case, (kind, source) in zip(cases, selections, strict=True):
        validate_case(case, kind, source)
    groups = [aggregate_group([c for c in cases if c["kind"] == kind]) for kind in (0, 1)]
    return dict(groups=groups, eligible_for_confirmation=eligible_for_confirmation(groups))


def audit():
    report = json.loads(REPORT.read_text(encoding="utf8"))
    selections = frozen_sources()
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    parent = model_roster()["candidate"]
    if report["frozen"] != dict(engine=sha(engine), rules=sha(rules), parent=parent, source=sha(SCAN)):
        raise ValueError("paired camp artifacts changed")
    cases = report["cases"]
    if report["summary"] != summarize(cases, selections) or report["confirmation_started"]:
        raise ValueError("paired outcomes changed or holdout was opened")
    bad = dict(cases[0], cells=cases[0]["cells"][:-1])
    try:
        validate_case(bad, *selections[0])
    except ValueError:
        pass
    else:
        raise ValueError("missing continuation context passed validation")
    for case in (cases[0], cases[-1]):
        if case != collect(engine, rules, parent, case["seed"], case["seat"], case["variant"],
                           mode="camp-control-pair", extra=(case["kind"],)):
            raise ValueError("fixed branch comparison did not reproduce")
    print("PASS public source replay, matched trials, context omissions and sealed confirmation")


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("paired camp archive exists; use audit")
    selections = frozen_sources()
    parent = model_roster()["candidate"]
    RUN.mkdir()
    engine, rules = RUN/"junqi.exe", RUN/"ruleset.json"
    shutil.copyfile(EXE, engine)
    shutil.copyfile(RULES, rules)
    started = time.monotonic()
    cases = []
    for kind, row in selections:
        cases.append(collect(engine, rules, parent, row["seed"], row["seat"], row["variant"],
                             mode="camp-control-pair", extra=(kind,)))
        if len(cases) % 8 == 0:
            print(f"Completed {len(cases)}/{len(selections)} camp sources", flush=True)
    report = dict(cases=cases, summary=summarize(cases, selections), confirmation_started=False,
                  seconds=time.monotonic()-started,
                  frozen=dict(engine=sha(engine), rules=sha(rules), parent=parent, source=sha(SCAN)))
    data = (json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(data) > 2*1024**2:
        raise ValueError("paired-camp data exceeded its byte cap")
    with REPORT.open("xb") as output:
        output.write(data)


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
