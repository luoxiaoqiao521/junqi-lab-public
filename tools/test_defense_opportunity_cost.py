"""Compare defender options from visible history on new replay layout groups."""
from collections import Counter, defaultdict
import json
from pathlib import Path
from random import Random
from statistics import mean
import subprocess
import sys
import tempfile

from archive_action_screen import layout_group
from audit_annotated_replays import COLLECTIONS, DOWNLOADS, annotated_replays, nested_sources
from audit_public_attack_windows import OUT, attack_outcome, write_json
from audit_qq_archive import INPUT, ROOT, decode, public_records
from compare_qq_rules import VC
from measure_flag_front_defense import RESULT as PRIOR, window_prefix
from partner_crossplay import model_roster

RESULT = OUT / "defense-opportunity-cost.json"
EXE = OUT / "defense_opportunity_cost.exe"
ARMS = ("frozen", "recorded", "intercept", "route", "finish")
COUNT = 24


def new_sources():
    folder = DOWNLOADS / COLLECTIONS[1]
    _, _, rar = nested_sources(folder)
    selected = annotated_replays()
    loose = [path.read_bytes() for path in sorted(folder.rglob("*.jgs"))]
    return selected + loose + rar


def select():
    previous = json.loads(PRIOR.read_text(encoding="utf-8"))
    excluded = {key.split("-")[1] for key in list(previous["data"]) + previous["absent"]}
    buckets = defaultdict(dict)
    seen = set()
    for raw in new_sources():
        record = decode(raw)
        shape = layout_group(record[1])
        if shape in seen or shape.hex()[:16] in excluded:
            raise ValueError("new source layout already sampled")
        seen.add(shape)
        yield "new-" + shape.hex()[:16], record, attack_outcome(record)
    sources = json.loads(INPUT.read_text(encoding="utf-8-sig"))
    for raw in public_records(Path(sources["corpus"])):
        try:
            record = decode(raw)
        except ValueError:
            continue
        shape = layout_group(record[1])
        if shape.hex()[:16] in excluded or shape in seen:
            continue
        window = attack_outcome(record)
        if window:
            groups = buckets[window["enemies_left"]]
            groups.setdefault(shape, (record, window))
    for enemies in (1, 2):
        pool = buckets[enemies]
        available = sorted(pool.keys()-seen)
        for shape in Random(935442+enemies).sample(available, COUNT):
            seen.add(shape)
            yield f"qq{enemies}-" + shape.hex()[:16], *pool[shape]


def compile_probe():
    paths = ("core/board.cpp", "core/rules.cpp", "core/state.cpp", "core/layout.cpp",
             "tests/defense_opportunity_cost.cpp")
    units = " ".join(f'"{ROOT.as_posix()}/{p}"' for p in paths)
    command = f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /W4 /WX /O2 /MT /utf-8 /I"{ROOT.as_posix()}/core" {units} /Fe:{EXE.name}'
    subprocess.run(f'cmd.exe /d /s /c "{command}"', cwd=OUT, check=True, timeout=75)


def parse(output):
    decisions, pairs = {}, defaultdict(list)
    for line in output.splitlines():
        kind, group, *args = line.split()
        values = list(map(int, args))
        if kind == "F" and group == "1":
            continue
        if kind == "X":
            decisions[group] = None
        elif kind == "D":
            prefix, actions = values[:5], values[5:]
            decisions[group] = [prefix, [actions[i:i+4] for i in range(0, len(actions), 4)]]
        elif kind == "P":
            pairs[group].append(values)
        else:
            raise ValueError("unexpected probe output")
    return decisions, dict(pairs)


def delta(observed, fallback):
    lo = 0 if observed < 0 else observed
    hi = 2 if observed < 0 else observed
    return (lo-(2 if fallback < 0 else fallback))/2, (hi-max(fallback, 0))/2


def summarize(decisions, pairs):
    result = {}
    for source in ("new", "qq1", "qq2"):
        rows = {key: row for key, row in decisions.items() if key.startswith(source+"-")}
        screens = {key: row for key, row in rows.items() if isinstance(row, list)}
        r = dict(sources=len(rows), no_window=list(rows.values()).count("no-window"),
                 no_turn=list(rows.values()).count("no-turn"),
                 strict_rejected=list(rows.values()).count(None), aligned=len(screens),
                 immediate_threat=sum(row[0][3]>0 for row in screens.values()))
        r["options"] = {}
        for i, name in enumerate(ARMS[1:], 1):
            r["options"][name] = profile(i, screens, pairs)
        result[source] = r
    return result


def profile(i, screens, pairs):
    eligible = {key: actions[i] for key, (_, actions) in screens.items() if actions[i][0]>=0}
    changed = [key for key, action in eligible.items() if action[:2]!=screens[key][1][0][:2]]
    runs = {}
    for variant in (0, 2):
        matches = [[(row[i+2], row[2]) for row in pairs.get(key, []) if row[0]==variant] for key in changed]
        batches = [batch for batch in matches if batch]
        bounds = [list(map(mean, zip(*(delta(a,b) for a,b in batch)))) for batch in batches]
        values = [(a,b) for batch in batches for a,b in batch]
        runs[str(variant)] = dict(blocks=len(batches), pairs=len(values), unknown=sum(min(a,b)<0 for a,b in values),
                                  improved=sum(min(a,b)>=0 and a>b for a,b in values),
                                  harmed=sum(min(a,b)>=0 and a<b for a,b in values),
                                  bounds=list(map(mean, zip(*bounds))) if bounds else None)
    return dict(available=len(eligible), overrides=len(changed),
                baselines_at_risk=sum(screens[key][1][0][3]>0 for key in changed),
                after_at_risk=sum(eligible[key][3]>0 for key in changed), rollouts=runs)


def run():
    decisions = {}
    cases = []
    roster = model_roster()["candidate"]
    for key, record, window in select():
        if window is None:
            decisions[key] = "no-window"
            continue
        prefix = window_prefix(record, window)
        if not prefix:
            decisions[key] = "no-turn"
            continue
        cases.append((key, prefix))
    with tempfile.TemporaryFile(mode="w+t", encoding="ascii", newline="\n", dir=OUT) as sink:
        print(len(cases), file=sink)
        print(*roster["weights"], file=sink)
        for key, (board, rows) in cases:
            # window_prefix currently admits only clockwise archive rows in this encoding.
            print(key, len(rows), 0, file=sink)
            for item in board+rows:
                print(*item, file=sink)
        sink.seek(0)
        output = subprocess.run([EXE], stdin=sink, capture_output=True, text=True, check=True, timeout=100)
    aligned, pairs = parse(output.stdout)
    if set(aligned) != {key for key, _ in cases}:
        raise ValueError("probe coverage differs from dispatched cases")
    decisions.update(aligned)
    write_json(RESULT, dict(model=roster["artifact_sha256"], decisions=decisions, pairs=pairs,
                           summary=summarize(decisions,pairs)))
    audit()


def audit():
    r = json.loads(RESULT.read_text(encoding="utf-8"))
    d, pairs = r["decisions"], r["pairs"]
    assert r["model"] == model_roster()["candidate"]["artifact_sha256"]
    assert Counter(key.split("-")[0] for key in d) == {"new":21, "qq1":COUNT, "qq2":COUNT}
    assert len({key.split("-")[1] for key in d}) == len(d)
    for key, row in d.items():
        if not isinstance(row, list):
            assert key not in pairs
            continue
        assert len(row[1]) == 5 and all(len(action)==4 for action in row[1])
        actions = row[1]
        target = {(arm, seed) for arm in (0,2) for seed in range(12)}
        trials = pairs.get(key, [])
        requires = row[0][3]>0 and any(a[0]>=0 and a[:2]!=actions[0][:2] for a in actions[2:])
        assert len(trials)==(24 if requires else 0)
        if requires:
            assert {(arm, seed) for arm, seed, *_ in trials}==target
        for _, _, *scores in trials:
            assert len(scores)==5
            for i, score in enumerate(scores):
                assert (score==-2 if actions[i][0]<0 else score in (-1,0,1,2))
                if actions[i][:2]==actions[0][:2]:
                    assert score==scores[0]
            if actions[4][0]>=0:
                assert scores[4]==2, "a publicly certain last flag failed referee verification"
    assert r["summary"] == summarize(d, pairs)
    print(json.dumps(r["summary"], sort_keys=True))


if __name__ == "__main__":
    if sys.argv[1:] == ["audit"]:
        audit()
    else:
        compile_probe()
        run()
