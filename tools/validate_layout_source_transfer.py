"""Measure which simulated outcomes change when only the focal layout source changes."""
from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from random import Random
from statistics import mean
import subprocess
import sys
import tempfile

from archive_action_screen import select
from audit_annotated_replays import COLLECTIONS, DOWNLOADS, layout_audit, layout_files, mirror
from audit_public_attack_windows import write_json
from audit_qq_archive import CAMPS, INPUT, ROOT, local_records, public_records
from compare_qq_rules import VC
from partner_crossplay import model_roster
from replay_deadline_transfer import encode_layout

OUT = ROOT / "build/layout-source-transfer"
EXE = OUT / "layout_source_transfer.exe"
RESULT = OUT / "matched.json"
COUNT = 128
SLOTS = tuple(i for i in range(30) if divmod(i, 5) not in CAMPS)
SWAPS = tuple(combinations(SLOTS, 2))
ARMS = ("generated", "recorded", "collection")
METRICS = ("block", "source", "orientation", "table", "seat", "first", "seed", "score", "plies", "first_contact",
           "contacts", "first_flag", "flag_front", "no_moves", "own_no_move", "own_flag_taken")


def canonical(layout):
    return min(layout, mirror(layout))


def too_close(layout, existing):
    """Same seat-local board after mirror or one exchange of unequal ranks."""
    if canonical(layout) in existing:
        return True
    for orientation in (layout, mirror(layout)):
        trial = bytearray(orientation)
        for i, j in SWAPS:
            if trial[i] == trial[j]:
                continue
            trial[i], trial[j] = trial[j], trial[i]
            if bytes(trial) in existing:
                return True
            trial[i], trial[j] = trial[j], trial[i]
    return False


def previously_sampled(public, local):
    excluded = {key.hex()[:16] for key in local}
    excluded.update(key.hex()[:16] for key in Random(31415).sample(sorted(public.keys() - local), 192))
    for file in ("flag-front-defense", "defense-opportunity-cost", "pre-flag-route-defense"):
        artifact = ROOT / "build/attack-opportunity" / f"{file}.json"
        data = json.loads(artifact.read_text(encoding="utf-8"))
        cases = list(data.get("decisions", data.get("data", {}))) + data.get("absent", [])
        excluded.update(case.split("-")[1] for case in cases)
    scan = ROOT / "build/replay-deadline-transfer/scan.json"
    excluded.update(row["group"].split("-")[1] for row in json.loads(scan.read_text(encoding="utf-8"))["rows"]
                    if row["group"].startswith("archive-"))
    return excluded


def records():
    manifest = {name: Path(path) for name, path in json.loads(INPUT.read_text(encoding="utf-8-sig")).items()}
    public = select(public_records(manifest["corpus"]))
    local = set(select(local_records(manifest["local"])))
    excluded = previously_sampled(public, local)
    keys = sorted(public)
    Random(635211).shuffle(keys)
    raw_replay_families = {canonical(layout) for row in public.values() for layout in row[1]}
    chosen, ids, used = [], [], set()
    for key in keys:
        if key.hex()[:16] in excluded:
            continue
        layout = canonical(public[key][1][len(chosen)%4])
        if too_close(layout, used):
            continue
        chosen.append(layout)
        ids.append(key.hex()[:16])
        used.add(layout)
        if len(chosen) == COUNT:
            break
    if len(chosen) != COUNT:
        raise ValueError("too few independent replay layouts")
    return chosen, ids, raw_replay_families, len(public), len(excluded)


def freeze_inputs():
    recorded, ids, replay_families, available, excluded = records()
    paths = [DOWNLOADS / name for name in COLLECTIONS]
    _, library = layout_audit(raw for path in paths for raw in layout_files(path))
    pool = sorted(library - replay_families)
    Random(635212).shuffle(pool)
    used = set(recorded)
    collected = []
    rejected_near = 0
    for layout in pool:
        if too_close(layout, replay_families) or too_close(layout, used):
            rejected_near += 1
            continue
        used.add(layout)
        collected.append(layout)
        if len(collected) == COUNT:
            break
    if len(collected) != COUNT:
        raise ValueError("too few independent collection layouts")
    # Validate the actual selection again, including cross-source near duplicates.
    prior = set()
    for layout in recorded + collected:
        if too_close(layout, prior):
            raise ValueError("overlapping selected layout families")
        prior.add(layout)
    source_info = dict(public_board_groups=available, public_excluded_groups=excluded,
                       archive_seat_families=len(replay_families), library_families=len(library),
                       library_exact_overlap=len(library & replay_families), rejected_near_in_draw=rejected_near,
                       archive_groups=ids, selected_focal_hashes=[sha256(layout).hexdigest() for layout in
                                                                 recorded+collected])
    return list(zip(recorded, collected, strict=True)), source_info


def compile_probe():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = ("core/board.cpp", "core/rules.cpp", "core/state.cpp", "core/layout.cpp",
             "tests/layout_source_transfer.cpp")
    units = " ".join(f'"{ROOT.as_posix()}/{p}"' for p in paths)
    cmd = f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /W4 /WX /O2 /MT /utf-8 /I"{ROOT.as_posix()}/core" {units} /Fe:{EXE.name}'
    subprocess.run(f'cmd.exe /d /s /c "{cmd}"', cwd=OUT, check=True, timeout=70)


def outcomes(rows):
    counts = Counter(row["score"] for row in rows)
    return dict(win=counts[2], draw=counts[1], loss=counts[0], cutoff=counts[-1],
                team_bounds=[mean(max(0, row["score"])/2 for row in rows),
                             mean((row["score"] if row["score"]>=0 else 2)/2 for row in rows)],
                plies=mean(row["plies"] for row in rows),
                contacts_per_100_plies=100*sum(row["contacts"] for row in rows)/sum(row["plies"] for row in rows),
                first_events={k:dict(games=sum(row[k]>0 for row in rows),
                                     mean_ply=mean([row[k] for row in rows if row[k]>0] or [0]))
                              for k in ("first_contact", "first_flag", "flag_front")},
                any_no_move=sum(row["no_moves"]>0 for row in rows),
                own_no_move=sum(row["own_no_move"] for row in rows),
                own_flag_taken=sum(row["own_flag_taken"] for row in rows))


def stat_bounds(test, base):
    return (max(test, 0)-(base if base>=0 else 2))/2, ((test if test>=0 else 2)-max(base, 0))/2


def interval(blocks):
    assert len(blocks) == COUNT
    rng = Random(635213)
    sample = [rng.choices(blocks, k=len(blocks)) for _ in range(1200)]
    lo = sorted(mean(x[0] for x in batch) for batch in sample)
    hi = sorted(mean(x[1] for x in batch) for batch in sample)
    return [lo[29], hi[1169]]


def combine_orientations(values):
    return [tuple(map(mean, zip(a,b,strict=True))) for a,b in zip(values[::2],values[1::2],strict=True)]


def summarize(rows):
    result = {}
    for table in (0, 1):
        cols = [[r for r in rows if r["source"] == s and r["table"] == table] for s in range(3)]
        arms = dict(zip(ARMS, map(outcomes, cols), strict=True))
        deltas = {}
        for test, base in ((1,0), (2,0), (2,1)):
            pairs = list(zip(cols[test], cols[base], strict=True))
            bounds = [stat_bounds(a["score"], b["score"]) for a, b in pairs]
            scores = dict(bounds=list(map(mean, zip(*bounds))), interval=interval(combine_orientations(bounds)),
                          unknown=sum(min(a["score"], b["score"])<0 for a,b in pairs))
            coverage = {}
            for key in ("first_contact", "first_flag", "flag_front", "no_moves"):
                shifts = [int(a[key]>0)-int(b[key]>0) for a,b in pairs]
                coverage[key] = dict(change=mean(shifts), interval=interval(combine_orientations([(x,x) for x in shifts])))
            deltas[f"{ARMS[test]}_minus_{ARMS[base]}"] = dict(team_score=scores, coverage=coverage)
        result[str(table)] = dict(arms=arms, deltas=deltas)
    return result


def run():
    selected, info = freeze_inputs()
    roster = model_roster()["candidate"]
    with tempfile.TemporaryFile(mode="w+t", encoding="ascii", newline="\n", dir=OUT) as feed:
        print(COUNT, file=feed)
        print(*roster["weights"], file=feed)
        for index, pair in enumerate(selected):
            print(635800+index, file=feed)
            for layout in pair:
                print(*encode_layout(layout), file=feed)
        feed.seek(0)
        output = subprocess.run([EXE], stdin=feed, check=True, capture_output=True, text=True, timeout=110)
    rows = [dict(zip(METRICS, map(int, line.split()), strict=True)) for line in output.stdout.splitlines()]
    write_json(RESULT,dict(source=info, model=roster["artifact_sha256"], rows=rows, summary=summarize(rows)))
    audit(info)


def audit(info=None):
    if info is None:
        _, info = freeze_inputs()
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    assert data["source"] == info and len(set(info["selected_focal_hashes"])) == 2*COUNT
    assert data["model"] == model_roster()["candidate"]["artifact_sha256"]
    rows = data["rows"]
    dispatch = [(b,s,o,t) for b in range(COUNT) for s in range(3) for o in range(2) for t in range(2)]
    assert [(r["block"],r["source"],r["orientation"],r["table"]) for r in rows] == dispatch
    for r in rows:
        b = r["block"]
        assert (r["seat"],r["first"],r["seed"]) == (b%4, b//4%4, 635800+b)
        assert r["score"] in (-1,0,1,2) and 0<r["plies"]<=800
        assert (r["score"]!=-1 or r["plies"]==800) and 0<=r["contacts"]<=r["plies"]
        assert all(0<=r[k]<=r["plies"] for k in ("first_contact", "first_flag", "flag_front"))
        assert (r["first_contact"]==0) == (r["contacts"]==0)
        assert r["own_no_move"]+r["own_flag_taken"]<=1
    assert data["summary"] == summarize(rows)
    print("PASS matched source games", len(rows), "archive/collection layout blocks", COUNT)
    print(json.dumps(data["summary"], sort_keys=True))


if __name__ == "__main__":
    if sys.argv[1:] == ["audit"]:
        audit()
    else:
        compile_probe()
        run()
