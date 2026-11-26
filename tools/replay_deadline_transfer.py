"""Measure natural deadline coverage and matched single-seat transfer."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tarfile
import tempfile

from archive_action_screen import layout_group
from audit_qq_archive import CAMPS, INPUT, LIMIT, ROOT, decode, local_records
from audit_replay_deadline import CASES
from compare_qq_rules import VC
from partner_crossplay import model_roster

COUNT = 64
OUT = ROOT / "build/replay-deadline-transfer"
EXE = OUT / "transfer.exe"
SCAN = OUT / "scan.json"
COMPARISON = OUT / "comparison.json"
COVERAGE = ("decisions", "win", "threat", "later")
BASE = ("group", "seat", "table", "first", "seed", "frozen", "plies", "kind", "offers",
        "shortlist_missed", "warning_removed", "verified_win")


def groups(manifest):
    public, anchors, local = {}, set(), set()
    for raw in local_records(manifest["local"]):
        try:
            local.add(layout_group(decode(raw)[1]))
        except ValueError:
            continue
    with tarfile.open(manifest["corpus"], "r:gz") as archive:
        for member in archive:
            name = Path(member.name)
            if not member.isfile() or not 412 <= member.size <= LIMIT or name.name.startswith("._"):
                continue
            try:
                with archive.extractfile(member) as stream:
                    key, layouts, _, _ = decode(stream.read())
            except ValueError:
                continue
            group = layout_group(layouts)
            public[group] = layouts
            if name.stem in CASES:
                anchors.add(group)
    previous_opening = set(random.Random(31415).sample(sorted(public.keys() - local), 192))
    previous_deadline = set(random.Random(208151).sample(sorted(public.keys() - local - anchors), 64))
    pool = sorted(public.keys() - local - anchors - previous_opening - previous_deadline)
    chosen = random.Random(900711).sample(pool, COUNT)
    real = [(f"archive-{group.hex()[:16]}", 900720+i, public[group]) for i, group in enumerate(chosen)]
    generated = [(f"generated-{i}", 900820+i, None) for i in range(COUNT)]
    return real + generated


def compile_probe():
    OUT.mkdir(parents=True, exist_ok=True)
    project = ROOT.as_posix()
    sources = ["core/board.cpp", "core/rules.cpp", "core/state.cpp", "core/layout.cpp",
               "tests/replay_deadline_transfer.cpp"]
    files = " ".join(f'"{project}/{f}"' for f in sources)
    command = (f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /W4 /WX /O2 /MT '
               f'/utf-8 /I"{project}/core" {files} /Fe:transfer.exe')
    subprocess.run(f'cmd.exe /d /s /c "{command}"', cwd=OUT, check=True, timeout=75)


def encode_layout(raw):
    return [rank-2 if rank<5 else 16-rank for i, rank in enumerate(raw)
            if divmod(i, 5) not in CAMPS]


def records(stage, selected, roster):
    with tempfile.TemporaryFile(mode="w+t", encoding="ascii", newline="\n", dir=OUT) as feed:
        print(len(selected), file=feed)
        for name in ("candidate", "historical"):
            print(*roster[name]["weights"], file=feed)
        for group, seed, layouts in selected:
            print(group, seed, file=feed)
            if layouts:
                for raw in layouts:
                    print(*encode_layout(raw), file=feed)
        feed.seek(0)
        process = subprocess.run([EXE, stage], stdin=feed, capture_output=True, text=True, check=True, timeout=125)
    rows = []
    for line in process.stdout.splitlines():
        key, *raw = line.split()
        data = list(map(int, raw))
        if len(data) != 23 + (2 if stage == "compare" else 0):
            raise ValueError("invalid transfer row width")
        row = dict(zip(BASE, [key, *data[:11]], strict=True))
        row["coverage"] = {name:data[11+3*i:14+3*i] for i, name in enumerate(COVERAGE)}
        if stage == "compare":
            row["shortlist"], row["expanded"] = data[-2:]
        rows.append(row)
    return rows


def replace_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")
    temporary.replace(path)


def manifest_and_roster():
    paths = {k: Path(v) for k, v in json.loads(INPUT.read_text(encoding="utf-8-sig")).items()}
    roster = model_roster()
    return groups(paths), roster


def identities(roster):
    return {name:roster[name]["artifact_sha256"] for name in ("candidate", "historical")}


def scan():
    selected, roster = manifest_and_roster()
    compile_probe()
    rows = records("scan", selected, roster)
    check_grid(rows)
    summary = coverage_summary(rows)
    replace_json(SCAN, dict(models=identities(roster), rows=rows, coverage=summary))
    print("PASS independent layouts and natural coverage", json.dumps(summary, separators=(",", ":")))


def compare():
    selected, roster = manifest_and_roster()
    observed = json.loads(SCAN.read_text(encoding="utf-8"))
    rows = records("compare", selected, roster)
    check_same_baseline(observed, rows, identities(roster))
    summary = comparisons(rows)
    replace_json(COMPARISON, dict(scan_sha256=hashlib.sha256(SCAN.read_bytes()).hexdigest(),
                                  models=identities(roster), rows=rows, comparisons=summary))
    print("PASS matched whole games and shortlist ablation")
    for key, finding in summary.items():
        print(key, json.dumps(finding, separators=(",", ":")))


def check_grid(rows):
    by_group = defaultdict(set)
    for row in rows:
        key = (row["seat"],row["table"])
        if key in by_group[row["group"]]:
            raise ValueError("duplicate layout/seat/table")
        by_group[row["group"]].add(key)
    expected = {(seat,table) for seat in range(4) for table in range(2)}
    if (len(rows) != COUNT*16 or Counter(key.split("-")[0] for key in by_group) !=
            {"archive":COUNT, "generated":COUNT} or any(value!=expected for value in by_group.values())):
        raise ValueError("incomplete independent layout grid")
    peers = defaultdict(set)
    for row in rows:
        peers[row["group"],row["seat"]].add((row["first"],row["seed"]))
    if any(len(values)!=1 for values in peers.values()):
        raise ValueError("opening or random streams depend on opponents")


def check_same_baseline(scan_report, rows, models):
    check_grid(rows)
    core = [{k:v for k,v in row.items() if k not in ("expanded", "shortlist")} for row in rows]
    if core != scan_report["rows"] or models != scan_report["models"]:
        raise ValueError("frozen trajectory, selected layouts or model changed between stages")


def strata(rows):
    for source in ("archive", "generated"):
        for table in range(2):
            yield f"{source}/{table}", [row for row in rows if row["group"].startswith(source) and
                                        row["table"]==table]


def coverage_summary(rows):
    return {key:dict(games=len(cell), outcomes=outcomes(cell, "frozen"),
                     windows={label:[sum(r["coverage"][label][i] for r in cell) for i in range(3)]
                              for label in COVERAGE},
                     intervention_blocks=len({r["group"] for r in cell if r["kind"]}),
                     win_blocks=len({r["group"] for r in cell if r["kind"]==1}),
                     guard_blocks=len({r["group"] for r in cell if r["kind"]==2}))
            for key,cell in strata(rows)}


def outcomes(rows, arm):
    counts = Counter(r[arm] for r in rows)
    return {name:counts[score] for score,name in [(2,"win"),(1,"draw"),(0,"loss"),(-1,"cutoff")]}


def score_delta(row, arm):
    base, test = row["frozen"], row[arm]
    if not row["kind"] or (arm == "shortlist" and row["shortlist_missed"]):
        if test!=base:
            raise ValueError("unchanged arm did not preserve result")
        return 0., 0.
    low = (0. if test<0 else test/2) - (1. if base<0 else base/2)
    high = (1. if test<0 else test/2) - (0. if base<0 else base/2)
    return low, high


def block_interval(rows, arm):
    blocks = defaultdict(list)
    for row in rows:
        blocks[row["group"]].append(score_delta(row, arm))
    lo, hi = zip(*(tuple(map(statistics.mean, zip(*pairs))) for pairs in blocks.values()))
    rng = random.Random(901150)
    samples = [rng.choices(range(len(lo)), k=len(lo)) for _ in range(1200)]
    lows = sorted(statistics.mean(lo[i] for i in indices) for indices in samples)
    highs = sorted(statistics.mean(hi[i] for i in indices) for indices in samples)
    return dict(bounds=[statistics.mean(lo), statistics.mean(hi)],
                interval=[lows[29], highs[1169]])


def comparisons(rows):
    result = {}
    for key, cell in strata(rows):
        result[key] = {}
        for arm in ("shortlist","expanded"):
            active = [r for r in cell if r["kind"] and not (arm=="shortlist" and r["shortlist_missed"])]
            result[key][arm] = dict(outcomes=outcomes(cell,arm), **block_interval(cell,arm),
                                    changes=len(active), independent=len({r["group"] for r in active}),
                                    missed_wins=sum(r["kind"]==1 for r in cell if r not in active),
                                    harms=sum(r[arm]>=0 and r["frozen"]>=0 and r[arm]<r["frozen"] for r in active),
                                    guards=sum(r["kind"]==2 for r in active),
                                    warnings_removed=sum(r["warning_removed"] for r in active if arm=="expanded"))
    return result


def audit():
    before = json.loads(SCAN.read_text(encoding="utf-8"))
    after = json.loads(COMPARISON.read_text(encoding="utf-8"))
    check_same_baseline(before, after["rows"], identities(model_roster()))
    if (after["scan_sha256"] != hashlib.sha256(SCAN.read_bytes()).hexdigest() or
            coverage_summary(before["rows"]) != before["coverage"] or
            comparisons(after["rows"]) != after["comparisons"]):
        raise ValueError("receipts do not reproduce")
    for row in after["rows"]:
        if row["kind"]==1 and (row["verified_win"]!=1 or row["expanded"]!=2):
            raise ValueError("confirmed team win failed in referee")
        if any(row[arm] not in (-1,0,1,2) for arm in ("frozen","shortlist","expanded")):
            raise ValueError("invalid outcome")
    print("PASS raw coverage, fixed prerequisites, no-change invariance, and winner verification")


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else scan()
