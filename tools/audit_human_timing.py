"""Audit publicly exposed flag timings in local JGS recordings."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time

from archive_action_screen import layout_group
from audit_qq_archive import LIMIT, ROOT, SEAT_BLOCKS, decode, public_records
from compare_qq_rules import VC

OUT = ROOT / "build/human-timing"
EXE = OUT / "human_timing.exe"
FIELDS = ("ply", "source_event", "open_from", "open_to", "opener", "receiver", "origin", "target",
          "origin_last_moved", "visible_to_opener", "decision", "cutoff", "enemy_actions", "first_close")


def sanitized_tape(raw, events):
    result = bytearray(412)
    result[:10] = raw[:10]
    result[26:28] = raw[26:28]
    for block in SEAT_BLOCKS:
        result[block-28] = raw[block-28]
        result[block:block+30] = raw[block:block+30]
    for event in events:
        result.extend(event if event[0] == 95 else event[:5] + bytes(5))
    return result.hex()


def old_layouts(index):
    if index is None:
        return set()
    path = Path(json.loads(index.read_text(encoding="utf-8-sig"))["corpus"])
    layouts = set()
    for raw in public_records(path):
        try:
            layouts.add(layout_group(decode(raw)[1]))
        except ValueError:
            continue
    return layouts


def gather(roots, feed, prior):
    games, tallies = {}, {}
    for source, root in roots.items():
        counts = Counter()
        tallies[source] = counts
        if not root.is_dir():
            raise ValueError(f"missing source directory: {source}")
        for file in sorted(root.rglob("*")):
            if file.suffix.lower() != ".jgs" or not file.is_file():
                continue
            counts["files"] += 1
            if file.stat().st_size > LIMIT:
                counts["oversize"] += 1
                continue
            raw = file.read_bytes()
            try:
                key, layouts, _, events = decode(raw)
            except ValueError as error:
                counts[f"rejected_{error}"] += 1
                continue
            if key in games:
                counts["duplicate"] += 1
                continue
            family = layout_group(layouts)
            held = int.from_bytes(family[:4], "big") % 4 == 0
            split = "historical_overlap" if family in prior else "holdout" if held else "development"
            counts["historical_layout_overlap"] += family in prior
            games[key] = dict(source=source, family=family.hex(), split=split)
            print(key, sanitized_tape(raw, events), file=feed)
    return games, tallies


def compile_probe():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ("core/board.cpp", "core/layout.cpp", "core/rules.cpp", "core/state.cpp", "tests/human_timing_probe.cpp")
    files = " ".join(f'"{ROOT.as_posix()}/{name}"' for name in source)
    command = (f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /O2 /MT /W4 /WX /utf-8 '
               f'/I"{ROOT.as_posix()}/core" {files} /Fe:human_timing.exe')
    subprocess.run(f'cmd.exe /d /s /c "{command}"', cwd=OUT, check=True, timeout=90)


def read_results(output, games, tallies):
    windows, accounted, kinds = [], set(), defaultdict(Counter)
    for line in output.splitlines():
        tag, key, *fields = line.split()
        game = games[key]
        if tag in ("G", "R"):
            if key in accounted:
                raise ValueError("duplicate game status")
            accounted.add(key)
            if tag == "G":
                tallies[game["source"]]["aligned_games"] += 1
                tallies[game["source"]]["aligned_plies"] += int(fields[0])
            else:
                tallies[game["source"]]["no_unique_legal_trace"] += 1
            continue
        if tag != "W" or key not in accounted:
            raise ValueError("unexpected probe record")
        kind, status, *numbers = fields
        item = dict(zip(FIELDS, map(int, numbers), strict=True))
        if item["decision"] and not item["ply"] < item["decision"] <= item["ply"]+8:
            raise ValueError("decision outside the observation window")
        windows.append(dict(key=key[:16], kind=kind, status=status, **game, **item))
        kinds[game["source"], game["split"], kind][status] += 1
    if accounted != games.keys():
        raise ValueError("probe omitted recordings")
    return windows, {"/".join(key): counts for key, counts in sorted(kinds.items())}


def run(roots, reference):
    started = time.perf_counter()
    compile_probe()
    with tempfile.TemporaryFile(mode="w+t", encoding="ascii", newline="\n", dir=OUT) as feed:
        games, tallies = gather(roots, feed, old_layouts(reference))
        if not games:
            raise ValueError("no complete JGS records found")
        feed.seek(0)
        executable = [EXE, ROOT / "config/ruleset.json"]
        execution = subprocess.run(executable, stdin=feed, capture_output=True, text=True, check=True, timeout=180)
    windows, kinds = read_results(execution.stdout, games, tallies)
    data = dict(sources=tallies, splits=kinds, windows=windows, checked_old_archive=reference is not None,
                rules_sha256=hashlib.sha256((ROOT/"config/ruleset.json").read_bytes()).hexdigest(),
                elapsed_seconds=round(time.perf_counter()-started, 3))
    result = OUT / "audit.json"
    temp = result.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
    temp.replace(result)
    print(json.dumps({k:v for k,v in data.items() if k!="windows"}, sort_keys=True))
    print(f"PASS {len(windows)} public timing observations")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("art", type=Path)
    cli.add_argument("replays", type=Path)
    cli.add_argument("--old-index", type=Path, help="path to the existing local public-archive manifest")
    args = cli.parse_args()
    run({"art": args.art, "replays": args.replays}, args.old_index)
