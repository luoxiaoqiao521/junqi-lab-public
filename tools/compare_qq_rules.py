"""Evaluate no-move timing on recorded board states with the C++ core."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from audit_qq_archive import ROOT, INPUT, decode, initial_board, archived_move, local_records, public_records

OUT = ROOT / "build/qq-evidence"
EXE = OUT / "qq_no_move_probe.exe"
RESULT = OUT / "native-timing.json"
VC = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat")


def compile_probe():
    OUT.mkdir(parents=True, exist_ok=True)
    project = ROOT.as_posix()
    sources = " ".join(f'"{project}/{f}"' for f in (
        "core/board.cpp", "core/rules.cpp", "core/state.cpp", "core/layout.cpp",
        "tests/qq_no_move_probe.cpp"))
    command = f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /O2 /MT /utf-8 /I"{project}/core" {sources} /Fe:qq_no_move_probe.exe'
    subprocess.run(f'cmd.exe /d /s /c "{command}"', cwd=OUT,
                   check=True, timeout=75)


def tape(layouts, seats, events):
    board = initial_board(layouts)
    entries = []
    previous = None
    turns = Counter()
    for event in events:
        if event[0] == 95:
            actor, _ = archived_move(board, event)
            if previous is not None:
                turns[(actor-previous) % 4] += 1
            previous = actor
            entries.append((event[2],16-event[3],event[4],16-event[5],event[1]&3))
            continue
        if event[0] != 245:
            raise ValueError("unknown event")
        kind, seat_code, cause = event[1:4]
        seat = seats.index(seat_code) if kind in (3,4) and seat_code in seats else -1
        if seat >= 0:
            board = {pos: piece for pos, piece in board.items() if piece[0] != seat}
            seat = (seat+1)%4
        entries.append((-1, seat, kind, 0, cause))
    return entries, int(turns[1] > turns[3])


def collect(sources):
    result = []
    for source, inputs in enumerate(sources):
        seen = set()
        for raw in inputs:
            try:
                key, layouts, seats, events = decode(raw)
            except ValueError as error:
                if str(error) == "four_complete_layouts":
                    continue
                raise
            if key not in seen:
                seen.add(key)
                result.append((source, layouts, seats, events))
    return result


def run():
    manifest = {k: Path(v) for k,v in json.loads(INPUT.read_text(encoding="utf-8-sig")).items()}
    compile_probe()
    games = collect([public_records(manifest["corpus"]), local_records(manifest["local"]),
                     [manifest["supplemental"].read_bytes()]])
    with tempfile.TemporaryFile(mode="w+t", encoding="ascii", newline="\n", dir=OUT) as stream:
        stream.write(f"{len(games)}\n")
        for source, layouts, seats, events in games:
            entries, ccw = tape(layouts, seats, events)
            stream.write(f"{source} {len(entries)} {ccw}\n")
            for (x,y), (_,rank) in initial_board(layouts).items():
                stream.write(f"{x} {y} {rank}\n")
            for entry in entries:
                stream.write(" ".join(map(str,entry))+"\n")
        stream.seek(0)
        output = subprocess.run([EXE], stdin=stream, capture_output=True,
                                text=True, check=True, timeout=110).stdout
    names = ("games","plies","contacts","bad_legal","bad_contact","no_moves","no_move_mismatches",
             "natural_windows","external_windows","natural_no_moves")
    models = ("on_turn", "permanent_then_on_turn", "all_now_then_on_turn",
              "involved_then_on_turn", "all_after_contact_then_on_turn")
    stats = {}
    for label,line in zip(("archive","local","supplemental"),output.splitlines(),strict=True):
        row = tuple(map(int,line.split()))
        stats[label] = dict(zip(names,row[:len(names)],strict=True))
        stats[label]["timing"] = {
            name: dict(zip(("different_windows","missed","extra"),row[i:i+3],strict=True))
            for name,i in zip(models,range(len(names),len(row),3),strict=True)}
        print(label,json.dumps(stats[label],sort_keys=True))
    temporary = RESULT.with_suffix(".tmp")
    temporary.write_text(json.dumps(stats,indent=2)+"\n",encoding="utf8")
    temporary.replace(RESULT)


def audit():
    stats = json.loads(RESULT.read_text(encoding="utf8"))
    tally = [(s["games"],s["plies"],s["contacts"],s["no_moves"]) for s in stats.values()]
    if tally != [(3235,654829,158940,995),(41,5265,1700,3),(1,245,59,2)]:
        raise ValueError("record coverage changed")
    if any(s["bad_legal"] or s["bad_contact"] or s["no_move_mismatches"] for s in stats.values()):
        raise ValueError("original actions, collisions or no-move legality did not align")
    print("PASS native board legality and timing alternatives",json.dumps(stats,sort_keys=True))


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
