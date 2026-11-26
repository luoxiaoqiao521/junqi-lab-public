"""Audit one-round public flag warnings on local QQ replay prefixes."""
from collections import defaultdict
from pathlib import Path
import json
import random
import subprocess
import sys
import tarfile
import tempfile

from archive_action_screen import layout_group
from audit_qq_archive import INPUT, LIMIT, ROOT, decode, initial_board, local_records
from compare_qq_rules import VC, tape
from partner_crossplay import model_roster

OUT = ROOT / "build/qq-evidence"
EXE = OUT / "replay_deadline_probe.exe"
RESULT = OUT / "replay-deadline.json"
CASES = {"000023": 84, "000346": 185, "002596": 139, "000008": 280}
MOVE_LABELS = ("human", "frozen", "deadline", "heuristic", "guard")


def sources(manifest):
    anchors, pool = {}, {}
    local = set()
    for raw in local_records(manifest["local"]):
        try:
            local.add(layout_group(decode(raw)[1]))
        except ValueError:
            continue
    with tarfile.open(manifest["corpus"], "r:gz") as archive:
        for member in archive:
            name = Path(member.name)
            if (not member.isfile() or not 412 <= member.size <= LIMIT or
                    name.suffix != ".jgs" or not name.stem.isdecimal()):
                continue
            try:
                with archive.extractfile(member) as stream:
                    record = decode(stream.read())
            except ValueError:
                continue
            if name.stem in CASES:
                anchors[name.stem] = record
            elif (group := layout_group(record[1])) not in local and group not in pool:
                pool[group] = record
    if set(anchors) != set(CASES):
        raise ValueError("archive is missing a named case")
    development = {layout_group(record[1]) for record in anchors.values()}
    sampled = random.Random(208151).sample(sorted(pool.keys() - development), 64)
    return [(name, anchors[name], ply) for name, ply in CASES.items()] + [
        ("ordinary-" + pool[group][0][:12], pool[group], 85 + 20*(i % 4))
        for i, group in enumerate(sampled)]


def compile_probe():
    OUT.mkdir(parents=True, exist_ok=True)
    project = ROOT.as_posix()
    files = " ".join(f'"{project}/{f}"' for f in
                     ("core/board.cpp", "core/rules.cpp", "core/state.cpp", "core/layout.cpp",
                      "tests/replay_deadline_probe.cpp"))
    command = (f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /W4 /WX /O2 /MT '
               f'/utf-8 /I"{project}/core" {files} /Fe:replay_deadline_probe.exe')
    subprocess.run(f'cmd.exe /d /s /c "{command}"', cwd=OUT, check=True, timeout=75)


def write_case(stream, name, record, ply):
    _, layouts, seats, events = record
    rows, ccw = tape(layouts, seats, events)
    plies = 0
    for index, row in enumerate(rows):
        plies += row[0] >= 0
        if plies == ply:
            break
    else:
        print(f"{name} 0 {ccw} 0", file=stream)
        for (x, y), (_, rank) in initial_board(layouts).items():
            print(x, y, rank, file=stream)
        return
    repeats = 64 if name in CASES else 16
    print(name, index+1, ccw, repeats, file=stream)
    for (x, y), (_, rank) in initial_board(layouts).items():
        print(x, y, rank, file=stream)
    for row in rows[:index+1]:
        print(*row, file=stream)


def parse(text):
    decisions, excluded, pairs = {}, [], defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for line in text.splitlines():
        tag, name, *tail = line.split()
        if tag == "X":
            excluded.append(name)
        elif tag == "P":
            kind, *numbers = tail
            table, sample, original, candidate = map(int, numbers)
            if sample != len(pairs[name][kind][table]):
                raise ValueError("incomplete paired stream")
            pairs[name][kind][table].append((original, candidate))
        elif tag == "D":
            ply, seat, actions, guards, wins, mass, changed, *options = tail
            row = dict(ply=int(ply), seat=int(seat), actions=int(actions), safe_count=int(guards),
                       win_count=int(wins), danger_mass=float(mass), changed=bool(int(changed)))
            for i, label in enumerate(MOVE_LABELS):
                move, turns, unchanged, win, uncertain, truth, higher, prob, enemy, owner = options[10*i:10*i+10]
                row[label] = dict(move=move, turns=int(turns), unchanged_turn=int(unchanged),
                                  win=bool(int(win)), uncertain=bool(int(uncertain)),
                                  diagnostic_truth=int(truth), higher=int(higher), probability=float(prob),
                                  threatening_seat=int(enemy), flag_owner=int(owner))
            decisions[name] = row
        else:
            raise ValueError("unknown probe row")
    return decisions, excluded, dict(pairs)


def run():
    inputs = {k: Path(v) for k, v in json.loads(INPUT.read_text(encoding="utf-8-sig")).items()}
    cases = sources(inputs)
    frozen = model_roster()["candidate"]
    compile_probe()
    with tempfile.TemporaryFile(mode="w+t", encoding="ascii", newline="\n", dir=OUT) as stream:
        print(len(cases), file=stream)
        print(*frozen["weights"], file=stream)
        for name, record, ply in cases:
            write_case(stream, name, record, ply)
        stream.seek(0)
        output = subprocess.run([EXE], stdin=stream, capture_output=True, check=True, text=True, timeout=120).stdout
    decisions, excluded, pairs = parse(output)
    result = dict(model=frozen["artifact_sha256"], cases=decisions, excluded=excluded, pairs=pairs)
    temporary = RESULT.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    temporary.replace(RESULT)
    audit()


def audit():
    report = json.loads(RESULT.read_text(encoding="utf-8"))
    decisions, excluded, pairs = report["cases"], report["excluded"], report["pairs"]
    if report["model"] != model_roster()["candidate"]["artifact_sha256"]:
        raise ValueError("paired results do not use the frozen reference")
    if set(excluded) & decisions.keys() or len(decisions) + len(excluded) != 68:
        raise ValueError("replay source groups missing")
    expected = {"000023": (4, 4, False), "000346": (3, 3, False),
                "002596": (1, 2, False), "000008": (4, 4, True)}
    for key, (deadline, unchanged, won) in expected.items():
        action = decisions[key]["human"]
        if (action["turns"], action["unchanged_turn"], action["win"]) != (deadline, unchanged, won):
            raise ValueError("public timing did not match named replay example " + key)
    exact_win = decisions["000008"]
    if not exact_win["changed"] or exact_win["deadline"]["diagnostic_truth"] != 2:
        raise ValueError("confirmed team win did not take priority")
    for key in ("000023", "000346", "002596"):
        if decisions[key]["safe_count"] != 1 or decisions[key]["guard"]["turns"] != 4:
            raise ValueError("guaranteed local interception lost")
    for key, row in decisions.items():
        kinds = pairs.get(key, {})
        expected_kinds = {"deadline"} if row["changed"] else set()
        if row["guard"]["move"] not in (row["frozen"]["move"], row["deadline"]["move"]):
            expected_kinds.add("guard")
        if key in CASES and len({row[label]["move"] for label in ("guard", "human", "frozen")}) == 3:
            expected_kinds.add("human_guard")
        if set(kinds) != expected_kinds:
            raise ValueError("paired run missing")
        for tables in kinds.values():
            if set(tables) != {"0", "1", "2"} or any(
                    len(r) != (64 if key in CASES else 16) for r in tables.values()):
                raise ValueError("missing opponent or partner table")
    print("PASS replay timing, terminal precedence and complete paired tables")
    print("ordinary",sum(key not in CASES for key in decisions), "excluded",len(excluded),
          "changed",sum(row["changed"] for key,row in decisions.items() if key not in CASES))


if __name__ == "__main__":
    audit() if sys.argv[1:] == ["audit"] else run()
