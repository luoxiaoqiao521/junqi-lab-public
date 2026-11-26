"""Compare the production referee on the exact decision frames of local JGS files."""
from collections import Counter
import json
import os
import subprocess
import sys
import tempfile

from audit_human_timing import gather
from compare_qq_rules import VC
from compare_replay_first_followthrough import require, write_result
from pilot_relay_candidate_budget import REPO as ROOT, ROOTS, file_sha

NAMES = ("archive", "selfplay_same_direction", "on_turn_only",
         "quiet_70", "no_reveal", "other_bomb_flag")
CONFIG = ROOT / "config/ruleset.json"
TRAINING = ROOT / "config/ruleset-selfplay.json"


def build(directory):
    sources = ("core/board.cpp", "core/layout.cpp", "core/rules.cpp", "core/state.cpp",
               "tests/replay_rule_variants.cpp")
    files = " ".join(f'"{ROOT.as_posix()}/{source}"' for source in sources)
    command = (f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /O2 /MT /W4 /WX /utf-8 '
               f'/I"{ROOT.as_posix()}/core" {files} /Fe:replay_rule_variants.exe')
    subprocess.run(f'cmd.exe /d /s /c "{command}"', cwd=directory, check=True, timeout=100)


def load_rows(output, sources):
    games = {}
    current = None
    for line in output.splitlines():
        tag, *parts = line.split()
        if tag in ("S", "R"):
            current, = parts
            require(current in sources and current not in games, "unregistered or repeated replay")
            games[current] = dict(valid=tag=="S", differences=[], flags=[], controls=[])
            continue
        require(current is not None and games[current]["valid"], "events outside a valid record")
        record = games[current]
        if tag == "G":
            record["end"] = dict(zip(("plies","result","quiet","ccw"), map(int,parts), strict=True))
        elif tag == "M":
            index, ply, reason, ending, simulated, archived = parts
            record["differences"].append([NAMES[int(index)], int(ply), reason, int(ending),
                                           int(simulated), int(archived)])
        else:
            require(tag in ("C","F"), "unknown native output")
            record["controls" if tag=="C" else "flags"].append(list(map(int,parts)))
    require(games.keys()==sources.keys(), "replay disappeared")
    require(all("end" in r for r in games.values() if r["valid"]), "missing terminal")
    return games


def summarize(games):
    good = [record for record in games.values() if record["valid"]]
    require(good, "no verified local replays")
    margins = {name: Counter() for name in NAMES}
    per_game = Counter()
    stable = Counter()
    for record in good:
        changes = set()
        for name, _, reason, *_ in record["differences"]:
            margins[name][reason] += 1
            changes.add(name)
        per_game.update(changes)
        for ply, cleared, locked, revealed, bomb in record["flags"]:
            require(cleared, f"archive rules did not clear after flag at ply {ply}")
            stable.update(flags=1, locked=locked, dark=not revealed, bomb=bomb)
    ends = [record["end"] for record in good]
    return dict(games=len(good), plies=sum(r["plies"] for r in ends), rejected=len(games)-len(good),
                mismatches=margins, games_with_mismatch=per_game, flags=stable,
                draw_tails=Counter(r["quiet"] for r in ends if r["result"]==3),
                excluded_windows=sum(len(r["controls"]) for r in good))


def main(audit):
    directory = ROOT/"build/replay-rule-variants"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True, exist_ok=True)
        build(directory)
    with tempfile.TemporaryFile(mode="w+t", encoding="ascii", dir=directory) as feed:
        sources, inventory = gather({"art":ROOTS[0],"review":ROOTS[1]}, feed, ())
        feed.seek(0)
        command = [directory/"replay_rule_variants.exe", CONFIG, TRAINING]
        result = subprocess.run(command,stdin=feed,capture_output=True,text=True,check=True,timeout=130)
    games = load_rows(result.stdout,sources)
    summary = summarize(games)
    receipt = dict(summary=summary, inventory=inventory, games=games, sources=sources,
                   rules=[file_sha(CONFIG),file_sha(TRAINING)])
    write_result(directory,receipt,audit)
    print(json.dumps(summary,sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]), "expected no arguments or audit")
    main(sys.argv[1:]==["audit"])
