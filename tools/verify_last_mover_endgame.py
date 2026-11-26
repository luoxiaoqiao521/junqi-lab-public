"""Verify the active last-mover contract and the named historic exceptions."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from audit_qq_archive import HQ, INPUT, archived_move, decode, initial_board, public_records
from audit_replay_retirement import mover, points
from verify_relay_continuity import ROOT, compile_probe, sha

WITNESSES = {
    "8f19733c9a897bc9": (264, (10, 6), (10, 10)),
    "9c0d7722649aa191": (295, (8, 15), (10, 15)),
    "c749306d00b2cc88": 287,
    "9c966a5ebecfa367": 177,
}
NATIVE_LINES = (
    "PASS last-mover flag race:",
    "PASS last-mover mutual collisions:",
    "PASS last-mover flag priority:",
    "PASS last-mover ranks and mines:",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def active_movers(board, seat):
    return [point for point, piece in board.items() if piece[0] == seat and mover(point, piece)]


def verify_mutual(board, after, seats, events, index, active, ply, expected):
    event = events[index]
    origin, target = points(event)
    attacker, defender = board[origin][0], board[target][0]
    require((ply, origin, target) == expected and (attacker, defender) == (3, 2),
            "mutual collision coordinates or actors changed")
    require(event[1] & 3 == 3 and active == {attacker, defender} and
            active_movers(board, attacker) == [origin] and active_movers(board, defender) == [target] and
            not active_movers(after, attacker) and not active_movers(after, defender),
            "mutual collision is missing the two last movable pieces")
    require([tuple(item[:4]) for item in events[index+1:]] ==
            [(245, 3, seats[defender], 2), (245, 5, 0, 4)],
            "defender-first terminal controls changed")
    return dict(ply=ply, sequence=index+1, move=[origin,target], attack_seat=attacker,
                defender_retirement="no_legal_move", terminal_cause=4)


def verify_blocked(board, seats, event, ply, index, active):
    seat = seats.index(event[2])
    if index+1 == 177:
        origin = (8, 0)
        require(ply == 172 and event[3] == 2 and active == {0, 1} and seat == 1 and
                (0, 2) in board.values() and (1, 2) in board.values() and
                active_movers(board,0) == [(16, 6)] and (15, 6) not in board and
                active_movers(board,seat) == [origin], "two-seat mine trap witness changed")
        require(board[(8, 1)] == (seat,3) and board[(9, 0)] == (seat,2) and
                (7, 0) in HQ and board[(7, 0)][0] == seat,
                "mine, flag and locked piece must occupy the three exits")
        return dict(ply=ply, sequence=index+1, seat=seat, last_mover=origin,
                    own_mine=(8, 1), own_flag=(9, 0), own_hq=(7, 0), opponent_has_exit=True, cause=2)
    origin = (10, 0)
    own_mine, own_hq = (10, 1), (9, 0)
    require((ply,index+1) == (285,287) and event[3] == 2 and active_movers(board,seat) == [origin] and
            board[origin] == (seat,12), "blocked last mover was absent or changed")
    require(board[own_mine] == (seat,3) and board[own_hq][0] == seat and own_hq in HQ,
            "the mine and headquarters on the two road exits changed")
    return dict(ply=ply, sequence=index+1, seat=seat, last_mover=origin,
                own_mine=own_mine, own_hq=own_hq, cause=2)


def trace(layouts, seats, events, expected):
    board = initial_board(layouts)
    active = set(range(4))
    ply = 0
    for index, event in enumerate(events):
        if event[0] == 95:
            ply += 1
            if isinstance(expected,tuple) and ply == expected[0]:
                after = board.copy()
                archived_move(after,event)
                return verify_mutual(board,after,seats,events,index,active,ply,expected)
            archived_move(board,event)
        elif event[0] == 245 and event[1] in (3,4):
            if index + 1 == expected:
                return verify_blocked(board,seats,event,ply,index,active)
            seat = seats.index(event[2])
            active.remove(seat)
            board = {point: piece for point, piece in board.items() if piece[0] != seat}
    raise ValueError("expected anonymous archive checkpoint was not found")


def audit_archive():
    manifest = json.loads(INPUT.read_text(encoding="utf-8-sig"))
    corpus = Path(manifest["corpus"])
    result = {}
    for raw in public_records(corpus):
        try:
            key, layouts, seats, events = decode(raw)
        except ValueError:
            continue
        key = key[:16]
        if key in WITNESSES and key not in result:
            result[key] = trace(layouts,seats,events,WITNESSES[key])
        if result.keys() == WITNESSES.keys():
            break
    require(result.keys() == WITNESSES.keys(), "named archived endgame sources missing")
    with corpus.open("rb") as archive:
        fingerprint = hashlib.file_digest(archive,"sha256").hexdigest()
    return dict(witnesses=result, archive_sha256=fingerprint, manifest_sha256=sha(INPUT),
                target_room_verified=False)


def verify_native(out):
    compile_probe(out)
    executable = out / "relay.exe"
    config = ROOT / "config/ruleset-selfplay.json"
    run = subprocess.run([executable,"--last-mover-endgame-fixtures",config], capture_output=True,
                         text=True, timeout=30, check=False)
    require(run.returncode == 0, f"native last-mover fixtures failed: {run.stderr.strip()}")
    observed = tuple(run.stdout.splitlines())
    require(len(observed) == len(NATIVE_LINES) and all(
            line.startswith(prefix) for line,prefix in zip(observed,NATIVE_LINES)),
            "native last-mover fixtures missing acceptance lines")
    print(*observed,sep="\n")
    return dict(verified=observed, executable_sha256=sha(executable), rules_sha256=sha(config))


def main(audit):
    key = os.environ.get("JUNQI_SERIAL_KEY","manual")
    require(bool(key) and all(char.isalnum() or char == "-" for char in key),"invalid local serial key")
    out = ROOT / "build" / "last-mover-endgame" / key
    out.mkdir(parents=True,exist_ok=True)
    if audit:
        result = audit_archive()
        print(json.dumps(result,sort_keys=True))
        print("PASS historic replay: two mutual last movers, defender first; two last movers boxed by own mines")
    else:
        result = verify_native(out)
    path = out / ("archive.json" if audit else "native.json")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    temporary.replace(path)
    print("JUNQI_VERDICT accepted")


if __name__ == "__main__":
    if sys.argv[1:] not in ([],["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(sys.argv[1:] == ["audit"])
