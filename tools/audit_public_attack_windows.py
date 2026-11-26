"""Extract first observed attacks in front of publicly exposed flags."""
from collections import Counter
from pathlib import Path
import json
import sys

from audit_qq_archive import INPUT, ROOT, archived_move, decode, grid, initial_board, public_records

OUT = ROOT / "build/attack-opportunity"
COVERAGE = OUT / "coverage.json"
FRONT_TO_HQ = {grid(seat,4,col):grid(seat,5,col) for seat in range(4) for col in (1,3)}
FORMAL_ENDS = (2, 4)


def ordinary(board, revealed, event):
    origin = (event[2],16-event[3])
    target = (event[4],16-event[5])
    actor, rank = board[origin]
    defender = board.get(target)
    outcome = event[1] & 3
    archived_move(board,event)
    if rank==5 and outcome in (2,3):
        revealed.add(actor)
    if defender and defender[1]==5 and outcome in (1,3):
        revealed.add(defender[0])
    return origin,target,actor,rank,defender,outcome


def attack_outcome(record):
    key,layouts,seats,events = record
    board=initial_board(layouts)
    seat_map={code:i for i,code in enumerate(seats)}
    revealed,retired=set(),set()
    window=None
    ply=0
    for event in events:
        if event[0]==245:
            notice,seat,cause=event[1:4]
            owner=seat_map.get(seat)
            if window and (notice in (2,5) or (notice in (3,4) and owner in
                           (window["actor"],window["defender"]) and owner not in retired)):
                return dict(window,end=ply,outcome="interrupted")
            if notice in (3,4) and owner is not None:
                retired.add(owner)
                board={pos:piece for pos,piece in board.items() if piece[0]!=owner}
            continue
        if event[0]!=95:
            raise ValueError("unknown replay event")
        ply+=1
        known_before=revealed.copy() if window is None else None
        origin,target,actor,rank,defender,outcome=ordinary(board,revealed,event)
        if window:
            flag=tuple(window["flag"])
            position=tuple(window["position"])
            took_flag=target==flag and defender and defender[1]==2 and outcome in (1,3)
            if actor==window["actor"]:
                took_with_attacker=bool(took_flag and origin==position)
                return dict(window,end=ply,outcome="own_capture" if took_with_attacker else "survived_until_turn")
            if took_flag and actor%2==window["actor"]%2:
                return dict(window,end=ply,outcome="ally_capture")
            if target==position and defender and defender[0]==window["actor"] and outcome in (1,3):
                return dict(window,end=ply,outcome="attacker_removed")
            continue
        flag=FRONT_TO_HQ.get(target)
        if not flag or not 5<=rank<=13 or outcome not in (0,1) or flag not in board:
            continue
        owner,identity=board[flag]
        if identity==2 and owner in known_before and owner%2!=actor%2:
            window=dict(source=key[:16],start=ply,position=target,flag=flag,actor=actor,defender=owner,
                        enemies_left=sum(x%2!=actor%2 and x not in retired for x in range(4)))
    return dict(window,end=ply,outcome="interrupted") if window else None


def write_json(path,data):
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data,ensure_ascii=False)+"\n",encoding="utf-8")
    temporary.replace(path)


def scan():
    paths=json.loads(INPUT.read_text(encoding="utf-8-sig"))
    seen,formal,errors=set(),set(),Counter()
    windows=[]
    for raw in public_records(Path(paths["corpus"])):
        try:
            record=decode(raw)
        except ValueError:
            continue
        key,_,_,events=record
        if key in seen:
            continue
        seen.add(key)
        ending=[row[3] for row in events if row[0]==245 and row[1]==5]
        if len(ending)!=1 or ending[0] not in FORMAL_ENDS:
            continue
        formal.add(key)
        try:
            result=attack_outcome(record)
        except (KeyError,ValueError) as error:
            errors[type(error).__name__]+=1
            continue
        if result:
            windows.append(result)
    outcomes=Counter(row["outcome"] for row in windows)
    OUT.mkdir(parents=True,exist_ok=True)
    report=dict(unique_records=len(seen),formal_records=len(formal),errors=dict(errors),
                missing_first_window=len(formal)-sum(errors.values())-len(windows),
                outcomes=dict(outcomes),windows=windows)
    write_json(COVERAGE,report)
    audit()


def audit():
    data=json.loads(COVERAGE.read_text(encoding="utf-8"))
    windows=data["windows"]
    categories={"own_capture","survived_until_turn","ally_capture","attacker_removed","interrupted"}
    if (len(windows)!=len({r["source"] for r in windows}) or not
            set(data["outcomes"])<=categories or not all(r["end"]>=r["start"] for r in windows)):
        raise ValueError("first-window contract failed")
    if (data["outcomes"]!=dict(Counter(row["outcome"] for row in windows)) or
            data["formal_records"]!=len(windows)+data["missing_first_window"]+sum(data["errors"].values())):
        raise ValueError("course denominator did not reconcile")
    print(json.dumps({k:v for k,v in data.items() if k!="windows"},ensure_ascii=False))


if __name__=="__main__":
    audit() if sys.argv[1:]==["audit"] else scan()
