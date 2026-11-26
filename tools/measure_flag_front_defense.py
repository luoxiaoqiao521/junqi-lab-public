"""Sample recorded defender turns without selecting on their eventual outcome."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile

from archive_action_screen import layout_group
from audit_qq_archive import INPUT, ROOT, archived_move, decode, initial_board, public_records
from audit_public_attack_windows import COVERAGE, OUT, write_json
from compare_qq_rules import VC, tape
from evidence_model import sha
from partner_crossplay import model_roster

EXE=OUT/"flag_front_defense.exe"
RESULT=OUT/"flag-front-defense.json"
NAMES=("path","contact","oracle")
COUNT=64


def window_prefix(record, window):
    _,layouts,seats,events=record
    board=initial_board(layouts)
    ply=0
    for e in events:
        if e[0]==245:
            if e[1] in (3,4) and e[2] in seats:
                owner=seats.index(e[2])
                board={p:piece for p,piece in board.items() if piece[0]!=owner}
            continue
        ply+=1
        actor=board[(e[2],16-e[3])][0]
        if ply>window["start"]:
            if actor==window["actor"]:
                return None
            if actor==window["defender"]:
                rows,direction=tape(layouts,seats,events)
                if direction:
                    return None
                ordinary=0
                for i,row in enumerate(rows):
                    ordinary+=row[0]>=0
                    if ordinary==ply:
                        return list((x,y,r) for (x,y),(_,r) in initial_board(layouts).items()),rows[:i+1]
        archived_move(board,e)
    return None


def select():
    paths=json.loads(INPUT.read_text(encoding="utf-8-sig"))
    index=json.loads(COVERAGE.read_text(encoding="utf-8"))["windows"]
    cases={row["source"]:row for row in index}
    groups=defaultdict(dict)
    seen=set()
    for raw in public_records(Path(paths["corpus"])):
        try:
            record=decode(raw)
        except ValueError:
            continue
        key=record[0][:16]
        if key not in cases or key in seen:
            continue
        seen.add(key)
        window=cases[key]
        group=layout_group(record[1])
        groups[window["enemies_left"]].setdefault(group,(record,window))
    used=set()
    for count in (1,2):
        keys=sorted(groups[count].keys()-used)
        for key in random.Random(953301+count).sample(keys,COUNT):
            used.add(key)
            record,window=groups[count][key]
            name=f"{count}-{key.hex()[:16]}"
            yield name,window_prefix(record,window)


def compile_probe():
    root=ROOT.as_posix()
    units=["core/board.cpp","core/rules.cpp","core/state.cpp","core/layout.cpp","tests/flag_front_defense.cpp"]
    source=" ".join(f'"{root}/{unit}"' for unit in units)
    cmd=f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /W4 /WX /O2 /MT /utf-8 /I"{root}/core" {source} /Fe:flag_front_defense.exe'
    subprocess.run(f'cmd.exe /d /s /c "{cmd}"',cwd=OUT,check=True,timeout=75)


def parse(text):
    data,pairs,test={},defaultdict(list),False
    for line in text.splitlines():
        code,group,*raw=line.split()
        values=list(map(int,raw))
        if code=="F":
            test=group=="1"
        elif code=="X":
            data[group]=None
        elif code=="D":
            data[group]=values
        elif code=="P":
            if values[0]!=len(pairs[group]):
                raise ValueError("incomplete frozen seed order")
            pairs[group].append(values[1:])
        else:
            raise ValueError("unexpected output")
    return data,dict(pairs),test


def summary(data,pairs,absent):
    results={}
    for count in (1,2):
        rows=[v for k,v in data.items() if k.startswith(f"{count}-") and v is not None]
        outcomes=[v for k,vs in pairs.items() if k.startswith(f"{count}-") for v in vs]
        changes=lambda index:sum(r[index] for r in rows)
        result=dict(requested=COUNT,missing_turn=sum(k.startswith(f"{count}-") for k in absent),
                    aligned=len(rows),frozen_warning=changes(1),
                    contact_differs_from_path=changes(14),oracle_differs_from_contact=changes(15),
                    continuation_sets=len(outcomes))
        result["profiles"]={}
        for i,name in enumerate(NAMES):
            result["profiles"][name]=dict(threat=sum(r[2+4*i]>0 for r in rows),
                                          forced=sum(r[3+4*i]>0 for r in rows),
                                          redirect=changes(4+4*i),remaining_warning=changes(5+4*i),
                                          natural_pairs=sum(min(row[0],row[i+1])>=0 for row in outcomes),
                                          outcome_difference=dict(Counter(row[i+1]-row[0] for row in outcomes
                                                                          if min(row[0],row[i+1])>=0)))
        results[str(count)]=result
    return results


def run():
    cases=list(select())
    roster=model_roster()["candidate"]
    available=[(key,prefix) for key,prefix in cases if prefix]
    absent=[key for key,prefix in cases if not prefix]
    with tempfile.TemporaryFile(mode="w+t",encoding="ascii",newline="\n",dir=OUT) as sink:
        print(len(available),file=sink)
        print(*roster["weights"],file=sink)
        for key,(board,rows) in available:
            print(key,len(rows),file=sink)
            for item in board+rows:
                print(*item,file=sink)
        sink.seek(0)
        result=subprocess.run([EXE],stdin=sink,capture_output=True,text=True,timeout=130)
    if result.returncode:
        raise RuntimeError(result.stderr or f"probe exit {result.returncode}")
    data,pairs,fixture=parse(result.stdout)
    write_json(RESULT,dict(model=roster["artifact_sha256"],coverage=sha(COVERAGE),data=data,pairs=pairs,
                           absent=absent,fixture=fixture,summary=summary(data,pairs,absent)))
    audit()


def audit():
    r=json.loads(RESULT.read_text(encoding="utf-8"))
    data,pairs,absent=r["data"],r["pairs"],r["absent"]
    all_cases=list(data)+absent
    if (len(set(all_cases))!=len(all_cases) or Counter(k.split("-")[0] for k in all_cases)!={"1":COUNT,"2":COUNT}
            or not r["fixture"] or r["model"]!=model_roster()["candidate"]["artifact_sha256"]
            or r["coverage"]!=sha(COVERAGE)):
        raise ValueError("archive sample or guard fixture did not reproduce")
    if set(pairs)!={k for k,v in data.items() if v and (v[4] or v[8] or v[12])}:
        raise ValueError("moved guard without paired outcomes")
    for key,rows in pairs.items():
        if len(rows)!=12 or any(len(r)!=4 or min(r)<-1 or max(r)>2 for r in rows):
            raise ValueError("paired guard scores failed bounds")
        for slot,index in enumerate((4,8,12),1):
            if not data[key][index] and any(row[slot]!=row[0] for row in rows):
                raise ValueError("unchanged guard changed the result")
    if r["summary"]!=json.loads(json.dumps(summary(data,pairs,absent))):
        raise ValueError("archive guard summary did not reproduce")
    print(json.dumps(r["summary"]))


if __name__=="__main__":
    if sys.argv[1:]==["audit"]:
        audit()
    else:
        compile_probe()
        run()
