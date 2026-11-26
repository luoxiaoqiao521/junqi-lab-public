"""Recover immutable generated move tapes and replay them under current adjudication."""
from collections import Counter, defaultdict
from copy import deepcopy
from io import BytesIO
import json
from math import isfinite
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from zipfile import ZipFile

from full_mode_runtime import FullModeRun, isolated, sha
from serial_dark_first_loss import PREVIOUS as SECOND
from serial_dark_ordinary_inheritance import PREVIOUS as FIRST
from serial_dark_ordinary_learning import RULES, model_wire, parse
from serial_dynamic_reply import ROOT, build, require
from serial_worker import read, write_receipt


ARCHIVES={1:(FIRST,"ba1938b","parent"),2:(SECOND,"f693c26","generation1")}
KEY=("seed","seat","table","arm")


def sample_key(row):
    return (*[row[k] for k in KEY],round(row["features"][0]*140))


def frozen_source(root,revision):
    tree=subprocess.check_output(["git","rev-parse",revision],cwd=ROOT,text=True).strip()
    snapshot=subprocess.check_output(["git","archive","--format=zip",tree],cwd=ROOT)
    with ZipFile(BytesIO(snapshot)) as bundle:
        for name in bundle.namelist():
            if name.split("/")[0] not in {"core","observe","sim","strategy","tests"} or name.endswith("/"):
                continue
            dest=(root/name).resolve()
            require(dest.is_relative_to(root),"invalid tracked archive path")
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(bundle.read(name))
    return tree


def instrument_archive(root):
    path=root/"tests/deadline_dark_ordinary_learning.h"
    text=path.read_text(encoding="utf8")
    anchors={
        "  const auto start=opening(seed,rules);":"""  const auto start=opening(seed,rules);
  std::cout<<"T "<<seed<<' '<<learner<<' '<<table<<' '<<arm<<' '<<int(start.to_move);
  for (const int rank:start.rank_of) std::cout<<' '<<rank;
  std::cout<<'\\n';""",
        "    game.play(move,learner,rules);":"""    StepInfo recorded;
    game.play(move,learner,rules,&recorded);
    std::cout<<"M "<<seed<<' '<<learner<<' '<<table<<' '<<arm<<' '<<int(move.from)<<' '<<int(move.to)
             <<' '<<int(recorded.seat)<<' '<<int(recorded.outcome)<<' '<<int(recorded.eliminated_mask)
             <<' '<<int(recorded.revealed_mask)<<' '<<int(recorded.skipped_mask);
    for (const auto exit:recorded.exit_reason) std::cout<<' '<<int(exit);
    std::cout<<' '<<int(recorded.end_reason)<<' '<<int(game.state.result)<<'\\n';"""}
    for old,new in anchors.items():
        require(text.count(old)==1,"legacy instrumentation anchor is missing or ambiguous")
        text=text.replace(old,new)
    path.write_text(text,encoding="utf8")


def recover_archive(scope,number):
    archive,revision,parent_name=ARCHIVES[number]
    phase=scope.folder/f"generation-{number}"
    phase.mkdir()
    legacy=FullModeRun(phase)
    legacy.engine.mkdir()
    root=(phase/"source").resolve()
    tree=frozen_source(root,revision)
    instrument_archive(root)
    build(legacy.engine,root=root)
    shutil.copyfile(archive/"engine/rules.json",legacy.rules)
    require(sha(legacy.rules)==sha(scope.rules),"historic rules must be replayed with an explicit mapping")
    for n,p in (("original.json",archive/"training.json"),("parent.json",archive/(parent_name+".json"))):
        shutil.copyfile(p,phase/n)
    history=read(phase/"original.json")
    grid=[tuple(r[k] for k in KEY[:3]) for r in history["games"]]
    wire=["D",model_wire(read(phase/"parent.json")),*(" ".join(map(str,g)) for g in grid)]
    with isolated(legacy,ROOT/"build"):
        raw=legacy.run("dark-ordinary-learning","\n".join(wire)+"\n")
    plain=[]
    tapes={}
    for line in raw.splitlines():
        tag,*fields=line.split()
        if tag=="T":
            seed,seat,table,arm,first,*board=map(int,fields)
            require(len(board)==100 and (seed,seat,table,arm) not in tapes,"legacy opening did not emit a valid tape")
            tapes[seed,seat,table,arm]=dict(first=first,board=board,events=[],picks=[])
        elif tag=="M":
            values=list(map(int,fields))
            require(len(values)==17,"old course event lost settlement fields")
            tapes[tuple(values[:4])]["events"].append(values[4:])
        else:
            plain.append(line)
    games,rows,_=parse("\n".join(plain),"D",grid)
    require(games==history["games"] and rows==history["examples"],
            "instrumented archival executable cannot reproduce the original output")
    grouped=defaultdict(list)
    for row in rows:
        grouped[sample_key(row)[:4]].append(sample_key(row)[4])
    records=[]
    for game in games:
        k=tuple(game[n] for n in KEY)
        tape=tapes[k]
        require(len(tape["events"])==game["plies"],"old move tape length differs from its receipt")
        records.append(dict(game=game,**dict(tape,picks=grouped[k])))
    write_receipt(phase/"tapes.json",records)
    return phase,history,records,dict(commit=tree,
            original_dataset_sha256=sha(phase/"original.json"),
            parent_sha256=sha(phase/"parent.json"),archived_executable_sha256=sha(archive/"engine/deadline_opening.exe"),
            rebuilt_executable_sha256=sha(legacy.binary),tape_sha256=sha(phase/"tapes.json"))


def tape_wire(record):
    game=record["game"]
    header=[*(game[k] for k in KEY),game["result"],record["first"],len(record["events"]),len(record["picks"])]
    rows=[header,record["board"],record["picks"],*record["events"]]
    return "\n".join(" ".join(map(str,r)) for r in rows)+"\n"


def replay(scope,records):
    with isolated(scope,ROOT/"build"):
        raw=scope.run("dark-course-tape-replay","".join(map(tape_wire,records)))
    checks={}
    emitted=defaultdict(list)
    for tag,*parts in map(str.split,raw.splitlines()):
        key=tuple(map(int,parts[:4]))
        if tag=="V":
            require(key not in checks,"duplicate current tape verdict")
            checks[key]=list(map(int,parts[4:]))
        else:
            require(tag=="E" and key in checks,"unexpected course replay output")
            result=int(parts[4])
            x=list(map(float,parts[5:]))
            require(len(x)==44 and all(map(isfinite,x)),"replay exported invalid features")
            emitted[key].append(dict(zip(KEY,key),outcome=result,features=x))
    require(list(checks)==[tuple(r["game"][n] for n in KEY) for r in records],"tape replay assignments changed")
    return checks,emitted


def classify(history,records,checks,raw):
    admitted=[]
    trace=[]
    fields=Counter()
    reasons=Counter()
    for record in records:
        game=record["game"]
        key=tuple(game[n] for n in KEY)
        verdict,matched,outcome=checks[key]
        current=raw[key]
        require(0<=matched<=game["plies"] and verdict in (0,1,2,3),"ambiguous current replay status")
        if verdict!=0 or outcome==-1:
            require(not current,"invalid or incomplete course exported a terminal label")
            why=("unsettled","illegal_action","public_event_changed","terminal_changed")[verdict]
            reasons[why]+=len(record["picks"])
            trace.append(dict(key=key,matched=matched,status=why,rows=len(record["picks"])))
            continue
        require(matched==game["plies"] and outcome==game["result"] and
                [sample_key(x)[4] for x in current]==record["picks"],"re-encoding changed the target or sampled turn")
        for x in current:
            k=sample_key(x)
            old=history[k]
            changed=x["features"]!=old["features"]
            if changed:
                fields.update(i for i,(a,b) in enumerate(zip(x["features"],old["features"],strict=True)) if a!=b)
            status="reencoded" if changed else "identical"
            reasons[status]+=1
            admitted.append(dict(data=x,status=status))
        trace.append(dict(key=key,matched=matched,status="complete",rows=len(current)))
    return admitted,trace,reasons,fields


def negative_controls(scope,record):
    key=tuple(record["game"][n] for n in KEY)
    bad_action,bad_event,bad_target=(deepcopy(record) for _ in range(3))
    bad_action["events"][0][1]=bad_action["events"][0][0]
    bad_event["events"][0][2]=(bad_event["events"][0][2]+1)%4
    bad_target["game"]["result"]=(bad_target["game"]["result"]+1)%3
    out={}
    for name,tape,expected in (("illegal_action",bad_action,1),("public_event",bad_event,2),("wrong_terminal",bad_target,3)):
        verdict,emitted=replay(scope,[tape])
        require(verdict[key][0]==expected and not emitted[key],"fault injection produced a training target")
        out[name]=verdict[key][0]
    return out


def main(audit):
    began=perf_counter()
    scope=FullModeRun(ROOT/"build/dark-course-tape"/os.environ["JUNQI_SERIAL_KEY"])
    number=2 if audit else 1
    if not audit:
        scope.engine.mkdir(parents=True,exist_ok=True)
        build(scope.engine)
        shutil.copyfile(RULES,scope.rules)
        write_receipt(scope.folder/"engine.json",{"rules":sha(scope.rules),"binary":sha(scope.binary)})
    require(read(scope.folder/"engine.json")=={"rules":sha(scope.rules),"binary":sha(scope.binary)},
            "current archived-tape referee changed")
    folder,history,records,provenance=recover_archive(scope,number)
    verdicts,features=replay(scope,records)
    old={sample_key(r):r for r in history["examples"]}
    admitted,trace,reasons,fields=classify(old,records,verdicts,features)
    require(len(old)==sum(reasons.values()) and admitted,"row validity classification failed")
    healthy=next(r for r in records if verdicts[tuple(r["game"][k] for k in KEY)][0]==0 and r["game"]["result"]>=0)
    controls=negative_controls(scope,healthy)
    report=dict(episodes=len(records),rows=len(old),complete=sum(r["status"]=="complete" for r in trace),
                statuses=reasons,changed_feature_indexes=fields,controls=controls,provenance=provenance,
                seconds=perf_counter()-began)
    write_receipt(folder/"verified-pool.json",dict(report=report,admitted=admitted,episodes=trace))
    if audit:
        print("JUNQI_RESEARCH supported")
    print(json.dumps(report))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected archived tape replay phase")
    main(bool(sys.argv[1:]))
