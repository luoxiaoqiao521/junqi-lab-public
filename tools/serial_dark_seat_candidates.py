"""Compare static and actual-seat-order nominations at equal candidate width."""
from collections import Counter
from itertools import product
import json
import math
import os
import shutil
import sys
from time import perf_counter

from full_mode_runtime import FullModeRun, isolated, sha
from serial_dark_ordinary_learning import RULES, assert_layout_separation
from serial_dark_replanning import INPUTS, SOURCE, model_payload
from serial_dark_response_distribution import effects, mean_value, parse
from serial_dynamic_reply import ROOT, build, require
from serial_formation_matrix import contrast
from serial_worker import read, write_receipt


def prepare(scope):
    folder=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    for n in ("original.json","mixed.json","peer.json"):
        shutil.copyfile(SOURCE/n,folder/n)
    grid={phase:[(start+i,i%4,4*(i//4%2),(20,32,52,68)[(i//2)%4])
                 for i in range(count)] for phase,start,count in (("development",24146000,16),("confirmation",24147000,36))}
    assert_layout_separation(grid)
    write_receipt(folder/"grid.json",grid)
    write_receipt(folder/"fingerprints.json",{n:sha(folder/n) for n in INPUTS})


def decode(raw,requests):
    plain=[]
    bank={}
    side=None
    for line in raw.splitlines():
        kind,*fields=line.split()
        if kind=="F":
            key=tuple(map(int,fields[:3]))
            side=dict(membership={},features={},nomination_steps=0)
            bank[key]=side
        if kind=="S":
            a,move,membership=map(int,fields)
            side["membership"][a]=membership
            plain.append(f"S {a} {move}")
        elif kind=="N":
            side["nomination_steps"]=int(fields[0])
        elif kind=="Q":
            a=int(fields[0])
            row=list(map(float,fields[1:]))
            require(a not in side["features"] and len(row)==44 and all(map(math.isfinite,row)),"bad cue features")
            side["features"][a]=row
        else:
            plain.append(line)
    roots=parse("\n".join(plain),requests)
    for key,r in roots.items():
        r.update(bank[key])
        if r["worlds"]!=4:
            require(not r["moves"] and not r["L"] and not r["R"],"unknown world changed nominations")
            continue
        members=r["membership"]
        moves=r["moves"]
        require(members.keys()==moves.keys()==r["features"].keys() and len(set(moves.values()))==len(moves) and
                len(moves)<=r["legal"] and all(x in (1,2,3) for x in members.values()),"unregistered candidate")
        require(r["L"].keys()==set(product(moves,(0,1),(0,2),(0,1))) and
                r["R"].keys()==set(product(moves,(2,3),(0,5),(60,61,62))),"incomplete tactical action matrix")
        require(r["G"].keys()==set(product(moves,(-1,),(key[2],),range(67,71))),"missing tactical result")
        require(sum(bool(x&1) for x in members.values())==sum(bool(x&2) for x in members.values())==r["width"],
                "candidate list width changed")
    return roots


def rank_items(rows,record):
    return {a:mean_value([v for (move,*_),v in record.items() if move==a]) for a in rows}


def summary(roots):
    values={p:[] for p in ("original","reference")}
    counts=Counter()
    toll=Counter()
    for r in roots.values():
        if r["worlds"]!=4:
            values["original"].append((0,0))
            continue
        sets={a:[k for k,m in r["membership"].items() if m&a] for a in (1,2)}
        base=next(k for k,a in r["moves"].items() if a==r["base"])
        select=rank_items(r["moves"],r["L"])
        ref=rank_items(r["moves"],r["R"])
        choice=[]
        keep={}
        for arm,indices in sets.items():
            a=max(indices,key=lambda x:select[x][0])
            a=a if select[a][0]>select[base][1] else base
            choice.append(a)
            keep[arm]=max(ref[k][0] for k in indices)
            toll[str(arm)]+=sum(v[1] for (k,*_),v in r["L"].items() if k in indices)
            counts[f"changed_{arm}"]+=a!=base
        old,new=choice
        r["selected"]=choice
        counts["new_only_candidates"]+=sum(m==2 for m in r["membership"].values())
        counts["higher_reference_ceiling"]+=keep[2]>max(ref[k][1] for k in sets[1])
        counts["lower_reference_ceiling"]+=keep[1]>max(ref[k][1] for k in sets[2])
        counts["actually_selected_new_nominee"]+=r["membership"][new]==2
        counts["new_selected_worse_in_reference"]+=ref[new][1]<ref[old][0]
        counts["new_selected_better_in_reference"]+=ref[new][0]>ref[old][1]
        for group,k in (("original","G"),("reference","R")):
            values[group].append(effects(r[k],new,old) if new!=old else (0,0))
        toll["nomination"]+=r["nomination_steps"]
    return dict(assigned=len(roots),covered=sum(r["worlds"]==4 for r in roots.values()),
                isolation=sum(r["isolation"] for r in roots.values()),nominees=counts,steps=toll,
                both_rollouts=sum(len(r["L"]) for r in roots.values()),
                compare={k:contrast(v) for k,v in values.items() if v})


def run(scope,requests,models):
    started=perf_counter()
    roots={}
    for pos in range(0,len(requests),8):
        chunk=requests[pos:pos+8]
        wire=model_payload(models)+[" ".join(map(str,r)) for r in chunk]
        with isolated(scope,ROOT/"build"):
            raw=scope.run("dark-seat-candidates","\n".join(wire)+"\n")
        roots.update(decode(raw,chunk))
    stats=summary(roots)
    stats["seconds"]=perf_counter()-started
    rows=[[*key,{k:[(*x,*v) for x,v in row.items()] if k in ("L","R","G") else row for k,row in r.items()}]
          for key,r in roots.items()]
    return dict(summary=stats,roots=rows)


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-seat-candidates"/os.environ["JUNQI_SERIAL_KEY"])
    if not audit:
        prepare(scope)
    folder=scope.folder
    require(all(sha(folder/n)==x for n,x in read(folder/"fingerprints.json").items()),"seat candidate inputs changed")
    stage="confirmation" if audit else "development"
    models={n:read(folder/(n+".json")) for n in ("original","peer")}
    result=run(scope,[tuple(r) for r in read(folder/"grid.json")[stage]],models)
    write_receipt(folder/(stage+".json"),result)
    if audit:
        stats=result["summary"]
        require(stats["isolation"]>0,"no alternate-board nomination control")
        real=stats["compare"]["original"]["interval"]
        ref=stats["compare"]["reference"]["interval"]
        verdict=("supported" if real[0]>0 and ref[0]>0 else
                 "refuted" if real[1]<0 or ref[1]<0 else "inconclusive")
        print("JUNQI_RESEARCH "+verdict)
    print(json.dumps(result["summary"]))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected seat candidate stage")
    main(bool(sys.argv[1:]))
