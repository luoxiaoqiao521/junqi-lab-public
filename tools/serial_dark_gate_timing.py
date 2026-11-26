"""Trace one planned route through actual opponent and ally turns."""
from collections import Counter
from itertools import product
import json
import os
import shutil
from statistics import mean
import sys
from time import perf_counter

from full_mode_runtime import FullModeRun, isolated, sha
from serial_dark_ordinary_learning import RULES, assert_layout_separation
from serial_dark_replanning import INPUTS, SOURCE, model_payload
from serial_dynamic_reply import ROOT, build, require
from serial_formation_matrix import contrast, score_difference
from serial_worker import read, write_receipt


FIELDS=("natural","arrival","provisional","persistent","restored","occupy_only","probes","steps","unknown","settled","retired")


def prepare(scope):
    folder=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    for n in ("original.json","mixed.json","peer.json"):
        shutil.copyfile(SOURCE/n,folder/n)
    grid={phase:[(start+i,i%4,(i//4%2)*4,(12,24,36,52,72,88)[i%6]) for i in range(count)]
          for phase,start,count in (("development",24155000,24),("confirmation",24156000,64))}
    assert_layout_separation(grid)
    write_receipt(folder/"grid.json",grid)
    write_receipt(folder/"fingerprints.json",{n:sha(folder/n) for n in INPUTS})


def decode(raw,grid):
    roots={}
    root=None
    for tag,*fields in map(str.split,raw.splitlines()):
        numbers=list(map(int,fields))
        if tag=="F":
            key=tuple(numbers[:3])
            ply,status,*detail=numbers[3:]
            require(key not in roots and (len(detail)==5 if status==3 else not detail),"invalid gate root")
            root=dict(ply=ply,status=status,timing={},games={})
            if status==3:
                a,b,x,y,control=detail
                require(a!=b and x!=y and control in (0,1),"gate contrast missing")
                root.update(actions=[a,b],gates=[x,y],isolation=control)
            roots[key]=root
        elif tag=="T":
            a,w,*values=numbers
            row=dict(zip(FIELDS,values,strict=True))
            require((a,w) not in root["timing"] and -4<=row["natural"]<=4 and all(-1<=row[k]<=4 for k in FIELDS[1:4])
                    and 0<=sum(row[k] for k in ("restored","unknown","settled","retired"))<=row["probes"]<=row["steps"],
                    "invalid timed conditions")
            root["timing"][a,w]=row
        elif tag=="C":
            require("choice" not in root and numbers[0] in (0,1),"invalid gate choice")
            root["choice"]=numbers[0]
        else:
            a,w,i,result,steps=numbers
            require(tag=="R" and (a,w,i) not in root["games"] and result in (-1,0,1,2) and steps>0,"bad gate continuation")
            root["games"][a,w,i]=(result,steps)
    require(list(roots)==[r[:3] for r in grid],"gate assignments differ")
    return roots


def audit(roots):
    counts=Counter()
    scores={k:[] for k in ("all_actual","covered_actual","heldout")}
    for root in roots.values():
        if root["status"]!=3:
            scores["all_actual"].append((0,0))
            continue
        timings=root["timing"]
        games=root["games"]
        require(timings.keys()==set(product((0,1),(0,1))) and games.keys()==set(product((0,1),(-1,2,3),range(4))),
                "incomplete timing/evaluation pairs")
        alt=root["choice"]
        counts["changed"]+=alt
        counts["isolation"]+=root["isolation"]
        for row in timings.values():
            counts["natural_pass"]+=row["natural"]>0
            counts["gate_arrival"]+=row["arrival"]>0
            counts["earliest_candidate_occupy_only"]+=row["occupy_only"]
            counts["provisional"]+=row["provisional"]>0
            counts["persistent"]+=row["persistent"]>0
            counts["restored_branches"]+=row["restored"]
            counts["probe_steps"]+=row["steps"]
            counts["unresolved_branches"]+=row["unknown"]
            counts["settled_branches"]+=row["settled"]
            counts["retired_branches"]+=row["retired"]
            for n,v in (("horizon",-2),("settled",-3),("retired",-4)):
                counts["natural_"+n]+=row["natural"]==v
        for tag,worlds in (("covered_actual",(-1,)),("heldout",(2,3))):
            diff=[score_difference(games[0,w,i][0],games[1,w,i][0]) for w,i in product(worlds,range(4))]
            delta=tuple(map(mean,zip(*diff,strict=True))) if alt else (0,0)
            scores[tag].append(delta)
            if tag=="covered_actual":
                scores["all_actual"].append(delta)
                if alt:
                    counts["win_to_loss"]+=sum(games[0,-1,i][0]==2 and games[1,-1,i][0]==0 for i in range(4))
    return dict(assigned=len(roots),coverage={s:sum(r["status"]==s for r in roots.values()) for s in range(4)},
                timing=counts,score={n:contrast(v) for n,v in scores.items() if v})


def run(scope,grid,models):
    start=perf_counter()
    roots={}
    for n in range(0,len(grid),8):
        chunk=grid[n:n+8]
        wire=model_payload(models)+[" ".join(map(str,r)) for r in chunk]
        with isolated(scope,ROOT/"build"):
            raw=scope.run("dark-gate-timing","\n".join(wire)+"\n")
        roots.update(decode(raw,chunk))
    summary=audit(roots)
    summary["seconds"]=perf_counter()-start
    rows=[[*k,{name:[(*key,value) for key,value in v.items()] if name in ("timing","games") else v
               for name,v in root.items()}] for k,root in roots.items()]
    return dict(summary=summary,roots=rows)


def main(audit_stage):
    scope=FullModeRun(ROOT/"build/dark-gate-timing"/os.environ["JUNQI_SERIAL_KEY"])
    folder=scope.folder
    if not audit_stage:
        prepare(scope)
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"gate inputs changed")
    phase="confirmation" if audit_stage else "development"
    rows=[tuple(r) for r in read(folder/"grid.json")[phase]]
    models={n:read(folder/(n+".json")) for n in ("original","peer")}
    results=run(scope,rows,models)
    write_receipt(folder/(phase+".json"),results)
    if audit_stage:
        limits=results["summary"]["score"]
        original=limits["all_actual"]["interval"]
        alternate=limits.get("heldout",{}).get("interval",[-1,1])
        verdict=("supported" if min(original[0],alternate[0])>0 else
                 "refuted" if max(original[1],alternate[1])<0 else "inconclusive")
        print("JUNQI_RESEARCH "+verdict)
    print(json.dumps(results["summary"]))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected gate stage")
    main(bool(sys.argv[1:]))
