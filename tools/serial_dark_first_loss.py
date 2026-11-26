"""Reproduce the recorded first fork, then cross first move with future policy."""
from collections import Counter
from itertools import groupby, product
import json
import os
import shutil
from statistics import mean
import sys

from full_mode_runtime import FullModeRun, isolated, sha
from serial_dark_ordinary_learning import RULES, model_wire
from serial_dynamic_reply import ROOT, build, require
from serial_formation_matrix import contrast, score_difference
from serial_worker import read, write_receipt


PREVIOUS=ROOT/"build/dark-ordinary-inheritance/compare-dark-ordinary-inheritance-4194ef43a0296aff3b07"
REPLICAS=8
CELLS=((0,0),(0,1),(1,0),(1,1))
CONTRASTS={"move_parent":(0,2),"move_child":(1,3),"policy_old_move":(0,1),"policy_new_move":(2,3),"total":(0,3)}


def historical():
    arm=read(PREVIOUS/"confirmation.json")["arms"]["inherit"]
    cases={(r["seed"],r["table"]):r for r in arm["first"]}
    keep=next(r for r in cases.values() if r["parent_result"]==r["child_result"]==2 and r["ply"]>=0)
    picked=[cases[k] for k in ((24112000,0),(24112000,2),(24112001,0))]+[keep]
    games={(r["seed"],r["table"],r["arm"]):r for r in arm["games"]}
    return [dict(r,control=[games[r["seed"],r["table"],a] for a in (2,3)]) for r in picked]


def prepare(scope):
    folder=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    for name in ("generation1","inherit"):
        shutil.copyfile(PREVIOUS/(name+".json"),folder/(name+".json"))
    fresh=[dict(seed=24120000+i,seat=i%4,table=t,ply=-1,parent=-1,child=-1)
           for i,t in product(range(16),(0,2))]
    write_receipt(folder/"manifest.json",dict(development=historical(),confirmation=fresh))
    write_receipt(folder/"fingerprints.json",{p:sha(folder/p) for p in
                  ("generation1.json","inherit.json","manifest.json","engine/deadline_opening.exe","engine/rules.json")})


def parse(raw,cases):
    result={}
    for kind,seed,seat,table,*tail in map(str.split,raw.splitlines()):
        key=(int(seed),int(seat),int(table))
        if kind=="M":
            result[key]=dict(skipped=True)
        elif kind=="F":
            cut,legal,shortlist,old=map(int,tail[:4])
            new=int(tail[6])
            scores=list(map(float,(tail[4],tail[5],tail[7],tail[8])))
            require(2<=shortlist<=legal and old!=new and scores[0]>=scores[2] and scores[1]<=scores[3],"invalid first-fork ranking")
            result[key]=dict(ply=cut,legal=legal,shortlist=shortlist,actions=[old,new],scores=scores,runs=[])
        else:
            require(kind=="R","invalid first-fork receipt")
            replicate,move,policy,score,plies,trace=map(int,tail)
            require(score in (-1,0,1,2) and result[key]["ply"]<plies<=720,"invalid branch result")
            result[key]["runs"].append([replicate,move,policy,score,plies,trace])
    require(result.keys()=={(r["seed"],r["seat"],r["table"]) for r in cases},"unregistered first-fork output")
    for r in cases:
        key=(r["seed"],r["seat"],r["table"])
        f=result[key]
        if "runs" not in f:
            require("control" not in r,"lost historical control case")
            continue
        runs={(i,a,m):(score,plies,trace) for i,a,m,score,plies,trace in f["runs"]}
        require(runs.keys()==set(product(range(REPLICAS+1),range(2),range(2))),"missing factorial results")
        if "control" in r:
            require([runs[0,0,0],runs[0,1,1]]==[(a["result"],a["plies"],a["hash"]) for a in r["control"]],
                    "historical diagonal no longer replays")
        f["deltas"]=contrasts(runs)
    return [dict(seed=s,seat=p,table=t,**row) for (s,p,t),row in result.items()]


def score_delta(a,b):
    return (0,0) if a[2]==b[2] else score_difference(a[0],b[0])


def contrasts(runs):
    differences={k:[] for k in (*CONTRASTS,"interaction")}
    for i in range(1,REPLICAS+1):
        cells=[runs[i,a,p] for a,p in CELLS]
        for key,(a,b) in CONTRASTS.items():
            differences[key].append(score_delta(cells[a],cells[b]))
        a,b=(differences[k][-1] for k in ("move_child","move_parent"))
        differences["interaction"].append((a[0]-b[1],a[1]-b[0]))
    return {k:dict(**contrast(seq),positive=sum(lo>0 for lo,_ in seq),negative=sum(hi<0 for _,hi in seq))
            for k,seq in differences.items()}


def summarize(roots):
    active=[r for r in roots if "runs" in r]
    table={}
    for desk in sorted({r["table"] for r in active}):
        selected=[r for r in active if r["table"]==desk]
        table[desk]={k:contrast([r["deltas"][k]["bounds"] for r in selected]) for k in (*CONTRASTS,"interaction")}
    groups=[list(rows) for _,rows in groupby(sorted(active,key=lambda r:r["seed"]),key=lambda r:r["seed"])]
    aggregate={k:contrast([tuple(map(mean,zip(*(r["deltas"][k]["bounds"] for r in g),strict=True))) for g in groups])
               for k in (*CONTRASTS,"interaction")} if groups else {}
    return dict(roots=len(roots),diverged=len(active),layout_groups=len(groups),cells=dict(Counter(r[-3] for g in active for r in g["runs"])),
                table=table,aggregate=aggregate)


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-first-loss"/os.environ["JUNQI_SERIAL_KEY"])
    folder=scope.folder
    if not audit:
        prepare(scope)
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"causal audit inputs changed")
    stage="confirmation" if audit else "development"
    cases=read(folder/"manifest.json")[stage]
    fields=("seed","seat","table","ply","parent","child")
    wire=[model_wire(read(folder/(n+".json"))) for n in ("generation1","inherit")]
    wire.extend(" ".join(map(str,(*(r[k] for k in fields),REPLICAS))) for r in cases)
    with isolated(scope,ROOT/"build"):
        raw=scope.run("dark-first-loss","\n".join(wire)+"\n")
    roots=parse(raw,cases)
    stats=summarize(roots)
    write_receipt(folder/(stage+".json"),dict(roots=roots,summary=stats))
    if audit:
        stable=any(r["interval"][0]>0 or r["interval"][1]<0 for r in stats["aggregate"].values())
        print("JUNQI_RESEARCH "+("supported" if stable else "inconclusive"))
    print(json.dumps(stats))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected first-loss audit stage")
    main(bool(sys.argv[1:]))
