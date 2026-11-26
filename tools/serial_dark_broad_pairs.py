"""Train on many shallow or fewer repeated real-position pairs at one simulator budget."""
from collections import Counter
from itertools import product
import json
import math
import os
import shutil
from statistics import mean
import sys
from time import perf_counter

from full_mode_runtime import FullModeRun, isolated, sha
from serial_dark_ordinary_learning import RATE, RULES, UPDATES, assert_layout_separation, run_native, summarize
from serial_dark_replanning import INPUTS, SOURCE, model_payload
from serial_dynamic_reply import ROOT, build, require
from serial_formation_matrix import contrast, score_difference
from serial_preference_retention import np, train
from serial_preference_step_match import parameter_distance, schedule
from serial_worker import read, write_receipt


NAMES=("ordinary","deep","broad","soft")
BUDGET=26000


def prepare(scope):
    folder=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    for n in ("original.json","mixed.json","peer.json","experience.json"):
        shutil.copyfile(SOURCE/n,folder/n)
    grid=dict(pool=[(24152000+i,i%4,4*(i//8%2),(8,20,32,52,76,100)[(i//4+i%4)%6])
                    for i in range(160)],
              spots=[(24153000+i,i%4,4*(i//4%2),(16,36,60,88)[(i//2)%4],3) for i in range(32)],
              games=[(24154000+i,i%4,t) for i,t in product(range(48),(0,4))])
    assert_layout_separation({**grid,"ordinary":[(r["seed"],) for r in read(folder/"experience.json")["examples"][0]]})
    write_receipt(folder/"grid.json",grid)
    write_receipt(folder/"fingerprints.json",{n:sha(folder/n) for n in (*INPUTS,"experience.json")})


def batch(scope,requests,models,budget):
    wire=model_payload(models)+[str(budget)]+[" ".join(map(str,r)) for r in requests]
    start=perf_counter()
    with isolated(scope,ROOT/"build"):
        text=scope.run("dark-broad-pairs","\n".join(wire)+"\n")
    rows={}
    row=None
    leftover=None
    for tag,*fields in map(str.split,text.splitlines()):
        if tag=="F":
            seed,seat,table,ply,old,new,control=map(int,fields)
            key=(seed,seat,table)
            require(key not in rows,"duplicate ordinary pair root")
            row=dict(seed=seed,seat=seat,table=table,ply=ply,moves=[old,new],isolation=control,x=[],pairs=[])
            rows[key]=row
        elif tag=="Q":
            x=list(map(float,fields))
            require(len(x)==44 and all(map(math.isfinite,x)),"bad legal features")
            row["x"].append(x)
        elif tag=="B":
            require(leftover is None,"duplicate budget exit")
            leftover=int(fields[0])
        else:
            i,a,b,x,y=map(int,fields)
            require(tag=="P" and i==len(row["pairs"]) and all(v in (-1,0,1,2) for v in (a,b)) and min(x,y)>0,
                    "invalid fixed-world outcomes")
            row["pairs"].append([a,b,x,y])
    require(list(rows)==[tuple(r[:3]) for r in requests[:len(rows)]],"teacher skipped a chronological root")
    for key,row in rows.items():
        valid=row["moves"][0]>=0
        require(len(row["x"])==(2 if valid else 0) and (valid or not row["pairs"]),"incomplete legal candidate features")
    steps=sum(row["ply"]+sum(x+y for a,b,x,y in row["pairs"]) for row in rows.values())
    require(leftover is not None and steps+leftover==budget,"teacher step ledger does not balance")
    return list(rows.values()),dict(steps=steps,leftover=leftover,roots=len(rows),
        pairs=sum(len(r["pairs"]) for r in rows.values()),isolation=sum(r["isolation"] for r in rows.values()),
        seconds=perf_counter()-start)


def evidence(rows):
    course=[]
    counts=Counter()
    for row in rows:
        lows,highs=[],[]
        for a,b,*_ in row["pairs"]:
            lo,hi=score_difference(a,b)
            lows.append(lo)
            highs.append(hi)
        if not lows or mean(lows)<=0<=mean(highs):
            counts["unresolved"]+=1
        elif row["x"][0]==row["x"][1]:
            counts["identical_features"]+=1
        else:
            counts["labelled"]+=1
            course.append(dict(**row,delta=mean(lows) if mean(lows)>0 else mean(highs),repeats=len(lows)))
    return course,counts


def team_value(model,features):
    x=np.clip((np.asarray(features)-model["mean"])/model["scale"],-6,6)
    w,v=map(np.asarray,model["models"]["collision"]["weights"])
    b,c=map(np.asarray,model["models"]["collision"]["bias"])
    z=(np.tanh(x@w+b)@v+c)[:,:3]
    p=np.exp(z-z.max(axis=1,keepdims=True))
    return (p[:,2]+.5*p[:,1])/p.sum(axis=1)


def pack(examples,parent,deep,broad):
    raw=[r["features"] for r in examples]
    edges={n:[] for n in NAMES}
    for name,course in (("deep",deep),("broad",broad)):
        for r in course:
            index=len(raw)
            raw.extend(r["x"])
            a,b=index,index+1
            edges[name].append((a,b,.65 if r["delta"]>0 else .35))
            if name=="broad":
                q=team_value(parent,r["x"])
                anchor=1/(1+math.exp(-4*(q[1]-q[0])))
                confidence=.35*r["repeats"]/(2+r["repeats"])
                target=(1-confidence)*anchor+confidence*(.5+.5*r["delta"])
                edges["soft"].append((a,b,target))
    y=np.full((len(raw),3),np.nan)
    y[:len(examples)]=np.eye(3)[[r["outcome"] for r in examples]]
    return np.asarray(raw),y,edges


def learn(scope,grids,models):
    folder=scope.folder
    pool=[tuple(r) for r in grids["pool"]]
    hard,cost_d=batch(scope,[(*r,4) for i,r in enumerate(pool) if i%4==i//4%4],models,BUDGET)
    wide,cost_b=batch(scope,[(*r,1) for r in pool],models,BUDGET)
    require(abs(cost_d["steps"]-cost_b["steps"])<.01*BUDGET,"simulator budgets underfilled")
    require(min(cost_d["isolation"],cost_b["isolation"])>0,"no hidden substitution budget case")
    deep,d_counts=evidence(hard)
    broad,b_counts=evidence(wide)
    require(deep and broad,"no paired observations to train")
    parent=models["original"]
    samples=read(folder/"experience.json")["examples"][0]
    raw,y,edges=pack(samples,parent,deep,broad)
    source=dict(mean=parent["mean"],scale=parent["scale"],parameters=parent["models"]["collision"])
    write_receipt(folder/"experience-pairs.json",dict(deep=hard,broad=wide))
    meta={}
    common=None
    for n in NAMES:
        trace={}
        pairs=edges[n]
        child=train(source,(raw,y,pairs,[]),bool(pairs),steps=UPDATES,rate=RATE,step_norms=common,trace=trace)
        if common is None:
            common=schedule(trace)
        require(np.allclose(common,schedule(trace),rtol=1e-10,atol=1e-12),"training used different step norms")
        applied=sum(p["gradient_norm"]>0 for step in trace["trajectory"] for p in step["pairs"])
        distance=parameter_distance(source,child)
        require(distance>0 and (not pairs or applied>0),"no student update")
        meta[n]=dict(ordinary=len(samples),pairs=len(pairs),applied=applied,distance=distance,step_budget=sum(common),updates=UPDATES)
        model=dict(parent,models={"collision":child["parameters"]},training=meta[n],
                   teacher_input=sha(folder/"experience-pairs.json"))
        write_receipt(folder/(n+".json"),model)
    write_receipt(folder/"trained.json",{n:sha(folder/n) for n in (*[k+".json" for k in NAMES],"experience-pairs.json")})
    return dict(budget=BUDGET,source=dict(deep=cost_d,broad=cost_b),filters=dict(deep=d_counts,broad=b_counts),models=meta)


def evaluate(scope,grids,models):
    folder=scope.folder
    spots,spot_cost=batch(scope,[tuple(r) for r in grids["spots"]],models,120000)
    lessons,filters=evidence(spots)
    require(lessons,"no held-out observed action differences")
    write_receipt(folder/"spot-games.json",spots)
    students={n:read(folder/(n+".json")) for n in NAMES}
    direction={}
    for n,m in students.items():
        gap=[np.diff(team_value(m,r["x"]))[0] for r in lessons]
        direction[n]=sum(bool(g*r["delta"]>0) for g,r in zip(gap,lessons,strict=True))
    panel={}
    raw={}
    for n in NAMES[1:]:
        games,_,first,run=run_native(scope,"X",[tuple(r) for r in grids["games"]],
                                      (students["ordinary"],students[n]),peer=models["peer"])
        raw[n]=dict(games=games,first=first,runtime=run)
        panel[n]=summarize(games,first)
    a,b,c=(raw[n]["games"] for n in NAMES[1:])
    require(a[::2]==b[::2]==c[::2],"ordinary confirmation control diverged")
    zipped=zip(b[1::2],c[1::2],strict=True)
    diffs={t:[] for t in (0,4)}
    for old,new in zipped:
        diffs[old["table"]].append((0,0) if old["hash"]==new["hash"] else score_difference(old["result"],new["result"]))
    write_receipt(folder/"student-games.json",raw)
    return dict(panels=panel,soft_vs_broad={t:contrast(v) for t,v in diffs.items()},
                heldout=dict(labels=len(lessons),factors=filters,correct=direction,work=spot_cost))


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-broad-pairs"/os.environ["JUNQI_SERIAL_KEY"])
    folder=scope.folder
    if not audit:
        prepare(scope)
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"ordinary-pair inputs changed")
    grids=read(folder/"grid.json")
    models={n:read(folder/(n+".json")) for n in ("original","peer")}
    if not audit:
        report=learn(scope,grids,models)
        write_receipt(folder/"development.json",report)
    else:
        require(all(sha(folder/n)==v for n,v in read(folder/"trained.json").items()),"broad-pair weights changed")
        report=evaluate(scope,grids,models)
        write_receipt(folder/"confirmation.json",report)
        rival=report["panels"]["soft"][4]["interval"]
        control=report["soft_vs_broad"][4]["interval"]
        verdict=("supported" if min(rival[0],control[0])>0 else
                 "refuted" if rival[1]<0 else "inconclusive")
        print("JUNQI_RESEARCH "+verdict)
    print(json.dumps(report))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected ordinary-pair stage")
    main(bool(sys.argv[1:]))
