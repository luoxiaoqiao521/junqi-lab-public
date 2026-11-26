"""Calibrate frozen tables, train one four-dark student and confirm actual whole-game decisions."""
from collections import Counter
from hashlib import sha256
from itertools import product
import json
from math import isfinite
import os
import shutil
import sys
from time import perf_counter

from compare_book_learning import OUTPUTS, fit, np, probabilities
from full_mode_runtime import FullModeRun, isolated, sha
from serial_dark_search_budget import CALIBRATION
from serial_dynamic_reply import ROOT, build, require
from serial_formation_matrix import contrast, score_difference
from serial_worker import read, write_receipt
from train_early_deadline_student import RULES, model_arguments


FIELDS="seed seat table arm result plies hash turns changed first base selected examples".split()
UPDATES,RATE=48,.025
SAMPLE_MODES=("D","L","M")
PAIR_MODES=("E","H","X")
PEER_MODES=("H","M","X")


def assert_layout_separation(grids):
    uses={}
    for phase,rows in grids.items():
        for seed,*_ in rows:
            require(uses.setdefault(seed,phase)==phase,"ordinary layout crosses phase boundaries")


def data_health(games,examples):
    keys=FIELDS[:4]
    indexed={tuple(r[k] for k in keys):r for r in games}
    require(len(indexed)==len(games),"duplicate ordinary game")
    samples={key:[] for key in indexed}
    stages=Counter()
    for row in examples:
        key=tuple(row[k] for k in keys)
        require(key in indexed and row["outcome"]==indexed[key]["result"],"ordinary label disagrees with terminal receipt")
        x=row["features"]
        require(len(x)==44 and all(map(isfinite,x)),"nonfinite or wrong-width ordinary feature")
        step=140*x[0]
        ply=round(step)
        require(abs(step-ply)<1e-7 and 0<=ply<indexed[key]["plies"],"ordinary sample is outside its trajectory")
        series=samples[key]
        require(not series or ply-series[-1]>=24,"duplicate or out-of-contract ordinary decision")
        series.append(ply)
        stages[min(ply//24,6)]+=1
    require(all(len(series)==indexed[key]["examples"]<=6 for key,series in samples.items()),"ordinary sample count changed")
    return dict(games=len(games),rows=len(examples),by_ply_band=dict(stages),
                by_result=dict(Counter(r["outcome"] for r in examples)),
                by_behavior=dict(Counter("student" if r["arm"]==2 else "tactical" for r in examples)),
                search_teacher_calls=0,search_teacher_labels=0)


def score_transitions(pairs):
    matrix={a:{b:0 for b in range(-1,3)} for a in range(-1,3)}
    for old,new in pairs:
        matrix[old["result"]][new["result"]]+=1
    return matrix


def model_wire(model):
    return model_arguments(model,"collision",model["encoding"])


def run_native(scope,mode,grid,models=(),*,peer=None):
    wires=list(map(model_wire,models))
    require((mode in PEER_MODES)==(peer is not None),"historical peer does not match its comparison")
    if peer is not None:
        wires.append(model_arguments(peer,"small_nn"))
    wire="\n".join([mode,*wires,*(" ".join(map(str,row)) for row in grid)])+"\n"
    started=perf_counter()
    with isolated(scope,ROOT/"build"):
        raw=scope.run("dark-ordinary-learning",wire)
    rows,examples,divergences=parse(raw,mode,grid)
    metrics=dict(seconds=perf_counter()-started,output_sha256=sha256(raw.encode()).hexdigest())
    if mode in SAMPLE_MODES:
        metrics["data_health"]=data_health(rows,examples)
    return rows,examples,divergences,metrics


def parse(raw,mode,grid):
    rows,examples,divergences=[],[],[]
    schema=FIELDS+["peer_turns","peer_changes"] if mode in PEER_MODES else FIELDS
    for tag,*fields in map(str.split,raw.splitlines()):
        if tag=="G":
            rows.append(dict(zip(schema,map(int,fields),strict=True)))
        elif tag=="E":
            seed,seat,table,arm,outcome=map(int,fields[:5])
            features=list(map(float,fields[5:]))
            require(len(features)==44 and np.isfinite(features).all(),"bad public action features")
            examples.append(dict(seed=seed,seat=seat,table=table,arm=arm,outcome=outcome,features=features))
        else:
            require(tag=="R","unknown ordinary game receipt")
            divergences.append(dict(zip(("seed","seat","table","ply","parent","child"),map(int,fields),strict=True)))
    arms=(0,1) if mode=="P" else (2,3)
    expected=[(*key,a) for key in grid for a in ((2,) if mode=="Y" else
              (2 if key[0]%2==0 else 1,) if mode in SAMPLE_MODES else arms)]
    require([tuple(r[k] for k in FIELDS[:4]) for r in rows]==expected,"ordinary game assignments differ")
    counts=Counter(tuple(r[k] for k in FIELDS[:4]) for r in examples)
    outcomes={tuple(r[k] for k in FIELDS[:4]):r["result"] for r in rows}
    require(all(r["examples"]==counts[tuple(r[k] for k in FIELDS[:4])] and
                -1<=r["result"]<=2 and 0<r["plies"]<=720 and r["turns"]>=r["changed"] for r in rows),
            "bad game result or example accounting")
    require(all(outcomes[tuple(r[k] for k in FIELDS[:4])]==r["outcome"] for r in examples),"invalid terminal label provenance")
    require([tuple(r[k] for k in FIELDS[:3]) for r in divergences]==(grid if mode in PAIR_MODES else []),
            "missing first divergence")
    if mode in PEER_MODES:
        active_tables={"H":(1,2,3),"M":(1,),"X":(1,3,4)}[mode]
        require(all(0<=r["peer_changes"]<=r["peer_turns"]<=r["plies"] and
                    (r["table"] in active_tables or not r["peer_turns"])
                    for r in rows),"historical peer accounting diverged from assigned tables")
    return rows,examples,divergences


def prepare(scope):
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    folder=scope.folder
    shutil.copyfile(CALIBRATION/"student.json",folder/"parent.json")
    grids={"pilot":[(24100000+i,i%4,t) for i,t in product(range(16),range(4))],
           "train":[(24101000+i,(i//8)%4,(i//2)%4) for i in range(96)],
           "value_test":[(24102000+i,(i//8)%4,(i//2)%4) for i in range(32)],
           "development":[(24103000+i,i%4,t) for i,t in product(range(16),range(4))],
           "confirmation":[(24104000+i,i%4,t) for i,t in product(range(32),range(4))]}
    write_receipt(folder/"manifest.json",grids)
    write_receipt(folder/"fingerprints.json",{n:sha(folder/n) for n in
                  ("manifest.json","parent.json","engine/deadline_opening.exe","engine/rules.json")})


def summarize(rows,divergences=None):
    panel={}
    paired=list(zip(rows[::2],rows[1::2],strict=True))
    if divergences is not None:
        for (a,b),r in zip(paired,divergences,strict=True):
            same=a["hash"]==b["hash"]
            require(same==(r["ply"]<0) and (not same or (a["result"],a["plies"])==(b["result"],b["plies"])),
                    "unchanged decisions produced a different game")
            r.update(parent_result=a["result"],child_result=b["result"])
    for t in sorted({a["table"] for a,_ in paired}):
        pairs=[(a,b) for a,b in paired if a["table"]==t]
        diffs=[(0,0) if a["hash"]==b["hash"] else score_difference(a["result"],b["result"]) for a,b in pairs]
        panel[t]=dict(**contrast(diffs),parent=dict(Counter(a["result"] for a,_ in pairs)),
                      candidate=dict(Counter(b["result"] for _,b in pairs)),
                      transitions=score_transitions(pairs),
                      changed=sum(a["hash"]!=b["hash"] for a,b in pairs),
                      better=sum(lo>0 for lo,_ in diffs),worse=sum(hi<0 for _,hi in diffs))
    return panel


def examples_xy(data,parent):
    rows=[r for r in data if r["outcome"]>=0]
    require(len(rows)>0,"no finished ordinary game labels")
    x=np.clip((np.asarray([r["features"] for r in rows])-parent["mean"])/parent["scale"],-6,6)
    y=np.full((len(rows),len(OUTPUTS)),np.nan)
    y[:,:3]=np.eye(3)[[r["outcome"] for r in rows]]
    return x,y


def value_error(data,artifact):
    x,y=examples_xy(data,artifact)
    p=artifact["models"]["collision"]
    hidden=np.tanh(x@np.asarray(p["weights"][0])+p["bias"][0])
    prob=probabilities(hidden@np.asarray(p["weights"][-1])+p["bias"][-1])[:,:3]
    return float(np.square(prob-y[:,:3]).sum(axis=1).mean())


def learn(scope,grids,parent):
    folder=scope.folder
    games,examples,_,run=run_native(scope,"D",grids["train"],(parent,))
    x,y=examples_xy(examples,parent)
    require(len({r["result"] for r in games if r["result"]>=0})>=2,"ordinary outcome labels have no variation")
    training=dict(updates=UPDATES,rate=RATE,rows=len(x),ignored_rows=len(examples)-len(x),
                  game_results=dict(Counter(g["result"] for g in games)),**run)
    learned=fit(x,y,48119,24,parent["models"]["collision"],updates=UPDATES,rate=RATE)
    artifact=dict(parent,models={"collision":learned},parent_sha256=sha(folder/"parent.json"),training=training)
    write_receipt(folder/"student.json",artifact)
    write_receipt(folder/"training.json",dict(games=games,examples=examples))
    require(model_wire(read(folder/"student.json"))!=model_wire(parent),"no updated student parameters")
    write_receipt(folder/"student-fingerprint.json",dict(model=sha(folder/"student.json")))


def main(stage):
    scope=FullModeRun(ROOT/"build/dark-ordinary-learning"/os.environ["JUNQI_SERIAL_KEY"])
    folder=scope.folder
    if stage=="pilot":
        prepare(scope)
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"frozen ordinary learning inputs changed")
    grids={k:[tuple(r) for r in v] for k,v in read(folder/"manifest.json").items()}
    assert_layout_separation(grids)
    if stage=="pilot":
        rows,_,_,run=run_native(scope,"P",grids[stage])
        result=dict(games=rows,summary=summarize(rows),**run)
    else:
        parent=read(folder/"parent.json")
        if stage=="development":
            learn(scope,grids,parent)
        require(sha(folder/"student.json")==read(folder/"student-fingerprint.json")["model"],"saved student changed")
        child=read(folder/"student.json")
        require(child["parent_sha256"]==sha(folder/"parent.json"),"student belongs to another parent")
        rows,_,differences,run=run_native(scope,"E",grids[stage],(parent,child))
        result=dict(games=rows,first_divergences=differences,summary=summarize(rows,differences),**run)
        if stage=="confirmation":
            _,labels,_,_=run_native(scope,"D",grids["value_test"],(parent,))
            result["observed_action_brier"]={name:value_error(labels,artifact) for name,artifact in (("parent",parent),("child",child))}
            summary=result["summary"]
            verdict="supported" if all(c["interval"][0]>0 for c in summary.values()) else (
                "refuted" if any(c["interval"][1]<0 for c in summary.values()) else "inconclusive")
            print("JUNQI_RESEARCH "+verdict)
    write_receipt(folder/(stage+".json"),result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("games","first_divergences")}))


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected an ordinary learning stage")
    for stage in (("confirmation",) if sys.argv[1:] else ("pilot","development")):
        main(stage)
    print("JUNQI_VERDICT accepted")
