"""Train a bounded full-observation student on certified choices in fresh geometric groups."""
from collections import Counter, defaultdict
import hashlib
import json
import os
import sys

from compare_book_learning import OUTPUTS, fit, np
from serial_evaluator_panel import ROOT, build, model_roster, native, require
from serial_formation_matrix import contrast, score_difference
from serial_worker import read, write_receipt

SCHEMAS={"board":"junqi-public-early-action-collision-v1","window":"junqi-full-reply-window-action-v1"}
DIMENSIONS={"board":44,"window":52}
HIDDEN=24
GAME_FIELDS=("seed seat table arm plies outcome opportunities changed work extended proved_better proved_worse "
             "micros first_change root_hash old_move new_move final_hash").split()
GAME_NAMES=("path","extension","student44","student52")


def examples(folder):
    raw=native(folder,"reply-window-examples")
    cases=[]
    unknown=0
    require(raw.splitlines()[0]=="C 1","full/dark corridor control failed")
    for line in raw.splitlines()[1:]:
        tag,*fields=line.split()
        require(tag=="P" and len(fields)==115,"invalid certified action pair")
        flag,trap,gate,source,defender,a,b,alo,ahi,blo,bhi=map(int,fields[:11])
        if alo!=ahi or blo!=bhi:
            unknown+=1
            continue
        require(alo!=blo and 0<=alo<=2 and 0<=blo<=2 and a!=b,"indistinguishable certified pair")
        features=[tuple(map(float,fields[11+i*52:11+(i+1)*52])) for i in range(2)]
        cases.append(dict(group=[flag,trap,gate,source],defender=defender,actions=[a,b],
                          scores=[alo,blo],preferred=int(blo>alo),features=features))
    require(len({(tuple(r["group"]),r["defender"]) for r in cases})==len(cases),"duplicate generated case")
    return cases,unknown,hashlib.sha256(raw.encode()).hexdigest()


def aliases(cases,width):
    same=defaultdict(set)
    for r in cases:
        same[tuple(x[:width] for x in r["features"])].add(r["preferred"])
    return sum(len(v)==2 for v in same.values())


def split(cases):
    groups=defaultdict(list)
    for case in cases:
        groups[tuple(case["group"])].append(case)
    # Hold out bomb collisions, temporarily blocked allies, and the other flag side.
    train=[r for k,v in groups.items() if k[0] in (0,56) for r in v]
    test=[r for k,v in groups.items() if k[0] in (1,2,58) for r in v]
    require(all(len(set(r["preferred"] for r in rows))==2 for rows in (train,test)) and len(groups)>=4,
            "positive and negative independent holdout shapes are required")
    return train,test


def model_wire(model,name):
    width=DIMENSIONS[name]
    weights,bias=model["parameters"]["weights"],model["parameters"]["bias"]
    numbers=[SCHEMAS[name],width,len(OUTPUTS),HIDDEN,*model["mean"],*model["scale"],
             *(v for row in weights[0] for v in row),*bias[0],*(v for row in weights[1] for v in row),*bias[1]]
    return " ".join(map(str,numbers))


def fit_models(records):
    y=np.full((2*len(records),len(OUTPUTS)),np.nan)
    y[:,:3]=0
    for i,score in enumerate(score for r in records for score in r["scores"]):
        y[i,score]=1
    rng=np.random.default_rng(23)
    first=rng.normal(0,.12/44**.5,size=(44,HIDDEN))
    last=rng.normal(0,.12/HIDDEN**.5,size=(HIDDEN,len(OUTPUTS)))
    bias=[np.zeros(HIDDEN),np.zeros(len(OUTPUTS))]
    models={}
    for name,width in DIMENSIONS.items():
        x=np.asarray([features[:width] for r in records for features in r["features"]])
        mean=x.mean(axis=0)
        scale=np.where(x.std(axis=0)>.025,x.std(axis=0),1.)
        norm=np.clip((x-mean)/scale,-6,6)
        inner=np.pad(first,((0,width-44),(0,0)))
        initial=dict(weights=[inner,last],bias=bias)
        models[name]=dict(mean=mean.tolist(),scale=scale.tolist(),parameters=fit(norm,y,23,HIDDEN,initial))
    return models


def judge(folder,artifact,raw,test):
    results={}
    for name,width in DIMENSIONS.items():
        payload=model_wire(artifact["models"][name],name)+"\n"+str(len(test))+"\n"
        payload+="\n".join(" ".join(map(str,x[:width])) for r in test for x in r["features"])
        rows=native(folder,"reply-window-judge-"+name,payload).splitlines()
        require(len(rows)==len(test),"native teacher-off judge dropped a pair")
        counts=Counter()
        mistakes=[]
        timing=Counter()
        for example,row in zip(test,rows,strict=True):
            tag,chosen,a,b=row.split()
            picked=int(chosen)
            require(tag=="J" and picked in (0,1),"invalid loaded student choice")
            counts["correct"]+=picked==example["preferred"]
            counts["tie"]+=abs(float(a)-float(b))<=1e-12
            counts["chosen_points"]+=example["scores"][picked]
            counts[f"correct_{example['preferred']}"]+=picked==example["preferred"]
            variant=example["group"][0]
            if variant in (1,2):
                label="bomb_timing" if variant==1 else "blocked_ally"
                timing[label]+=1
                counts[label]+=picked==example["preferred"]
            if picked!=example["preferred"]:
                mistakes.append(dict(group=example["group"],defender=example["defender"]))
        results[name]=dict(counts=counts,mistakes=mistakes,timing_pairs=timing)
    return dict(test_pairs=len(test),oracle=sum(max(r["scores"]) for r in test),data_sha=raw,
                errors=results,first_action=sum(r["preferred"]==0 for r in test))


def games(folder,artifact,audit):
    first=2767000 if audit else 2766000
    count=16
    roster=model_roster()["historical"]
    wire=" ".join(map(str,(first,count,*roster["weights"],0)))
    wire+=" "+model_wire(artifact["models"]["board"],"board")
    wire+=" "+model_wire(artifact["models"]["window"],"window")
    rows=[]
    for line in native(folder,"reply-window-games",wire).splitlines():
        tag,*fields=line.split()
        require(tag=="G","unexpected game entry output")
        rows.append(dict(zip(GAME_FIELDS,map(int,fields),strict=True)))
    require([(r["seed"],r["seat"],r["table"],r["arm"]) for r in rows]==[
        (s,s%4,t,a) for s in range(first,first+count) for t in range(2) for a in range(4)],"game sets changed")
    summaries={}
    for table in (0,1):
        diffs={name:[] for name in GAME_NAMES}
        for s in range(first,first+count):
            group=[r for r in rows if r["seed"]==s and r["table"]==table]
            old=group[0]
            for name,row in zip(GAME_NAMES,group,strict=True):
                require(0<=row["changed"]<=row["opportunities"]<=2 and row["outcome"] in (-1,0,1,2),
                        "bad real decision telemetry")
                if row["changed"]==0:
                    require(row["final_hash"]==old["final_hash"] and row["plies"]==old["plies"],
                            "the same selections changed the game")
                diffs[name].append((0.,0.) if row["changed"]==0 else score_difference(old["outcome"],row["outcome"]))
        cell={}
        for arm,name in enumerate(GAME_NAMES):
            chosen=[r for r in rows if r["table"]==table and r["arm"]==arm]
            counts=Counter({field:sum(r[field] for r in chosen) for field in (
                "opportunities","changed","work","extended","proved_better","proved_worse","micros")})
            for r in chosen:
                counts[{-1:"unresolved",0:"loss",1:"draw",2:"win"}[r["outcome"]]]+=1
                counts["win100"]+=r["outcome"]==2 and r["plies"]<=100
                counts["win140"]+=r["outcome"]==2 and r["plies"]<=140
                counts["loss140"]+=r["outcome"]==0 and r["plies"]<=140
            for lo,hi in diffs[name]:
                counts["better_games"]+=lo>0
                counts["worse_games"]+=hi<0
            cell[name]=dict(counts=counts,difference=contrast(diffs[name]),mean_plies=sum(r["plies"] for r in chosen)/count)
        summaries[str(table)]=cell
    return dict(rows=rows,summary=summaries,opponent_hash=roster["artifact_sha256"])


def main(audit):
    folder=ROOT/"build/reply-window"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        folder.mkdir(parents=True,exist_ok=True)
        build(folder)
    cases,unknown,source=examples(folder)
    train,test=split(cases)
    diagnostics=dict(pairs=len(cases),ignored_bounds=unknown,train=len(train),test=len(test),
                     train_groups=len({tuple(r["group"]) for r in train}),
                     test_groups=len({tuple(r["group"]) for r in test}),
                     test_targets=dict(Counter(r["preferred"] for r in test)),
                     aliases_board=aliases(cases,44),aliases_window=aliases(cases,52))
    require(aliases(test,44)>0 and aliases(test,52)==0,"held-out timing example failed to isolate observation")
    print(json.dumps(diagnostics))
    if not audit:
        write_receipt(folder/"students.json",dict(models=fit_models(train),source=source,diagnostics=diagnostics))
    artifact=read(folder/"students.json")
    require(artifact["source"]==source and artifact["diagnostics"]["pairs"]==len(cases),"proof data drift")
    score=judge(folder,artifact,source,test)
    if audit:
        require(read(folder/"evaluation.json")==json.loads(json.dumps(score)),
                "loaded student selection was not reproducible")
    else:
        write_receipt(folder/"evaluation.json",score)
    actual=games(folder,artifact,audit)
    write_receipt(folder/("games-confirmation.json" if audit else "games-development.json"),actual)
    if audit:
        previous=read(folder/"games-development.json")
        require(previous["opponent_hash"]==actual["opponent_hash"] and
                {r["seed"] for r in previous["rows"]}.isdisjoint(r["seed"] for r in actual["rows"]),
                "opening confirmation was not independent")
        print("JUNQI_RESEARCH inconclusive")
    print(json.dumps({key:v["counts"] for key,v in score["errors"].items()}))
    print(json.dumps(actual["summary"]))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
