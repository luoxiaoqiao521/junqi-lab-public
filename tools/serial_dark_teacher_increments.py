"""Measure whether archived teacher preferences add value beyond ordinary four-dark returns."""
from collections import Counter
from itertools import product
import json
import os
import shutil
from statistics import mean
import sys
from time import perf_counter

from full_mode_runtime import FullModeRun, isolated, sha
from serial_dark_ordinary_inheritance import by_layout, difference
from serial_dark_ordinary_learning import RATE, RULES, UPDATES, assert_layout_separation, data_health, model_wire, run_native, summarize
from serial_dark_reply_leaf import explain, parse
from serial_dynamic_reply import ROOT, build, require
from serial_formation_matrix import contrast
from serial_preference_retention import np, train
from serial_preference_step_match import parameter_distance, schedule
from serial_worker import read, write_receipt


ORDINARY=ROOT/"build/dark-ordinary-inheritance/compare-dark-ordinary-inheritance-4194ef43a0296aff3b07"
ARCHIVE=ROOT/"build/dark-reply-leaf/audit-dark-response-versus-leaf-value-rescreen-e8b9bc9e8f66f9515c29"
ARMS=("direct","old","new")
TEACHERS={"old":"natural_leaf","new":"pressure_final"}
INPUTS=("training.json","generation1.json","archive.json","replay-0.txt","replay-1.txt",
        "engine/deadline_opening.exe","engine/rules.json","grid.json")


def prepare(scope):
    folder=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    for source,name,target in ((ORDINARY,"training.json","training.json"),(ORDINARY,"generation1.json","generation1.json"),
                               (ARCHIVE,"manifest.json","archive.json"),(ARCHIVE,"replay-0.txt","replay-0.txt"),
                               (ARCHIVE,"replay-1.txt","replay-1.txt")):
        shutil.copyfile(source/name,folder/target)
    cohorts=read(folder/"archive.json")["cohorts"]
    require(len({r["family"] for fold in cohorts for r in fold})==sum(map(len,cohorts)),"archive family crosses cohorts")
    grids={phase:[(base+i,i%4,t) for i,t in product(range(count),range(4))]
           for phase,base,count in (("development",24121000,16),("confirmation",24122000,32))}
    assert_layout_separation(dict(grids,training=[(r["seed"],) for r in read(folder/"training.json")["games"]]))
    write_receipt(folder/"grid.json",grids)
    write_receipt(folder/"fingerprints.json",{p:sha(folder/p) for p in INPUTS})


def direction(bounds):
    return 1 if bounds[0]>1e-8 else (0 if bounds[1]<-1e-8 else None)


def teach(scope,parent,fold):
    folder=scope.folder
    payload=model_wire(parent)+"\n"+(folder/f"replay-{fold}.txt").read_text(encoding="ascii")
    begin=perf_counter()
    with isolated(scope,ROOT/"build"):
        raw=scope.run("dark-reply-course",payload)
    seconds=perf_counter()-begin
    roots=parse(raw,read(folder/"archive.json")["cohorts"][fold])
    rows=[]
    costs={arm:Counter() for arm in TEACHERS}
    for key,root in roots.items():
        if "skipped" in root:
            continue
        tables=explain(root)
        decisions={}
        for index,(arm,teacher) in enumerate(TEACHERS.items()):
            bounds=tuple(map(mean,zip(*(t["delta"][teacher] for t in tables.values()),strict=True)))
            decisions[arm]=dict(bounds=bounds,pick=direction(bounds))
            used=[cell for (_,_,k,_),cell in root["cells"].items() if k==index]
            cost=costs[arm]
            cost["checkpoints"]+=len(used)
            cost["short_steps"]+=sum(cell["plies"] for cell in used)
            if index==1:
                cost["terminal_runs"]+=2*len(used)
                cost["additional_steps"]+=sum(sum(cell["extra"]) for cell in used)
                cost["truncated"]+=sum(v<0 for cell in used for v in cell["final"])
        referee=tuple(map(mean,zip(*(t["referee"] for t in tables.values()),strict=True)))
        rows.append(dict(key=key,moves=root["moves"],features=[root["features"][m] for m in root["moves"]],
                         teachers=decisions,referee=referee,referee_pick=direction(referee)))
    matched=[r for r in rows if all(d["pick"] is not None for d in r["teachers"].values())]
    summary=dict(roots=len(roots),recovered=len(rows),paired=len(matched),joint_seconds=seconds,
                 unequal=sum(r["teachers"]["old"]["pick"]!=r["teachers"]["new"]["pick"] for r in matched),
                 skipped=dict(Counter(r["skipped"] for r in roots.values() if "skipped" in r)),teacher_cost=costs)
    for arm in TEACHERS:
        summary[arm]=dict(Counter("unresolved" if r["referee_pick"] is None else
                                  ("agreement" if r["teachers"][arm]["pick"]==r["referee_pick"] else "contradiction")
                                  for r in matched))
    write_receipt(folder/f"teacher-{fold}.json",dict(rows=rows,summary=summary))
    return matched,summary


def learn(folder,parent,matched):
    require(matched,"neither teacher produced a common consumable course")
    data=read(folder/"training.json")
    provenance=data_health(data["games"],data["examples"])
    ordinary=[r for r in data["examples"] if r["outcome"]>=0]
    raw=np.asarray([r["features"] for r in ordinary]+[x for r in matched for x in r["features"]])
    y=np.full((len(raw),3),np.nan)
    y[:len(ordinary)]=np.eye(3)[[r["outcome"] for r in ordinary]]
    source=dict(mean=parent["mean"],scale=parent["scale"],parameters=parent["models"]["collision"])
    sizes=None
    for arm in ARMS:
        pairs=[]
        if arm in TEACHERS:
            for i,row in enumerate(matched):
                start=len(ordinary)+2*i
                chosen=row["teachers"][arm]["pick"]
                pairs.append((start+1-chosen,start+chosen,.65))
        trace={}
        begin=perf_counter()
        updated=train(source,(raw,y,pairs,[]),bool(pairs),steps=UPDATES,rate=RATE,log_odds=True,trace=trace,step_norms=sizes)
        if sizes is None:
            sizes=schedule(trace)
        require(np.allclose(sizes,schedule(trace),rtol=1e-10,atol=1e-12),"parameter-step budgets differ")
        consumed=[sum(pair["gradient_norm"]>0 for pair in step["pairs"]) for step in trace["trajectory"]]
        distance=parameter_distance(source,updated)
        require(distance>0 and (not pairs or sum(consumed)>0),"ordinary or teacher objective failed to produce a gradient")
        stats=dict(ordinary=provenance,ordinary_consumed=len(ordinary),pair_labels=len(pairs),
                   pair_gradient_applications=sum(consumed),updates=UPDATES,step_budget=sum(sizes),distance=distance,
                   seconds=perf_counter()-begin,clipped_features=trace["clipped_features"])
        artifact=dict(parent,models={"collision":updated["parameters"]},training=stats,
                      parent_sha256=sha(folder/"generation1.json"),ordinary_source=sha(folder/"training.json"),
                      teacher_course=sha(folder/"teacher-0.json"))
        write_receipt(folder/(arm+".json"),artifact)
    names=("teacher-0.json",*(a+".json" for a in ARMS))
    write_receipt(folder/"trained.json",{n:sha(folder/n) for n in names})


def evaluate(scope,parent,grid):
    evaluated={}
    for arm in ARMS:
        model=read(scope.folder/(arm+".json"))
        games,_,first,run=run_native(scope,"E",grid,(parent,model))
        evaluated[arm]=dict(games=games,first=first,panel=summarize(games,first),training=model["training"],**run)
    baseline=evaluated["direct"]["games"][::2]
    require(all(arm["games"][::2]==baseline for arm in evaluated.values()),"frozen parent paths changed by training style")
    outcomes={a:r["games"][1::2] for a,r in evaluated.items()}
    outcomes["parent"]=baseline
    compare={}
    for a,b in (("parent","direct"),("direct","old"),("direct","new"),("old","new")):
        pairs=list(zip(outcomes[a],outcomes[b],strict=True))
        compare[b+"_vs_"+a]=dict(layouts=by_layout(pairs),
            desks={t:contrast([difference(x,y) for x,y in pairs if x["table"]==t]) for t in range(4)})
    return dict(arms=evaluated,contrasts=compare,games=2*len(grid)*len(ARMS),
                layouts=len({s for s,*_ in grid}))


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-teacher-increments"/os.environ["JUNQI_SERIAL_KEY"])
    folder=scope.folder
    if not audit:
        prepare(scope)
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"teacher comparison inputs changed")
    parent=read(folder/"generation1.json")
    matched,teaching=teach(scope,parent,int(audit))
    if not audit:
        learn(folder,parent,matched)
    require(all(sha(folder/n)==v for n,v in read(folder/"trained.json").items()),"trained comparison weights changed")
    stage="confirmation" if audit else "development"
    report=evaluate(scope,parent,[tuple(r) for r in read(folder/"grid.json")[stage]])
    write_receipt(folder/(stage+".json"),report)
    if audit:
        new=report["contrasts"]["new_vs_direct"]["layouts"]["interval"]
        against_old=report["contrasts"]["new_vs_old"]["layouts"]["interval"]
        verdict=("supported" if new[0]>0 and against_old[0]>0 else
                 "refuted" if new[1]<0 or against_old[1]<0 else "inconclusive")
        print("JUNQI_RESEARCH "+verdict)
    print(json.dumps(dict(teacher=teaching,contrasts=report["contrasts"],layouts=report["layouts"],games=report["games"])))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected teacher comparison stage")
    main(bool(sys.argv[1:]))
