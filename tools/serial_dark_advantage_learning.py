"""Compare ordinary returns, ranking and relative team value on the same supported edges."""
from collections import Counter
from itertools import product
import json
import os
import shutil
from statistics import mean
import sys

from full_mode_runtime import FullModeRun, sha
from serial_dark_ordinary_learning import RATE, UPDATES, assert_layout_separation, run_native, summarize
from serial_dark_replanning import INPUTS, SOURCE
from serial_dark_seat_candidates import run
from serial_dynamic_reply import ROOT, require
from serial_formation_matrix import contrast, score_difference
from serial_preference_retention import np, train
from serial_preference_step_match import parameter_distance, schedule
from serial_worker import read, write_receipt


SEAT=ROOT/"build/dark-seat-candidates/audit-four-dark-seat-aware-candidates-59e0f5abf22dab3b8a44"
PRIOR=ROOT/"build/dark-replan-course/qualify-four-dark-post-reply-training-e857dd6de2b75528f9af/development.json"
NAMES=("ordinary","ranking","advantage")
FILES=(*INPUTS,"experience.json","old-course.json","seat-course.json")


def prepare(scope):
    folder=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    for n in ("original.json","mixed.json","peer.json","engine/deadline_opening.exe","engine/rules.json"):
        shutil.copyfile(SEAT/n,folder/n)
    shutil.copyfile(SOURCE/"experience.json",folder/"experience.json")
    shutil.copyfile(PRIOR,folder/"old-course.json")
    shutil.copyfile(SEAT/"development.json",folder/"seat-course.json")
    grid=dict(spots=[(24148000+i,i%4,4*(i//4%2),(20,32,52,68)[(i//2)%4]) for i in range(24)],
              games=[(24149000+i,i%4,t) for i,t in product(range(48),(0,4))])
    groups=dict(grid,ordinary=[(r["seed"],) for r in read(folder/"experience.json")["examples"][0]],
                earlier=[(r["seed"],) for r in read(folder/"old-course.json")["course"]],
                seat=[tuple(r[:3]) for r in read(folder/"seat-course.json")["roots"]])
    assert_layout_separation(groups)
    write_receipt(folder/"grid.json",grid)
    write_receipt(folder/"fingerprints.json",{n:sha(folder/n) for n in FILES})


def values(rows):
    data={(a,w,t,i):result for a,w,t,i,result,_ in rows}
    moves={k[0] for k in data}
    if any(result<0 for result in data.values()):
        return {}
    return {a:[mean(data[a,w,t,i]/2 for i in (60,61,62)) for w,t in product((2,3),(0,5))] for a in moves}


def course_from_seat(receipt):
    counts=Counter()
    output=[]
    for seed,seat,table,root in receipt["roots"]:
        if root["worlds"]!=4:
            counts["uncovered"]+=1
            continue
        ref=values(root["R"])
        if not ref:
            counts["unresolved"]+=1
            continue
        first={a:mean(result/2 for m,_,_,_,result,_ in root["L"] if m==a)
               for a in ref if all(result>=0 for m,_,_,_,result,_ in root["L"] if m==a)}
        if not first:
            counts["unresolved"]+=1
            continue
        winner=max(sorted(first),key=first.get)
        if mean(ref[winner])<max(map(mean,ref.values())):
            counts["third_better"]+=1
            continue
        features={int(k):x for k,x in root["features"].items()}
        pairs=[]
        for a in first:
            margin=[x-y for x,y in zip(ref[winner],ref[a],strict=True)]
            if first[winner]<=first[a] or min(margin)<-1e-8 or mean(margin)<.125:
                continue
            if features[a]==features[winner]:
                counts["identical_features"]+=1
                continue
            pairs.append(dict(worse=a,better=winner,delta=margin))
        if pairs:
            output.append(dict(seed=seed,table=table,seat=seat,features=features,
                               pairs=sorted(pairs,key=lambda x:(-mean(x["delta"]),x["worse"]))[:3]))
            counts["qualified"]+=1
        else:
            counts["no_stable_edge"]+=1
    return output,counts


def training_arrays(examples,course):
    observations=[r["features"] for r in examples]
    pairs=[]
    for root in course:
        index={}
        for k,row in root["features"].items():
            index[int(k)]=len(observations)
            observations.append(row)
        for pair in root["pairs"]:
            pairs.append((index[pair["worse"]],index[pair["better"]],mean(pair["delta"])))
    labels=np.full((len(observations),3),np.nan)
    labels[:len(examples)]=np.eye(3)[[r["outcome"] for r in examples]]
    return np.asarray(observations),labels,pairs


def learn(folder,parent,course):
    direct=read(folder/"experience.json")["examples"][0]
    raw,y,edge=training_arrays(direct,course)
    require(0<len(edge) and all(.125<=target<=1 for a,b,target in edge),"unsupported advantage targets")
    source=dict(mean=parent["mean"],scale=parent["scale"],parameters=parent["models"]["collision"])
    common=None
    meta={}
    for name in NAMES:
        labels=[] if name=="ordinary" else [(a,b,v if name=="advantage" else .65) for a,b,v in edge]
        trace={}
        child=train(source,(raw,y,labels,[]),bool(labels),steps=UPDATES,rate=RATE,
                    trace=trace,step_norms=common,advantage=name=="advantage")
        if common is None:
            common=schedule(trace)
        require(np.allclose(common,schedule(trace),rtol=1e-10,atol=1e-12),"step budgets differed")
        distance=parameter_distance(source,child)
        applied=sum(p["gradient_norm"]>0 for step in trace["trajectory"] for p in step["pairs"])
        require(distance>0 and (not labels or applied>0),"teacher loss was not consumed")
        meta[name]=dict(ordinary=len(direct),pairs=len(labels),updates=UPDATES,step_budget=sum(common),
                        distance=distance,applied=applied)
        model=dict(parent,models={"collision":child["parameters"]},training=meta[name],
                   teacher_course=sha(folder/"course.json"))
        write_receipt(folder/(name+".json"),model)
    write_receipt(folder/"trained.json",{n:sha(folder/n) for n in ("course.json",*(a+".json" for a in NAMES))})
    return meta


def portfolio(model,course):
    w,v=map(np.asarray,model["models"]["collision"]["weights"])
    b,c=map(np.asarray,model["models"]["collision"]["bias"])
    hits=Counter()
    errors=[]
    for root in course:
        moves=list(root["features"])
        x=np.asarray(list(root["features"].values()))
        x=np.clip((x-model["mean"])/model["scale"],-6,6)
        z=(np.tanh(x@w+b)@v+c)[:,:3]
        p=np.exp(z-z.max(axis=1,keepdims=True))
        q=(p[:,2]+.5*p[:,1])/p.sum(axis=1)
        hits["top"]+=moves[int(np.argmax(q))]==root["pairs"][0]["better"]
        for pair in root["pairs"]:
            gap=q[moves.index(pair["better"])]-q[moves.index(pair["worse"])]
            hits["pairs"]+=1
            hits["direction"]+=gap>0
            errors.append(abs(float(gap)-mean(pair["delta"])))
    return dict(roots=len(course),correct={k:int(v) for k,v in hits.items()},mean_absolute_error=mean(errors) if errors else None)


def whole(scope,grid,models,peer):
    answers={}
    rows={}
    for n in NAMES[1:]:
        games,_,first,info=run_native(scope,"X",grid,(models["ordinary"],models[n]),peer=peer)
        answers[n]=dict(games=games,first=first,panel=summarize(games,first),**info)
        rows[n]=games
    a,b=(rows[n] for n in NAMES[1:])
    require(a[::2]==b[::2],"teacher-free baseline game changed")
    both={}
    for t in (0,4):
        both[t]=contrast([(0,0) if x["hash"]==y["hash"] else score_difference(x["result"],y["result"])
                          for x,y in zip(a[1::2],b[1::2],strict=True) if x["table"]==t])
    write_receipt(scope.folder/"student-games.json",answers)
    return dict(panels={n:v["panel"] for n,v in answers.items()},advantage_vs_ranking=both)


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-advantage-learning"/os.environ["JUNQI_SERIAL_KEY"])
    folder=scope.folder
    if not audit:
        prepare(scope)
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"advantage inputs changed")
    base={n:read(folder/(n+".json")) for n in ("original","peer")}
    if not audit:
        prior=read(folder/"old-course.json")["course"]
        fresh,counts=course_from_seat(read(folder/"seat-course.json"))
        course=prior+fresh
        write_receipt(folder/"course.json",dict(roots=course,old=len(prior),new=len(fresh),filters=counts))
        report=dict(roots=len(course),edges=sum(len(r["pairs"]) for r in course),filters=counts,
                    fitted=learn(folder,base["original"],course))
        write_receipt(folder/"fit.json",report)
    else:
        require(all(sha(folder/n)==v for n,v in read(folder/"trained.json").items()),"advantage model changed")
        grids={k:[tuple(row) for row in v] for k,v in read(folder/"grid.json").items()}
        audit=run(scope,grids["spots"],base)
        course,counts=course_from_seat(audit)
        write_receipt(folder/"holdout.json",audit)
        models={n:read(folder/(n+".json")) for n in NAMES}
        report=whole(scope,grids["games"],models,base["peer"])
        report.update(spot_filters=counts,spot={n:portfolio(m,course) for n,m in models.items()})
        write_receipt(folder/"confirmation.json",report)
        margin=report["advantage_vs_ranking"][4]["interval"]
        trial=report["panels"]["advantage"][4]["interval"]
        verdict=("supported" if min(margin[0],trial[0])>0 else "refuted" if max(margin[1],trial[1])<0 else "inconclusive")
        print("JUNQI_RESEARCH "+verdict)
    print(json.dumps(report))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected advantage stage")
    main(bool(sys.argv[1:]))
