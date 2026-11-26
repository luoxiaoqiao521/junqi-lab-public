"""Independently judge all submitted moves and distill only confirmed preferences."""
from collections import Counter, defaultdict
from itertools import product
import json
import os
import shutil
from statistics import mean
import sys

from full_mode_runtime import FullModeRun, isolated, sha
from serial_dark_ordinary_learning import RATE, RULES, UPDATES, assert_layout_separation, model_wire, run_native, summarize
from serial_dark_replanning import INPUTS, SOURCE, model_payload, parse, serialize, summarize as summarize_roots
from serial_dynamic_reply import ROOT, build, require
from serial_preference_retention import np, train
from serial_preference_step_match import parameter_distance, schedule
from serial_worker import read, write_receipt


MIN_MARGIN=.125


def prepare(scope):
    folder=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    for name in ("original.json","mixed.json","peer.json","experience.json"):
        shutil.copyfile(SOURCE/name,folder/name)
    grid={phase:[(start+i,i%4,4*(i//4%2),(20,28,44,60)[(i//2)%4],-1) for i in range(count)]
          for phase,start,count in (("development",24141000,32),("confirmation",24142000,36))}
    grid["whole_games"]=[(24143000+i,i%4,t) for i,t in product(range(36),(0,4))]
    assert_layout_separation(grid)
    write_receipt(folder/"grid.json",grid)
    write_receipt(folder/"fingerprints.json",{n:sha(folder/n) for n in (*INPUTS,"experience.json")})


def complete_values(root,features,outcomes):
    choices=set(features)
    require(len(choices)==root["wide"] and set(root["moves"])<=choices,"incomplete audited candidate set")
    require(outcomes.keys()==set(product(choices,(2,3),(0,4),range(2))),"incomplete independent action values")
    if any(v<0 for v in outcomes.values()):
        return None
    return {a:[mean(outcomes[a,w,t,i]/2 for i in range(2)) for w,t in product((2,3),(0,4))] for a in choices}


def qualify(root,features,outcomes):
    values=complete_values(root,features,outcomes)
    if values is None:
        return "truncated",None
    old,_,selected,_=root["moves"]
    if old==selected:
        return "kept",None
    delta=[a-b for a,b in zip(values[selected],values[old],strict=True)]
    maximum=max(mean(v) for v in values.values())
    if any(d<0 for d in delta) or mean(delta)<MIN_MARGIN:
        return "reference_disagrees",None
    if mean(values[selected])<maximum:
        return "third_move_better",None
    # Only preferences confirmed within the audited portfolio may enter this course.
    pairs=[]
    for move,value in values.items():
        edge=[a-b for a,b in zip(values[selected],value,strict=True)]
        if min(edge)>=0 and mean(edge)>=MIN_MARGIN:
            pairs.append(dict(worse=move,better=selected,delta=edge))
    require(pairs,"confirmed teacher root did not include its reference pair")
    return "qualified",dict(features=features,pairs=pairs,selected=selected,original=old)


def audit_portfolios(scope,rows,models):
    replicas=2
    wire=model_payload(models)+[" ".join(map(str,(*r,replicas))) for r in rows]
    with isolated(scope,ROOT/"build"):
        raw=scope.run("dark-replanning-course","\n".join(wire)+"\n")
    ordinary,features,judgements=[],defaultdict(dict),defaultdict(dict)
    for line in raw.splitlines():
        tag,*fields=line.split()
        if tag=="K":
            continue
        if tag not in ("Q","V"):
            ordinary.append(line)
            continue
        key=tuple(map(int,fields[:3]))
        move=int(fields[3])
        if tag=="Q":
            x=list(map(float,fields[4:]))
            require(len(x)==44 and np.isfinite(x).all() and move not in features[key],"invalid course feature")
            features[key][move]=x
        else:
            world,desk,rep,result,steps=map(int,fields[4:])
            cell=(move,world,desk,rep)
            require(steps>0 and result in (-1,0,1,2) and cell not in judgements[key],"invalid course judge")
            judgements[key][cell]=result
    roots=parse("\n".join(ordinary),rows,replicas)
    require(features.keys()==judgements.keys()=={k for k,r in roots.items() if r.get("worlds")==4},
            "course and verified roots differ")
    counts=Counter()
    course=[]
    for key,root in roots.items():
        tag,pair=qualify(root,features[key],judgements[key]) if key in features else ("no_coverage",None)
        counts[tag]+=1
        if pair:
            course.append(dict(seed=key[0],seat=key[1],table=key[2],**pair))
    summary=summarize_roots(roots)
    for name in ("original_boards","new_worlds"):
        summary[name].pop("interaction",None)
    summary.update(decisions=counts,selected_preferences=sum(len(r["pairs"]) for r in course),
                   independent_runs=sum(len(v) for v in judgements.values()))
    audit=serialize(roots,summary)
    audit.update(course=course,judgements=[(*k,[(*a,v) for a,v in panel.items()]) for k,panel in judgements.items()])
    return course,audit


def training_arrays(ordinary,course):
    raw=[r["features"] for r in ordinary]
    pairs=[]
    for root in course:
        index={}
        for move,x in root["features"].items():
            index[move]=len(raw)
            raw.append(x)
        for row in root["pairs"]:
            pairs.append((index[row["worse"]],index[row["better"]],.65))
    y=np.full((len(raw),3),np.nan)
    y[:len(ordinary)]=np.eye(3)[[r["outcome"] for r in ordinary]]
    return np.asarray(raw),y,pairs


def learn(folder,parent,course):
    data=read(folder/"experience.json")["examples"][0]
    raw,y,pairs=training_arrays(data,course)
    source=dict(mean=parent["mean"],scale=parent["scale"],parameters=parent["models"]["collision"])
    steps=None
    metadata={}
    for name in ("direct","guided"):
        active=pairs if name=="guided" else []
        trace={}
        learned=train(source,(raw,y,active,[]),bool(active),steps=UPDATES,rate=RATE,log_odds=True,
                      trace=trace,step_norms=steps)
        if steps is None:
            steps=schedule(trace)
        require(np.allclose(steps,schedule(trace),rtol=1e-10,atol=1e-12),"course update effort differs")
        consumed=sum(r["gradient_norm"]>0 for step in trace["trajectory"] for r in step["pairs"])
        distance=parameter_distance(source,learned)
        require(distance>0 and (not active or consumed>0),"no parameter or preference update")
        metadata[name]=dict(ordinary=len(data),pairs=len(active),applications=consumed,updates=UPDATES,step_budget=sum(steps),distance=distance)
        model=dict(parent,models={"collision":learned["parameters"]},training=metadata[name],
                   teacher_course=sha(folder/"development.json"))
        require(model_wire(model)!=model_wire(parent),"no course weight change")
        write_receipt(folder/(name+".json"),model)
    write_receipt(folder/"trained.json",{n:sha(folder/n) for n in ("development.json","direct.json","guided.json")})
    return metadata


def heldout_order(model,course):
    weights=model["models"]["collision"]["weights"]
    bias=model["models"]["collision"]["bias"]
    stats=Counter()
    for root in course:
        moves=list(root["features"])
        x=np.asarray(list(root["features"].values()))
        x=np.clip((x-model["mean"])/model["scale"],-6,6)
        z=(np.tanh(x@np.asarray(weights[0])+bias[0])@np.asarray(weights[1])+bias[1])[:,:3]
        p=np.exp(z-z.max(axis=1,keepdims=True))
        scores=(p[:,2]+.5*p[:,1])/p.sum(axis=1)
        idx=moves.index(root["selected"])
        stats["roots"]+=1
        stats["above_original"]+=int(scores[idx]>scores[moves.index(root["original"])])
        stats["first_in_portfolio"]+=int(idx==np.argmax(scores))
    return dict(stats)


def confirm_students(scope,rows,models,course):
    folder=scope.folder
    if not (folder/"trained.json").is_file():
        require(not read(folder/"development.json")["course"],"qualified study did not update students")
        return {"trained":False}
    require(all(sha(folder/n)==v for n,v in read(folder/"trained.json").items()),"course weights changed")
    a,b=(read(folder/(n+".json")) for n in ("direct","guided"))
    rows,_,first,metrics=run_native(scope,"X",rows,(a,b),peer=models["peer"])
    panel=summarize(rows,first)
    write_receipt(folder/"student-games.json",dict(games=rows,first=first,panel=panel,**metrics))
    return dict(trained=True,panel=panel,direct=a["training"],guided=b["training"],
                heldout_order={n:heldout_order(m,course) for n,m in (("direct",a),("guided",b))})


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-replan-course"/os.environ["JUNQI_SERIAL_KEY"])
    if not audit:
        prepare(scope)
    folder=scope.folder
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"course inputs changed")
    stage="confirmation" if audit else "development"
    grid={phase:[tuple(r) for r in rows] for phase,rows in read(folder/"grid.json").items()}
    models={n:read(folder/(n+".json")) for n in ("original","mixed","peer")}
    course,judged=audit_portfolios(scope,grid[stage],models)
    write_receipt(folder/(stage+".json"),judged)
    if not audit and course:
        print(json.dumps(dict(training=learn(folder,models["original"],course))))
    if audit:
        result=confirm_students(scope,grid["whole_games"],models,course)
        write_receipt(folder/"student-summary.json",result)
        print(json.dumps(result))
        bounds=result.get("panel",{}).get(4,{}).get("interval",(-1,1))
        verdict=("supported" if bounds[0]>0 else "refuted" if bounds[1]<0 else "inconclusive")
        print("JUNQI_RESEARCH "+verdict)
    print(json.dumps(judged["summary"]))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected course stage")
    main(bool(sys.argv[1:]))
