"""Train equal-budget students from terminal samples and masked action preferences."""
from collections import Counter, defaultdict
from itertools import combinations
import json
import os
import shutil
from statistics import mean
import sys

from full_mode_runtime import FullModeRun, isolated, sha
from serial_dynamic_reply import ROOT,require
from serial_formation_matrix import contrast,score_difference
from serial_full_scoring_audit import collect,verified_scope
from serial_full_student_evolution import games
from serial_preference_retention import np,train
from serial_preference_step_match import schedule
from serial_reply_window import examples,model_wire,split
from serial_worker import read,write_receipt

SCORING=ROOT/"build/full-scoring-audit/confirm-actual-scoring-splits-and-reference-search-bbde4110d57d3af71fdb"
LEAF=ROOT/"build/full-teacher-terms/verify-six-ply-teacher-terms-and-active-responses-c0ae7320b424e3478478"
ARMS=("terminal","one_pair","many_pairs","mixed")
EVAL_SEED=2864100


def samples(receipt):
    data=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    for seed,action,desk,stream,score in receipt["outcomes"]:
        require(score in (0,1,2),"incomplete result cannot supply a target")
        data[seed][action][desk].append(score)
    require(all(sorted(row)==[2,3,4] and all(len(v)==6 for v in row.values())
                for actions in data.values() for row in actions.values()),"incomplete measured response grid")
    return data


def directional(actions):
    qualified=[]
    for a,b in combinations(sorted(actions),2):
        diffs=[mean(actions[b][t])-mean(actions[a][t]) for t in (2,3,4)]
        if max(diffs)<0:
            a,b=b,a
            diffs=[-v for v in diffs]
        if min(diffs)>0:
            qualified.append((a,b,mean(diffs)/2))
    return sorted(qualified,key=lambda row:(-row[2],row[0],row[1]))


def course(roots,outcomes,anchors,arm):
    raw,labels,pairs=[],[],[]
    for seed,actions in outcomes.items():
        start=len(raw)
        order=sorted(actions)
        x=[roots[seed]["actions"][a]["x"] for a in order]
        known=[]
        for a in order:
            counts=Counter(actions[a][2])
            known.append([counts[k]/6 for k in (0,1,2)])
        raw.extend(x)
        labels.extend(known if arm in ("terminal","mixed") else [[np.nan]*3]*len(x))
        if arm=="terminal":
            continue
        available=directional(actions)
        if arm=="mixed" and available:
            start=len(raw)
            raw.extend(x)
            labels.extend([[np.nan]*3]*len(x))
        if arm=="one_pair":
            available=available[:1]*len(available)
        pairs.extend((start+order.index(a),start+order.index(b),.5+.45*gap) for a,b,gap in available)
    for record in anchors:
        raw.extend(record["features"])
        labels.extend(np.eye(3)[record["scores"]])
    require(arm=="terminal" or pairs,"teacher has no stable preference to consume")
    return np.asarray(raw),np.asarray(labels),pairs,[]


def prepare(scope):
    scope.engine.mkdir(parents=True,exist_ok=True)
    files=read(LEAF/"source.json")
    for name,expected in files.items():
        path=LEAF/name
        require(sha(path)==expected,"lesson source changed")
        shutil.copyfile(path,scope.folder/name)
    write_receipt(scope.folder/"source.json",files)
    old=read(SCORING/"development.json")
    c=read(LEAF/"development.json")
    outcomes=samples(c)
    roots={int(seed):dict(row,actions={int(a):x for a,x in row["actions"].items()})
           for seed,row in old["roots"].items()}
    require(set(roots)==set(outcomes),"lesson decisions and terminal outcomes disagree")
    parent=read(scope.folder/"reset.json")
    with isolated(scope,ROOT/"build"):
        cases,_,_=examples(scope.engine)
    anchors,_=split(cases)
    models={}
    stats={}
    norms=None
    for arm in ARMS:
        curriculum=course(roots,outcomes,anchors,arm)
        trace={}
        model=train(parent,curriculum,arm!="terminal",steps=40,rate=.025,trace=trace,
                    step_norms=norms,log_odds=arm!="terminal")
        if norms is None:
            norms=schedule(trace)
        require(np.allclose(schedule(trace),norms,rtol=1e-12,atol=1e-15) and model!=parent,
                "training did not match steps or update the model")
        models[arm]=model
        stats[arm]=dict(rows=len(curriculum[0]),ordered_pairs=len(curriculum[2]),steps=schedule(trace))
    write_receipt(scope.folder/"students.json",models)
    write_receipt(scope.folder/"course.json",dict(stats=stats,
        sources=dict(candidate_features=sha(SCORING/"development.json"),training_returns=sha(LEAF/"development.json"),
                     holdout_returns=sha(LEAF/"confirmation.json")),model_sha=sha(scope.folder/"students.json"),
        compared_actions=sum(map(len,outcomes.values())),train_roots=sorted(outcomes)))


def preference_panel(scope,models):
    manifest=read(LEAF/"manifest.json")
    parent=read(scope.folder/"reset.json")
    _,wire=verified_scope(scope.folder)
    roots=collect(scope,wire,manifest)
    known=samples(read(LEAF/"confirmation.json"))
    require(set(roots)==set(known),"different source in the frozen held-out candidate evidence")
    holdout=[]
    for seed,actions in known.items():
        for a,b,_ in directional(actions):
            rows=roots[seed]["actions"]
            holdout.append((rows[a]["x"],rows[b]["x"]))
    summary={}
    for name,model in {"parent":parent,**models}.items():
        tape=model_wire(model,"window")+"\n"+str(len(holdout))+"\n"
        tape+="\n".join(" ".join(map(str,x)) for a,b in holdout for x in (a,b))
        with isolated(scope,ROOT/"build"):
            encoded=scope.run("reply-window-judge-window",tape)
        scores=[(tag,int(chosen),float(lo),float(hi)) for tag,chosen,lo,hi in map(str.split,encoded.splitlines())]
        require(len(scores)==len(holdout) and all(r[0]=="J" for r in scores),"native scorer lost a candidate comparison")
        summary[name]=dict(right=sum(r[1] for r in scores),total=len(scores),scores=scores)
    return summary


def full_panel(scope,models):
    parent=read(scope.folder/"reset.json")
    all_games={}
    aggregate={}
    for name,model in models.items():
        with isolated(scope,ROOT/"build"):
            data=games(scope.engine,parent,model,EVAL_SEED)
        all_games[name]=data
        diffs=defaultdict(list)
        frozen={(r["seed"],r["table"]):r for r in data["rows"] if r["arm"]==3}
        for row in data["rows"]:
            if row["arm"]!=4:
                continue
            base=frozen[row["seed"],row["table"]]
            delta=(0,0) if row["final_hash"]==base["final_hash"] else score_difference(base["result"],row["result"])
            diffs[row["seed"]].append(delta)
        aggregate[name]=contrast([tuple(map(mean,zip(*row,strict=True))) for row in diffs.values()])
    controls=[[(r["seed"],r["table"],r["arm"],r["final_hash"]) for r in panel["rows"] if r["arm"] in (0,1,3)]
              for panel in all_games.values()]
    require(all(r==controls[0] for r in controls),"opening control diverged across objectives")
    return dict(panels=all_games,layout_clustered=aggregate,eval_layouts=16)


def main(audit):
    scope=FullModeRun(ROOT/"build/full-mode-return-mix"/os.environ["JUNQI_SERIAL_KEY"])
    if not audit:
        prepare(scope)
        course_inputs=read(scope.folder/"course.json")
        print(json.dumps({k:v for k,v in course_inputs.items() if k!="stats"}))
    else:
        course_inputs=read(scope.folder/"course.json")
        path=scope.folder/"students.json"
        require(sha(path)==course_inputs["model_sha"],"student not loaded from frozen course")
        require(set(course_inputs["train_roots"]).isdisjoint(range(EVAL_SEED,EVAL_SEED+16)) and
                set(course_inputs["train_roots"]).isdisjoint(row["seed"] for row in read(LEAF/"manifest.json")),
                "lesson layout reused in confirmation")
        models=read(path)
        results=dict(local=preference_panel(scope,models),games=full_panel(scope,models))
        write_receipt(scope.folder/"confirmation.json",results)
        signature={name:row["right"] for name,row in results["local"].items()}
        metrics=results["games"]["layout_clustered"]
        supported=[name for name,v in metrics.items() if v["interval"][0]>0 and signature[name]>signature["parent"]]
        print(json.dumps(dict(local=signature,whole_games=metrics)))
        print("JUNQI_RESEARCH "+("supported" if supported else "inconclusive"))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
