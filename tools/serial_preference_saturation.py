"""Exercise the 52-feature scorer with a probability versus log-odds preference loss."""
import json
import os
import shutil
import sys

from serial_dynamic_reply import ROOT, PRIOR, require
from serial_preference_retention import np, team_log_odds, train, training_rows
from serial_reply_window import examples, judge, split
from serial_student_lesson_credit import PREVIOUS, measure_root_scores
from serial_worker import read, write_receipt

PRODUCT=ROOT/"build/product-shadow/validate-product-shadow-after-startup-fix-13d3781f6df5b8bf5744"


def project(model,features):
    x=np.clip((np.asarray(features)-model["mean"])/model["scale"],-6,6)
    w,v=map(np.asarray,model["parameters"]["weights"])
    b,c=map(np.asarray,model["parameters"]["bias"])
    logits=(np.tanh(x@w+b)@v+c)[:,:3]
    odds,_=team_log_odds(logits)
    return np.exp(-np.logaddexp(0.,-odds))


def numeric_controls():
    logits=np.array([[-7.,-112.,-103.],[4.,-8.,7.],[1.,4.,-5.]])
    values,slope=team_log_odds(logits)
    require(np.isfinite(values).all() and np.isfinite(slope).all(),"nonfinite saturated log-odds")
    for k in range(3):
        step=np.eye(3)[k]*1e-5
        numerical=(team_log_odds(logits+step)[0]-team_log_odds(logits-step)[0])/(2e-5)
        require(np.allclose(numerical,slope[:,k],atol=2e-9),"log-odds derivative differs")
    require(np.allclose(team_log_odds(logits+50.)[0],values),"common logit shift changed value")


def check_native(folder,model,screen):
    scores=measure_root_scores(folder,model,screen)
    for key,row in scores.items():
        actions=list(row["actions"])
        expected=project(model,[screen["review"][key]["x"][action] for action in actions])
        loaded=[row["actions"][action]["score"] for action in actions]
        require(np.allclose(expected,loaded,atol=2e-10,rtol=7e-6),"Python/C++ WDL payoff mismatch")
    return scores


def main(audit):
    folder=ROOT/"build/preference-saturation"/os.environ["JUNQI_SERIAL_KEY"]
    frozen=read(PRIOR/"students.json")["models"]
    parent=frozen["window"]
    screen=read(PREVIOUS/"screen.json")
    numeric_controls()
    if not audit:
        folder.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(PRODUCT/"shadow-engine/deadline_opening.exe",folder/"deadline_opening.exe")
    cases,_,source=examples(folder)
    old,test=split(cases)
    if not audit:
        raw,y,pairs,retain=training_rows(screen,old)
        indices=[i for a,b,_ in pairs for i in (a,b)]
        courses=dict(isolated=(raw[indices],y[indices],[(i,i+1,.99) for i in range(0,len(indices),2)],[]),
                     mixed=(raw,y,[(a,b,.99) for a,b,_ in pairs],retain))
        models=dict(parent=parent)
        details={}
        for label,course in courses.items():
            for transform in ("probability","log_odds"):
                name=f"{label}-{transform}"
                diagnostics={}
                models[name]=train(parent,course,True,steps=320,rate=.12,decay=0.,
                                   log_odds=transform=="log_odds",trace=diagnostics)
                details[name]=diagnostics
        write_receipt(folder/"training-trace.json",details)
        write_receipt(folder/"models.json",models)
    models=read(folder/"models.json")
    results={}
    for name,model in models.items():
        rows=check_native(folder,model,screen)
        results[name]=dict(positive=sum(r["lesson_gap"]>0 for r in rows.values()),
                           chosen=sum(r["chosen"]==r["lesson"] for r in rows.values()),
                           margins=[r["lesson_gap"] for r in rows.values()],actions=rows,
                           retained=judge(folder,dict(models=dict(frozen,window=model)),source,test)["errors"]["window"]["counts"]["correct"])
    write_receipt(folder/("restored.json" if audit else "development.json"),results)
    if audit:
        require(results==read(folder/"development.json"),"saved weights changed native scoring")
        count=results["isolated-log_odds"]["chosen"]-results["isolated-probability"]["chosen"]
        print("JUNQI_RESEARCH "+("supported" if count>0 else "refuted"))
    print(json.dumps({k:{x:v for x,v in row.items() if x!="actions"} for k,row in results.items()}))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
