"""Compare two preference objectives under the same observed parameter-step budgets."""
import json
import os
import shutil
import sys

from full_mode_runtime import sha
from serial_dynamic_reply import ROOT, PRIOR, require
from serial_preference_retention import np, train, training_rows
from serial_preference_saturation import check_native, numeric_controls
from serial_reply_window import examples, judge, split
from serial_student_lesson_credit import PREVIOUS
from serial_worker import read, write_receipt

SCORING=ROOT/"build/full-scoring-audit/confirm-actual-scoring-splits-and-reference-search-bbde4110d57d3af71fdb"


def parameter_distance(a,b):
    difference=[np.asarray(v)-np.asarray(w) for key in ("weights","bias")
                for v,w in zip(a["parameters"][key],b["parameters"][key],strict=True)]
    return float(np.sqrt(sum(np.sum(x*x) for x in difference)))


def schedule(trace):
    return [step["parameter_step_l2"] for step in trace["trajectory"]]


def train_four(parent,course):
    models={}
    traces={}
    arms=(("probability",False,None),("logit",True,None),
          ("probability_large",False,"logit"),("logit_small",True,"probability"))
    for name,transformed,reference in arms:
        trace={}
        norms=schedule(traces[reference]) if reference else None
        models[name]=train(parent,course,True,steps=320,rate=.12,decay=0.,
                           log_odds=transformed,trace=trace,step_norms=norms)
        if norms is not None:
            require(np.allclose(schedule(trace),norms,rtol=1e-12,atol=1e-15),"update amplitude control failed")
        traces[name]=trace
    return models,traces


def train_courses(parent,screen,anchors):
    raw,y,pairs,retain=training_rows(screen,anchors)
    indices=[i for a,b,_ in pairs for i in (a,b)]
    courses=dict(isolated=(raw[indices],y[indices],[(i,i+1,.99) for i in range(0,len(indices),2)],[]),
                 mixed=(raw,y,[(a,b,.99) for a,b,_ in pairs],retain))
    models,traces={},{}
    for course,rows in courses.items():
        models[course],traces[course]=train_four(parent,rows)
    return models,traces


def evaluate(folder,parent,frozen,models,screen,source,test,traces):
    result={}
    for course,arms in models.items():
        result[course]={}
        for name,model in arms.items():
            scores=check_native(folder,model,screen)
            path=schedule(traces[course][name])
            retained=judge(folder,dict(models=dict(frozen,window=model)),source,test)
            result[course][name]=dict(positive=sum(r["lesson_gap"]>0 for r in scores.values()),
                chosen=sum(r["chosen"]==r["lesson"] for r in scores.values()),
                margins=[r["lesson_gap"] for r in scores.values()],details=scores,
                first_step=path[0],sum_step_l2=sum(path),net_l2=parameter_distance(model,parent),
                retained=retained["errors"]["window"]["counts"]["correct"])
    return result


def main(audit):
    folder=ROOT/"build/preference-step-match"/os.environ["JUNQI_SERIAL_KEY"]
    frozen=read(PRIOR/"students.json")["models"]
    parent=frozen["window"]
    screen=read(PREVIOUS/"screen.json")
    binary=SCORING/"engine/deadline_opening.exe"
    require(sha(binary)==read(SCORING/"source.json")["engine/deadline_opening.exe"],"scoring executable changed")
    numeric_controls()
    if not audit:
        folder.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(binary,folder/"deadline_opening.exe")
    else:
        require(sha(folder/"deadline_opening.exe")==sha(binary),"confirmation binary changed")
    cases,_,source=examples(folder)
    old,test=split(cases)
    if not audit:
        models,traces=train_courses(parent,screen,old)
        write_receipt(folder/"models.json",models)
        write_receipt(folder/"trace.json",traces)
    result=evaluate(folder,parent,frozen,read(folder/"models.json"),screen,source,test,read(folder/"trace.json"))
    output=folder/("confirmation.json" if audit else "development.json")
    write_receipt(output,result)
    if audit:
        require(result==read(folder/"development.json"),"saved identical update changed output")
        differences=[(r["logit"]["chosen"]-r["probability_large"]["chosen"],
                      r["logit_small"]["chosen"]-r["probability"]["chosen"]) for r in result.values()]
        verdict="supported" if all(x>0 for row in differences for x in row) else \
                "refuted" if all(x<=0 for row in differences for x in row) else "inconclusive"
        print("JUNQI_RESEARCH "+verdict)
    print(json.dumps({k:{name:{f:v for f,v in data.items() if f!="details"} for name,data in arms.items()}
                      for k,arms in result.items()}))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
