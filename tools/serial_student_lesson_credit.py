"""Hold the update fixed while masking only newly admitted lesson targets."""
import json
from math import isfinite
import os
import sys

from serial_dynamic_reply import ROOT, PRIOR, build, native, require
from serial_formation_matrix import contrast, score_difference
from serial_full_student_evolution import games
from serial_local_lesson_cycle import train
from serial_reply_window import examples, judge, model_wire, split
from serial_visited_reply_value import choose
from serial_worker import read, write_receipt

PREVIOUS=ROOT/"build/local-lesson-cycle/screen-and-replay-local-full-mode-lessons-6b68ce7fd1d3722196ad"
START=2840000


def measure_root_scores(folder,model,screen):
    items=[]
    for key,d in screen["decisions"].items():
        if d["admitted"]:
            r=screen["review"][key]
            items.extend((key,action,r["x"][d["base"]],v) for action,v in r["x"].items())
    payload=model_wire(model,"window")+"\n"+str(len(items))+"\n"
    payload+="\n".join(" ".join(map(str,v)) for _,_,a,b in items for v in (a,b))
    rows=native(folder,"reply-window-judge-window",payload).splitlines()
    require(len(rows)==len(items),"incomplete native lesson score trace")
    audit={}
    for (key,action,_,_),row in zip(items,rows,strict=True):
        tag,picked,base,score=row.split()
        require(tag=="J" and picked in ("0","1") and isfinite(float(score)),"invalid output from restored scorer")
        audit.setdefault(key,{})[action]={"score":float(score),"gap":float(score)-float(base)}
    result={}
    for key,scores in audit.items():
        d=screen["decisions"][key]
        actions=list(scores)
        high=choose(actions,scores)
        result[key]=dict(actions=scores,chosen=high,base=d["base"],lesson=d["candidate"],
                         lesson_gap=scores[d["candidate"]]["gap"])
    return result


def compare_panels(a,b):
    rows={(r["seed"],r["table"],r["arm"]):r for r in a["rows"]}
    paired={0:[],1:[],2:[]}
    for other in b["rows"]:
        r=rows[other["seed"],other["table"],other["arm"]]
        if r["arm"] in (0,1,3):
            require(r["final_hash"]==other["final_hash"],"independent ablation baseline drift")
        if r["arm"]==4:
            paired[r["table"]].append(score_difference(other["result"],r["result"]) if
                                      r["final_hash"]!=other["final_hash"] else (0.,0.))
    return {str(t):contrast(pairs) for t,pairs in paired.items()}


def main(audit):
    folder=ROOT/"build/student-lesson-credit"/os.environ["JUNQI_SERIAL_KEY"]
    frozen=read(PRIOR/"students.json")
    parent=frozen["models"]["window"]
    learned=read(PREVIOUS/"student.json")
    screen=read(PREVIOUS/"screen.json")
    artifact=folder/"masked.json"
    if not audit:
        folder.mkdir(parents=True,exist_ok=True)
        build(folder)
        cases,_,source=examples(folder)
        old,test=split(cases)
        replay,rows=train(parent,old,screen["review"],screen["decisions"])
        require(replay==learned["models"]["window"] and rows==screen["stats"]["action_rows"],
                "cannot exactly reproduce the frozen training update")
        control,count=train(parent,old,screen["review"],screen["decisions"],include_new=False)
        require(rows==count and control!=parent and control!=replay,"mask-only ablation failed")
        masked=dict(models={"board":frozen["models"]["board"],"window":control})
        write_receipt(artifact,masked)
        models={"parent":frozen,"masked":read(artifact),"learned":learned}
        diagnostic={k:dict(lessons=measure_root_scores(folder,m["models"]["window"],screen),
                           old_course=judge(folder,m,source,test)["errors"]["window"]["counts"]["correct"])
                    for k,m in models.items()}
        write_receipt(folder/"diagnostic.json",diagnostic)
        print(json.dumps({k:{p:v["lesson_gap"] for p,v in d["lessons"].items()} for k,d in diagnostic.items()}))
    else:
        masked=read(artifact)
        diagnosis=read(folder/"diagnostic.json")
        with_new=games(folder,parent,learned["models"]["window"],START)
        without_new=games(folder,parent,masked["models"]["window"],START)
        control=compare_panels(with_new,without_new)
        result=dict(with_new=with_new,without_new=without_new,new_minus_masked=control,diagnostic=diagnosis)
        write_receipt(folder/"confirmation.json",result)
        print(json.dumps(dict(compared_to_masked=control,
                              with_new=with_new["differences"],without_new=without_new["differences"])))
        print("JUNQI_RESEARCH inconclusive")
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
