"""Reopen archived lessons with frozen new teacher and preserve both label versions."""
from collections import Counter
import hashlib
from itertools import product
import json
import os
import sys

from serial_dynamic_reply import ROOT, PRIOR, build, frozen_wire, require
from serial_formation_matrix import score_difference
from serial_full_student_evolution import games
from serial_leaf_teacher import collect, value_rows
from serial_local_lesson_cycle import train
from serial_quiet_options import FROZEN
from serial_reply_window import examples, model_wire, split
from serial_student_lesson_credit import PREVIOUS, compare_panels
from serial_visited_reply_value import manifest
from serial_worker import read, write_receipt

VALUE=ROOT/"build/leaf-teacher/train-and-calibrate-full-mode-leaf-value-0e2026c31bf0100c16e4/value.json"
SCREEN=PREVIOUS/"screen.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rerun(folder,wire,screen,audit):
    sources=manifest(False,seed_offset=1800,archive_offset=6,include_witness=False)
    data=collect(folder,wire,False,sources=sources,recheck=audit)
    original=screen["discovery"]["roots"]
    require(data["roots"].keys()==original.keys() and data["spec"]==screen["discovery"]["spec"],
            "archive cohort or root manifest changed")
    for key,root in data["roots"].items():
        old=original[key]
        require(root["hash"]==old["hash"] and root["cut"]==old["ply"] and
                root["leaves"].keys()==old["scores"]["six"].keys(),"archived root or candidate changed")
        for a,leaves in root["leaves"].items():
            require(abs(sum(l["old"] for l in leaves.values())-old["scores"]["six"][a]["score"])<1e-10,
                    "old teacher identity failed under new interface")
    predictions,_=value_rows(data,read(VALUE))
    facts=dict(screen_sha=sha(SCREEN),value_sha=sha(VALUE),mode="all-identities-four-seat",
               root_source="previously-reviewed-development",review_seed_family=84569 if audit else 68569)
    return dict(facts=facts,data=data,predictions=predictions)


def effect(root,base,action):
    out=[]
    for table in range(4):
        points=[(0.,0.) if base==action else score_difference(root["review"][base][f"{table}/{i}"],
                         root["review"][action][f"{table}/{i}"]) for i in range(8)]
        out.append([sum(x)/8 for x in zip(*points,strict=True)])
    return out


def reliable(first,second):
    return all(min(low for low,_ in values)>=0 and sum(low>0 for low,_ in values)>=2
               for values in (first,second))


def scope(first,second,stress=()):
    if reliable(first,second) and all(low>=0 for low,_ in stress):
        return "corroborated-under-tested-policies"
    if (any(a[1]<0 and b[1]<0 for a,b in zip(first,second,strict=True)) or
            any(upper<0 for _,upper in stress)):
        return "conditional-or-counterexample"
    return "uncertain"


def label_ledger(first,second,screen):
    require(all(first["facts"][k]==second["facts"][k] for k in ("screen_sha","value_sha","mode")) and
            first["facts"]["review_seed_family"]!=second["facts"]["review_seed_family"],
            "new evidence cannot replace independent review")
    result={}
    for key,d in screen["decisions"].items():
        a,b=(x["data"]["roots"][key] for x in (first,second))
        old_choices=first["predictions"][key]["choice"]
        new_choices=second["predictions"][key]["choice"]
        require(old_choices==new_choices,"review trials changed model inference")
        require({(action,s,l["hash"]) for action,group in a["leaves"].items() for s,l in group.items()}==
                {(action,s,l["hash"]) for action,group in b["leaves"].items() for s,l in group.items()},
                "same search trajectory was not restored")
        old_effect,new_effect=(effect(r,d["base"],d["candidate"]) for r in (a,b))
        labels={}
        for teacher in ("legacy","state"):
            action=old_choices[teacher]
            x,y=(effect(r,d["base"],action) for r in (a,b))
            labels[teacher]=dict(action=action,first=x,second=y,safe=reliable(x,y) and action!=d["base"])
        old_label=dict(action=d["candidate"],admitted=d["admitted"],stress=d["stressed"],
                       current_scope=scope(old_effect,new_effect,d["stressed"]),first=old_effect,second=new_effect)
        result[key]=dict(root_hash=a["hash"],cut=a["cut"],old_label=old_label,proposals=labels)
    return dict(schema="full-mode-lesson-reanalysis-v1",first=first["facts"],second=second["facts"],rows=result)


def matched_lessons(ledger):
    return [k for k,r in ledger["rows"].items()
            if all(r["proposals"][m]["safe"] for m in ("legacy","state")) and
            r["proposals"]["legacy"]["action"]!=r["proposals"]["state"]["action"]]


def evaluate_students(folder,ledger,screen,review,keys):
    source=read(PRIOR/"students.json")
    parent=source["models"]["window"]
    examples_old,_,_=examples(folder)
    anchors,_=split(examples_old)
    roots={}
    for key in keys:
        r=review["data"]["roots"][key]
        trials={f"{a}/{t}/0/{s}":dict(result=r["review"][a][f"{t}/{s}"])
                for a,t,s in product(r["review"],range(3),range(8))}
        roots[key]=dict(x=screen["review"][key]["x"],trials=trials)
    panels={}
    for teacher in ("legacy","state"):
        labels={k:dict(admitted=True,base=screen["decisions"][k]["base"],
                       candidate=ledger["rows"][k]["proposals"][teacher]["action"]) for k in keys}
        model,count=train(parent,anchors,roots,labels)
        require(count==2*len(keys),"unequal teacher course size")
        path=folder/(teacher+"-student.json")
        write_receipt(path,model)
        panels[teacher]=games(folder,parent,read(path),2843600)
    return dict(panels=panels,state_vs_legacy=compare_panels(panels["state"],panels["legacy"]))


def main(audit):
    folder=ROOT/"build/reanalysis"/os.environ["JUNQI_SERIAL_KEY"]
    wire,_=frozen_wire()
    wire+=" "+model_wire(read(FROZEN/"student.json")["models"]["window"],"window")
    screen=read(SCREEN)
    if not audit:
        folder.mkdir(parents=True,exist_ok=True)
        build(folder)
        capture=rerun(folder,wire,screen,False)
        write_receipt(folder/"reanalyzed.json",capture)
        print(json.dumps(dict(restored=len(capture["data"]["roots"]),facts=capture["facts"])))
    else:
        first=read(folder/"reanalyzed.json")
        fresh=rerun(folder,wire,screen,True)
        ledger=label_ledger(first,fresh,screen)
        old=Counter(r["old_label"]["current_scope"] for r in ledger["rows"].values() if r["old_label"]["admitted"])
        selected=matched_lessons(ledger)
        qualified={m:sum(r["proposals"][m]["safe"] for r in ledger["rows"].values()) for m in ("legacy","state")}
        ledger.update(originally_admitted=old,eligible_teacher_lessons=qualified,matched=selected,
                      student_update="isolated-comparison" if len(selected)>=2 else "retain-frozen-student")
        write_receipt(folder/"recheck.json",fresh)
        write_receipt(folder/"label-ledger.json",ledger)
        if len(selected)>=2:
            write_receipt(folder/"student-comparison.json",evaluate_students(folder,ledger,screen,fresh,selected))
        print(json.dumps(dict(restored=len(ledger["rows"]),old_labels=old,qualified=qualified,matched=len(selected),
                              student_update=ledger["student_update"])))
        print("JUNQI_RESEARCH inconclusive")
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
