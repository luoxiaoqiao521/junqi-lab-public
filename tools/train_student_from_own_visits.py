"""Evaluate one feedback iteration at matched label and optimization budgets."""
from collections import Counter
from hashlib import sha256
import json
import os
import sys

from cross_partner_deadline_student import encode, persist, subset
from export_early_deadline_teacher_pairs import parse, public_record
from measure_deadline_opening import ROOT, build, model_roster, paired, require
from train_early_deadline_student import WIDTH, evaluation, fit_student, model_arguments, native, panels


PARENT = (ROOT/"build/deadline-opening/train-cross-partner-deadline-student-f8184e6b396e5fec09dd"
          /"historic_teacher.json")
PARENT_NAME = "historic_teacher"
START, COUNT, EVAL, EVAL_COUNT = 1094000,24,1094200,32
TRAINED = ("frozen_visits","own_visits")
CONTACT_FIELDS = ("legal_contact","shortlisted_contact","selected_contact","attacker_wins")


def export(directory, parent, feedback, start=START, count=COUNT, salt=0, parent_name=PARENT_NAME):
    roster=model_roster()
    weights=[w for name in ("candidate","historical") for w in roster[name]["weights"]]
    payload=" ".join(map(str,(start,count,*weights)))
    mode="student-feedback-visit" if feedback else "student-feedback-frozen"
    if salt:
        require(feedback,"reseeded teacher needs the declared visitation policy")
        mode="student-feedback-reseeded"
        payload+=f" {salt}"
    if feedback:
        payload+=" "+model_arguments(parent,parent_name)
    output=native(directory,mode,payload)
    moves, remaining={},[]
    for line in output.splitlines():
        if line.startswith("S "):
            seed,seat,table,initial,played=map(int,line.split()[1:])
            key=(seed,seat,table)
            require(key not in moves,"duplicate visit decision")
            moves[key]=(initial,played)
        else:
            remaining.append(line)
    roots,teachers,branches,tracked=parse("\n".join(remaining),start,count)
    expected={(s,j,c//2) for s,j,c in roots if c%2==0}
    require(moves.keys()==expected,"visit decisions not aligned with opening ledger")
    for (s,j,t),(first,played) in moves.items():
        key=(s,j,2*t)
        require(first==teachers[key][0] and played in branches[key] and (feedback or first==played),
                "visited first action outside declared classroom")
    records={}
    for key in sorted(roots):
        row=public_record(key,roots[key],teachers[key],branches[key],start)
        row["branches"]={str(move):label for move,label in row["branches"].items()}
        records[key]=row
    return records,moves,tracked


def classrooms(a,b):
    old,old_moves,old_games=a
    new,new_moves,new_games=b
    require(old_games==new_games and old_moves.keys()==new_moves.keys(),"first-window exposure shifted")
    common=sorted(old.keys()&new.keys())
    hist,own=[],[]
    counts=Counter()
    for key in common:
        left,right=old[key],new[key]
        previous=key[:2]+(key[2]//2,)
        changed=old_moves.get(previous)!=new_moves.get(previous)
        phase=left["phase"]
        if not phase or not changed:
            require(left==right,"identical prefix generated different labels")
        if left["partition"]!="development_train":
            continue
        cap=min(len(left["branches"]),len(right["branches"]))
        hist.append(subset(left,cap))
        own.append(subset(right,cap))
        counts[f"phase_{phase}_roots"]+=1
        counts[f"phase_{phase}_changed_observation"]+=left["observation"]!=right["observation"]
        counts["labels_per_arm"]+=cap
    require(counts["phase_1_changed_observation"]>0,"student visits did not change any training root")
    counts.update(first_action_changes=sum(old_moves[k]!=new_moves[k] for k in old_moves),
                  only_frozen_roots=len(old)-len(common),only_own_roots=len(new)-len(common))
    return hist,own,dict(counts)


def compare(arms):
    panel=panels(arms)
    for partner in (1,2):
        for table in (0,1):
            cell=panel[f"{partner}/{table}"]
            rows={name:[r for r in all_rows if (r["partner"],r["table"])==(partner,table)]
                  for name,all_rows in arms.items()}
            for name,values in rows.items():
                cell[name]["contact"]={field:sum(r[field] for r in values) for field in CONTACT_FIELDS}
            for name in (PARENT_NAME,"frozen_visits"):
                cell[f"own_vs_{name}"]=paired(rows[name],rows["own_visits"],cell[name],cell["own_visits"])
            cell["own_vs_frozen_visits"]["win_to_loss"]=sum(a["outcome"]==2 and b["outcome"]==0
                for a,b in zip(rows["frozen_visits"],rows["own_visits"],strict=True))
    return panel


def main(audit):
    directory=ROOT/"build/deadline-opening"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        build(directory)
    source=PARENT.read_bytes()
    parent=json.loads(source)
    hist,own,counts=classrooms(export(directory,parent,False),export(directory,parent,True))
    model_paths=[]
    for name,rows in zip(TRAINED,(hist,own),strict=True):
        wire=encode(rows)
        persist(directory/f"{name}-data.json",wire,audit)
        model=fit_student(directory,rows,((name,WIDTH),),sha256(wire).hexdigest(),(parent,PARENT_NAME))
        model["parent"]=sha256(source).hexdigest()
        require(model["labels"]==counts["labels_per_arm"] and model["mean"]==parent["mean"]
                and model["scale"]==parent["scale"] and model["models"][name]!=parent["models"][PARENT_NAME],
                "warm start or parameter update failed")
        path=directory/f"{name}.json"
        persist(path,encode(model),audit)
        model_paths.append(path)
    arms=evaluation(directory,(parent,*(json.loads(p.read_bytes()) for p in model_paths)),EVAL,EVAL_COUNT,True)
    result=compare(arms)
    persist(directory/"summary.json",encode(dict(counts=counts,result=result,games=arms)),audit)
    print(json.dumps(dict(counts=counts,result=result),sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(sys.argv[1:]==["audit"])
