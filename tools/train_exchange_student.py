"""Resample natural continuations and compare an original and collision-aware student."""
from collections import Counter
from hashlib import sha256
import json
import os
import sys

from cross_partner_deadline_student import encode, persist
from measure_deadline_opening import ROOT, build, measure, paired, require
from train_early_deadline_student import WIDTH, evaluation, fit_student, panels
from train_student_from_own_visits import CONTACT_FIELDS, export


PARENT = (ROOT/"build/deadline-opening/train-student-from-own-visits-53bd170e2ea17edfa6ab"/"own_visits.json")
START, COUNT, EVAL, EVAL_COUNT = 1095000,16,1095200,32
SALTS = (8101,8102,9201,9202)
TRAIN = ("replay_control","collision")
SCORES = {"win":1.,"draw":.5,"loss":0.}


def sign(row, contact, quiet, batch):
    left = row["branches"][contact]["reply_outcomes"][batch:batch+2]
    right = row["branches"][quiet]["reply_outcomes"][batch:batch+2]
    if any(item["final"] is None for item in left+right):
        return None
    return sum(SCORES[a["final"]]-SCORES[b["final"]] for a,b in zip(left,right,strict=True))/2


def precedent(row):
    board = row["observation"]["board"]
    contacts,quiet=[],[]
    for move in row["branches"]:
        (contacts if board[int(move)%129][0]>=0 else quiet).append(move)
    candidates = []
    for a in contacts:
        for b in quiet:
            delta=sign(row,a,b,0)
            if delta is not None:
                candidates.append((abs(delta),int(a),int(b),delta))
    if not contacts or not quiet:
        return {"kind":"missing_arm"}
    if not candidates:
        return {"kind":"unfinished"}
    _,a,b,initial=max(candidates)
    confirmation=sign(row,str(a),str(b),2)
    if confirmation is None:
        status="unfinished"
    elif not initial or not confirmation:
        status="no_repeatable_difference"
    elif initial*confirmation<0:
        status="reversed"
    else:
        status="contact" if initial>0 else "quiet"
    return dict(kind=status, contact=a, quiet=b, batch_gaps=[initial,confirmation])


def collect(directory, parent, audit):
    streams=[export(directory,parent,True,START,COUNT,salt,"own_visits") for salt in SALTS]
    primary, moves, count = streams[0]
    for records,decisions,games in streams[1:]:
        require(records.keys()==primary.keys() and decisions==moves and games==count,
                "reseeded teacher changed visits")
        for key,first in primary.items():
            other=records[key]
            require(other["observation"]==first["observation"] and other["choices"]==first["choices"]
                    and other["branches"].keys()==first["branches"].keys(),"reseeded teacher changed a root or candidate")
    rows = list(primary.values())
    examples = []
    for row,key in zip(rows,primary,strict=True):
        for move,branch in row["branches"].items():
            branch["reply_outcomes"]=[records[key]["branches"][move]["outcome"]
                                      for records,_,_ in streams]
        examples.append(dict(root=key, partition=row["partition"], **precedent(row)))
    data=dict(rows=rows,examples=examples,
              counts=dict(Counter(r["partition"]+"/"+r["kind"] for r in examples)))
    persist(directory/"classroom.json",encode(data),audit)
    print(json.dumps(dict(roots=len(rows), labels=sum(len(r["branches"]) for r in rows), signs=data["counts"]),
                     sort_keys=True))


def assess(arms):
    result = panels(arms)
    parent=arms["own_visits"]
    original,changed=(arms[name] for name in TRAIN)
    comparisons={}
    for key in result:
        p,t=map(int,key.split("/"))
        select=lambda rows:[r for r in rows if (r["partner"],r["table"])==(p,t)]
        a,b,c=map(select,(parent,original,changed))
        comparisons[key]={"collision_vs_control":paired(b,c,measure(b),measure(c)),
                          "collision_vs_parent":paired(a,c,measure(a),measure(c))}
    for desk,row in result.items():
        p,t=map(int,desk.split("/"))
        for name,games in arms.items():
            row[name]["contact"]={field:sum(r[field] for r in games if (r["partner"],r["table"])==(p,t))
                                  for field in CONTACT_FIELDS}
    comparisons["all"]={"collision_vs_control":paired(original,changed,measure(original),measure(changed)),
                        "collision_vs_parent":paired(parent,changed,measure(parent),measure(changed)),
                        "control_vs_parent":paired(parent,original,measure(parent),measure(original))}
    return result,comparisons


def train(directory, parent, audit):
    source = (directory/"classroom.json").read_bytes()
    rows = [row for row in json.loads(source)["rows"] if row["partition"]=="development_train"]
    require(rows and all(len(b["reply_outcomes"])==len(SALTS) for r in rows for b in r["branches"].values()),
            "incomplete classroom")
    artifacts = [parent]
    for name in TRAIN:
        fit=fit_student(directory,rows,((name,WIDTH),),sha256(source).hexdigest(),
                        (parent,"own_visits"),exchange=name=="collision")
        fit["parent"]=sha256(PARENT.read_bytes()).hexdigest()
        path=directory/f"{name}.json"
        persist(path,encode(fit),audit)
        artifacts.append(json.loads(path.read_bytes()))
    arms=evaluation(directory,tuple(artifacts),EVAL,EVAL_COUNT,True)
    result,comparisons=assess(arms)
    receipt=dict(result=result, comparisons=comparisons, games=arms, training_roots=len(rows),
                 labels_per_arm=sum(len(r["branches"]) for r in rows))
    persist(directory/"summary.json",encode(receipt),audit)
    print(json.dumps(dict(roots=len(rows),comparisons=comparisons),sort_keys=True))


def main():
    directory = ROOT/"build/deadline-opening"/os.environ["JUNQI_SERIAL_KEY"]
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    audit = sys.argv[1:]==["audit"]
    parent = json.loads(PARENT.read_bytes())
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        build(directory)
    collect(directory,parent,audit)
    train(directory,parent,audit)
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    main()
