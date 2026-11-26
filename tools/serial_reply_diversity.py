"""Separate actual first-reply coverage from the allocation of the same search plies."""
from collections import Counter
from itertools import product
import json
from math import isfinite
import os
import sys

from serial_dynamic_reply import ROOT, build, frozen_wire, native, require
from serial_formation_matrix import contrast, score_difference
from serial_full_student_evolution import digest
from serial_quiet_options import FROZEN
from serial_reply_window import model_wire
from serial_visited_reply_value import choose, manifest
from serial_worker import read, write_receipt

ARMS=("fixed","fixed_diverse","priority","priority_diverse")


def collect(folder,wire,audit):
    spec,tape=manifest(audit,seed_offset=6900 if audit else 6100,archive_offset=6 if audit else 18,
                       include_witness=False,cuts=(1,24,49,66))
    roots,missing={},{}
    for tag,key,*fields in map(str.split,native(folder,"reply-diversity-cross",wire+"\n"+tape).splitlines()):
        require(key in spec,"undeclared source")
        if tag=="M":
            missing[key]=fields
        elif tag=="R":
            cut,h,n=map(int,fields)
            roots[key]=dict(hash=h,cut=cut,actions=n,scores={m:{} for m in ARMS},review={})
        elif tag=="S":
            arm,move,score,*rest=fields
            row=dict(zip(("moves","changed","checked","forced","distinct","micros"),map(int,rest),strict=True))
            root=roots[key]
            require(isfinite(float(score)) and row["moves"]<=24 and row["changed"]<=2 and
                    0<=row["forced"]<=2 and 0<=row["distinct"]<=4,"invalid compute assignment")
            root["scores"][ARMS[int(arm)]][move]=dict(row,score=float(score))
        else:
            require(tag=="G","unrecognized fixed root evidence")
            action,table,stream,cut,result,plies,final=map(int,fields)
            root=roots[key]
            idx=f"{action}/{table}/{stream}"
            require(idx not in root["review"] and root["cut"]==cut and result in (-1,0,1,2),"invalid evaluator trace")
            root["review"][idx]=result
    require(set(roots)|set(missing)==set(spec) and len(roots)>=16,"incomplete planned sample")
    for root in roots.values():
        actions=list(root["scores"]["fixed"])
        require(len(actions)==root["actions"] and all(list(x)==actions for x in root["scores"].values()) and
                {f"{a}/{t}/{s}" for a,t,s in product(actions,range(4),range(8))}==root["review"].keys(),
                "candidate or review grid drift")
        for action in actions:
            a,b,c,d=[root["scores"][m][action] for m in ARMS]
            require(a["forced"]==c["forced"]==0 and b["forced"]==d["forced"] and
                    a["distinct"]==c["distinct"] and b["distinct"]==d["distinct"],
                    "allocation changed first-reply coverage")
        root["chosen"]={m:choose(actions,x) for m,x in root["scores"].items()}
    return dict(spec=spec,roots=roots,missing=missing)


def difference(root,old,new,t):
    a,b=(root["chosen"][m] for m in (old,new))
    ds=[(0.,0.) if a==b else score_difference(root["review"][f"{a}/{t}/{s}"],root["review"][f"{b}/{t}/{s}"])
        for s in range(8)]
    return tuple(sum(x)/8 for x in zip(*ds,strict=True))


def summarize(data):
    roots=list(data["roots"].values())
    work={}
    for arm in ARMS:
        work[arm]=dict(Counter({k:sum(r[k] for root in roots for r in root["scores"][arm].values())
                                for k in ("moves","changed","checked","forced","distinct","micros")}))
    estimates={}
    for tag,old,new in (("diversity","fixed","fixed_diverse"),("priority","fixed","priority"),
                        ("priority_with_diversity","fixed_diverse","priority_diverse")):
        estimates[tag]=dict(changed=sum(r["chosen"][old]!=r["chosen"][new] for r in roots),
                            tables={t:contrast([difference(r,old,new,t) for r in roots]) for t in range(4)})
    rows={}
    for t in range(4):
        small=[difference(r,"fixed","priority",t) for r in roots]
        expanded=[difference(r,"fixed_diverse","priority_diverse",t) for r in roots]
        rows[t]=contrast([(c-b,d-a) for (a,b),(c,d) in zip(small,expanded,strict=True)])
    return dict(work=work,estimates=estimates,interaction=rows,roots=len(roots),
                root_trials=sum(len(r["review"]) for r in roots))


def main(audit):
    folder=ROOT/"build/reply-diversity"/os.environ["JUNQI_SERIAL_KEY"]
    wire,_=frozen_wire()
    wire+=" "+model_wire(read(FROZEN/"student.json")["models"]["window"],"window")
    if not audit:
        folder.mkdir(parents=True,exist_ok=True)
        build(folder)
    data=collect(folder,wire,audit)
    stats=summarize(data)
    if audit:
        prior=read(folder/"development.json")
        require(prior["wire"]==digest(wire) and set(prior["data"]["spec"]).isdisjoint(data["spec"]) and
                {r["family"] for r in prior["data"]["spec"].values() if "family" in r}.isdisjoint(
                r["family"] for r in data["spec"].values() if "family" in r),"factorial confirmation leaked")
        print("JUNQI_RESEARCH inconclusive")
    write_receipt(folder/("confirmation.json" if audit else "development.json"),dict(data=data,summary=stats,wire=digest(wire)))
    print(json.dumps(stats))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
