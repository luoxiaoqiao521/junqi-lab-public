"""Run proposal, independent reply screening and optional isolated learning locally."""
from collections import Counter
from itertools import product
import json
import os
import sys

from compare_book_learning import OUTPUTS, fit, np
from serial_dynamic_reply import ROOT, PRIOR, build, frozen_wire, native, require
from serial_formation_matrix import score_difference
from serial_full_student_evolution import digest, games
from serial_quiet_options import FROZEN
from serial_reply_window import examples, judge, model_wire, split
from serial_visited_reply_value import collect, manifest
from serial_worker import read, write_receipt


def independent_review(folder,wire,sources,prior):
    _,tape=sources
    data={}
    missed=set()
    for tag,key,*fields in map(str.split,native(folder,"visited-lesson-review",wire+"\n"+tape).splitlines()):
        if tag=="M":
            missed.add(key)
        elif tag=="R":
            ply,digest_value,seat,base,count=map(int,fields)
            data[key]=dict(ply=ply,hash=digest_value,base=str(base),seat=seat,count=count,x={},trials={})
        elif tag=="F":
            action,*values=fields
            x=list(map(float,values))
            require(len(x)==52 and np.isfinite(x).all(),"invalid training observation")
            data[key]["x"][action]=x
        else:
            require(tag=="V","unknown independent lesson evidence")
            action,t,pressure,s,cut,outcome,actor,reply,changed,h=map(int,fields)
            r=data[key]
            index=f"{action}/{t}/{pressure}/{s}"
            require(index not in r["trials"] and outcome in (-1,0,1,2) and r["ply"]==cut,
                    "incomplete first-reply trial")
            r["trials"][index]=dict(result=outcome,actor=actor,reply=reply,changed=changed,hash=h)
    require(set(data)|missed==set(sources[0]),"incomplete replay admission")
    for key,r in data.items():
        if key not in prior["roots"]:
            require(r["count"]==1,"replay found a different reached root")
            continue
        original=prior["roots"][key]
        require(r["hash"]==original["hash"] and r["base"]==str(original["parent"]) and
                r["x"].keys()==original["scores"]["six"].keys(),"teacher or candidates drifted")
        trial_ids={f"{a}/{t}/{p}/{s}" for a,t,p,s in product(r["x"],range(3),range(2),range(8))}
        require(r["trials"].keys()==trial_ids,"first-reply grid incomplete")
        for a,t,s in product(r["x"],range(3),range(8)):
            natural,probe=[r["trials"][f"{a}/{t}/{p}/{s}"] for p in (0,1)]
            require(natural["actor"]==probe["actor"] and
                    (natural["reply"]!=probe["reply"])==bool(probe["changed"]) and
                    (probe["changed"] or natural["hash"]==probe["hash"]),"forced reply contaminated a matched branch")
    return data


def difference(base,candidate,get,streams):
    return [[sum(points)/len(streams) for points in zip(*(score_difference(
        get(base,t,s),get(candidate,t,s)) for s in streams),strict=True)] for t in range(3)]


def lesson_proposals(discovery,confirmation):
    outcomes={}
    for key,root in discovery["roots"].items():
        r=confirmation[key]
        base=r["base"]
        contenders=[a for a in r["x"] if a!=base]
        first=lambda a,t,s:root["review"][f"{a}/{t}/{s}"]["result"]
        second=lambda a,t,s:r["trials"][f"{a}/{t}/0/{s}"]["result"]
        pressure=lambda a,t,s:r["trials"][f"{a}/{t}/1/{s}"]["result"]
        options=[(a,difference(base,a,first,range(4))) for a in contenders]
        candidate,d=max(options,key=lambda item:(min(lo for lo,_ in item[1]),sum(lo for lo,_ in item[1])))
        replication=difference(base,candidate,first,range(4,8))
        natural=difference(base,candidate,second,range(8))
        stressed=difference(base,candidate,pressure,range(8))
        early=[lo for lo,_ in d]
        reasons=[]
        if min(early)<0 or sum(x>0 for x in early)<2:
            reasons.append("discovery")
        if min(lo for lo,_ in replication)<0 or sum(lo for lo,_ in replication)<=0:
            reasons.append("within_root_replication")
        if min(lo for lo,_ in natural)<0 or sum(lo for lo,_ in natural)<.25:
            reasons.append("new_natural_replies")
        if min(lo for lo,_ in stressed)<-.25:
            reasons.append("response_sensitive")
        if r["x"][base]==r["x"][candidate]:
            reasons.append("feature_alias")
        if any(second(a,t,s)<0 for a,t,s in product((base,candidate),range(3),range(8))):
            reasons.append("unfinished_label")
        outcomes[key]=dict(base=base,candidate=candidate,first=d,replication=replication,natural=natural,
                           stressed=stressed,admitted=not reasons,reasons=reasons)
    return outcomes


def train(parent,anchors,roots,proposals,*,include_new=True,optimizer=fit):
    rows=[]
    for key,proposal in proposals.items():
        if not proposal["admitted"]:
            continue
        r=roots[key]
        for action in (proposal["base"],proposal["candidate"]):
            totals=Counter(r["trials"][f"{action}/{t}/0/{s}"]["result"] for t,s in product(range(3),range(8)))
            rows.append((r["x"][action],[totals[i]/24 for i in range(3)]))
    if not rows:
        return None,0
    count=len(rows)
    for row in anchors:
        rows.extend((x,[int(i==score) for i in range(3)]) for x,score in zip(row["features"],row["scores"],strict=True))
    x=np.clip((np.asarray([v for v,_ in rows])-parent["mean"])/parent["scale"],-6,6)
    y=np.full((len(rows),len(OUTPUTS)),np.nan)
    y[:,:3]=[v for _,v in rows]
    if not include_new:
        y[:count,:3]=np.nan
    parameters=optimizer(x,y,54,24,parent["parameters"],updates=40,rate=.03)
    return dict(mean=parent["mean"],scale=parent["scale"],parameters=parameters),count


def main(audit):
    folder=ROOT/"build/local-lesson-cycle"/os.environ["JUNQI_SERIAL_KEY"]
    frozen=read(PRIOR/"students.json")
    parent=frozen["models"]["window"]
    wire,_=frozen_wire()
    wire+=" "+model_wire(read(FROZEN/"student.json")["models"]["window"],"window")
    artifact=folder/"student.json"
    if not audit:
        folder.mkdir(parents=True,exist_ok=True)
        build(folder)
        sources=manifest(False,seed_offset=1800,archive_offset=6,include_witness=False)
        discovery=collect(folder,wire,False,sources)
        review=independent_review(folder,wire,sources,discovery)
        decisions=lesson_proposals(discovery,review)
        cases,_,_=examples(folder)
        anchors,_=split(cases)
        child,rows=train(parent,anchors,review,decisions)
        if child:
            require(digest(model_wire(child,"window"))!=digest(model_wire(parent,"window")),"update did not change model")
            write_receipt(artifact,dict(models={"board":frozen["models"]["board"],"window":child}))
        stats=dict(roots=len(decisions),admitted=sum(d["admitted"] for d in decisions.values()),action_rows=rows,
                   updates=40 if child else 0,review_trials=sum(len(r["trials"]) for r in review.values()),
                   changed_replies=sum(g["changed"] for r in review.values() for g in r["trials"].values()),
                   reasons=dict(Counter(reason for row in decisions.values() for reason in row["reasons"])))
        write_receipt(folder/"screen.json",dict(stats=stats,decisions=decisions,review=review,discovery=discovery,
                                               source_sha256=digest(wire)))
        print(json.dumps(stats))
    else:
        screen=read(folder/"screen.json")
        require(screen["source_sha256"]==digest(wire) and bool(screen["stats"]["updates"])==artifact.exists(),
                "frozen configs or autonomous decision drift")
        result=dict(stats=screen["stats"],disposition="retain-parent-no-verified-lesson")
        signal="inconclusive"
        if artifact.exists():
            child=read(artifact)
            cases,_,source=examples(folder)
            _,test=split(cases)
            old=judge(folder,frozen,source,test)["errors"]["window"]["counts"]["correct"]
            new=judge(folder,child,source,test)["errors"]["window"]["counts"]["correct"]
            panel=games(folder,parent,child["models"]["window"],2839000)
            watch=[panel["differences"][f"{t}/learn_continuously"]["interval"] for t in range(3)]
            result.update(original_skill=[old,new],panel=panel,disposition="retain-parent-review-isolated-candidate")
            if old<=new and all(lo>0 for lo,_ in watch):
                signal="supported"
                result["disposition"]="candidate-awaits-next-independent-evaluation"
            elif new<old or any(hi<0 for _,hi in watch):
                signal="refuted"
        write_receipt(folder/"decision.json",result)
        print(json.dumps(dict(stats=result["stats"],disposition=result["disposition"],
                              outcome=result.get("panel",{}).get("differences",{}))))
        print("JUNQI_RESEARCH "+signal)
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
