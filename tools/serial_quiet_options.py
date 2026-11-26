"""Compare an external quiet-move hypothesis against local widening under matched search."""
from collections import Counter
from itertools import product
import json
import os
import sys

from serial_dynamic_reply import ROOT, build, frozen_wire, native, require
from serial_formation_matrix import contrast, score_difference
from serial_full_student_evolution import digest
from serial_reply_window import model_wire
from serial_worker import read, write_receipt

FROZEN = ROOT/"build/full-student-evolution/learn-from-matched-student-rank-disagreements-a4bb618c6e3a82778f56"
SEEDS = (2829200,2829500)


def collect(folder, wire, audit):
    declared = [(i,i%2+1) for i in range(SEEDS[audit],SEEDS[audit]+16)]
    if not audit:
        declared[:0]=[(2825001,1),(2825002,1)]
        original = read(FROZEN/"confirmation.json")["rows"]
        by_key = {(r["seed"],r["table"],r["arm"]):r["result"] for r in original}
        require(all(by_key[s,t,3]==2 and by_key[s,t,4]==0 for s,t in declared[:2]),"historical win-to-loss witness changed")
    tape = wire+"\n"+"\n".join(f"{s} {t}" for s,t in declared)
    roots = {}
    missing = {}
    for tag,key,*parts in map(str.split,native(folder,"quiet-options-audit",tape).splitlines()):
        if tag=="M":
            missing[key]=int(parts[0])
        elif tag=="R":
            root=dict(zip(("ply","hash","table","parent","child","path","legal"),map(int,parts),strict=True))
            require(root["parent"]!=root["child"] and root["table"]==dict(declared)[int(key)],"divergence changed")
            roots[key]=dict(**root,nominations=[],arms=[],games={})
        elif tag=="P":
            row=dict(zip(("kind","move","ally_gain","enemy_loss","flag_attack_loss","outside"),map(int,parts),strict=True))
            roots[key]["nominations"].append(row)
        elif tag=="A":
            row=dict(zip(("arm","move","candidates","trials_per_candidate","plies","probes","options_generated","micros"),
                         map(int,parts),strict=True))
            roots[key]["arms"].append(row)
        else:
            require(tag=="G","invalid quiet option output")
            cut,_,table,stream,action,outcome,plies,flag,end=map(int,parts)
            root=roots[key]
            game=f"{action}/{table}/{stream}"
            require(game not in root["games"] and cut==root["ply"] and outcome in (-1,0,1,2),"invalid review trial")
            root["games"][game]=dict(result=outcome,plies=plies,flag=flag,final_hash=end)
    require(set(roots)|set(missing)=={str(s) for s,_ in declared} and (audit or {"2825001","2825002"}<=set(roots)),
            "not all declared openings replayed")
    for root in roots.values():
        a,b,c=root["arms"]
        expected={f"{action}/{t}/{s}" for action,t,s in product(
            {root["parent"],root["child"],*(r["move"] for r in (a,b,c))},range(2),range(8))}
        require(set(root["games"])==expected and [r["arm"] for r in (a,b,c)]==[0,1,2] and
                all(r["plies"]<=96 for r in (a,b,c)) and
                b["candidates"]==c["candidates"]==min(8,root["legal"]),"search or review budget drift")
        for first,second in ((a,b),(b,c),(a,c)):
            if first["move"]==second["move"]:
                require(first["move"] in {int(g.split("/")[0]) for g in expected},"shared choice lost")
    return dict(roots=roots,missing=missing,declared=declared)


def summarize(data,audit):
    roots=[r for key,r in data["roots"].items() if audit or not key.startswith("28250")]
    require(len(roots)>=4,"first-disagreement sample too small to compare")
    review={}
    for name,before,after in (("wide_vs_short",0,1),("quiet_vs_wide",1,2),("quiet_vs_child","child",2)):
        for table in range(2):
            pairs=[]
            for r in roots:
                a,b=[r[index] if isinstance(index,str) else r["arms"][index]["move"] for index in (before,after)]
                samples=[(0.,0.) if a==b else score_difference(r["games"][f"{a}/{table}/{s}"]["result"],
                         r["games"][f"{b}/{table}/{s}"]["result"]) for s in range(8)]
                pairs.append(tuple(sum(values)/8 for values in zip(*samples,strict=True)))
            review[f"{name}/{table}"]=dict(**contrast(pairs),better=sum(lo>0 for lo,_ in pairs),worse=sum(hi<0 for _,hi in pairs))
    counts=Counter()
    for root in roots:
        counts["nominated_outside"]+=sum(p["outside"] for p in root["nominations"])
        counts["roots_with_outside_choice"]+=any(p["outside"] and p["move"]==root["arms"][2]["move"]
                                                 for p in root["nominations"])
        counts["different_wide_quiet"]+=root["arms"][1]["move"]!=root["arms"][2]["move"]
        counts["review_trials"]+=len(root["games"])
        for r in root["arms"]:
            for name in ("plies","probes","options_generated","micros"):
                counts[f'arm{r["arm"]}_{name}']+=r[name]
    return dict(roots=len(roots),counts=counts,comparisons=review)


def main(audit):
    folder=ROOT/"build/quiet-options"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        folder.mkdir(parents=True,exist_ok=True)
        build(folder)
    baseline,_=frozen_wire()
    child=model_wire(read(FROZEN/"student.json")["models"]["window"],"window")
    wire=baseline+" "+child
    data=collect(folder,wire,audit)
    data.update(model_sha256=digest(wire),summary=summarize(data,audit))
    if audit:
        previous=read(folder/"development.json")
        require(set(previous["roots"]).isdisjoint(data["roots"]) and data["model_sha256"]==previous["model_sha256"],
                "quiet options confirmation leaked a source or changed the learner")
        intervals=[r["summary"]["comparisons"][f"quiet_vs_wide/{t}"]["interval"] for r in (previous,data) for t in range(2)]
        print("JUNQI_RESEARCH "+("supported" if all(lo>0 for lo,_ in intervals) else
                                "refuted" if all(hi<0 for _,hi in intervals) else "inconclusive"))
    write_receipt(folder/("confirmation.json" if audit else "development.json"),data)
    print(json.dumps(data["summary"]))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
