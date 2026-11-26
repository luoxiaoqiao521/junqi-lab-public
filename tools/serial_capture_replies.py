"""Screen all legal archive replies and independently replay only locked comparisons."""
from collections import Counter
from hashlib import sha256
import json
import os
import sys

from measure_deadline_opening import build, model_roster, require
from serial_capture_history import COURSE_KEY, SOURCE, select
from serial_replay_phase import bound
from serial_worker import ROOT, write_receipt
from train_early_deadline_student import native

ROOT_FIELDS = ("ply", "original", "legal", "worlds", "rejected", "unresolved", "attributed_losses",
               "bomb_min", "bomb_max")
DEVELOPMENT, CONFIRMATION = 2026112301, 2026112302


def sources():
    excluded = {family for holdout in (False,True) for _,family in select(holdout)[0].values()}
    pool = [r for r in json.loads((SOURCE/"probe.json").read_bytes())["rows"]
            if not r["prior_layout"] and not r["prior_trajectory"] and r["format"]==".jgs"
            and not r["stopped"] and not r["notice"] and r["mid"] and r["result"] in (1,2,3)]
    keys = {COURSE_KEY: ("H",172,0)}
    for row in sorted(pool,key=lambda r:sha256(("capture-replies-v1:"+r["family"]).encode()).digest()):
        if row["family"] in excluded:
            continue
        keys[row["key"]] = "G",row["mid"],row["direction"]
        excluded.add(row["family"])
        if len(keys)==7:
            break
    require(len(keys)==7,"missing independent capture reply prefixes")
    result={}
    for line in (SOURCE/"games.txt").read_text(encoding="utf8").splitlines():
        tag,key,tape=line.split(maxsplit=2)
        if key in keys:
            kind,ply,ccw=keys[key]
            require(tag==kind,"archive format changed")
            if kind=="H":
                tape=" ".join([str(ply),*tape.split()[1:1+300+4*ply]])
            result[key]=(kind,ply,ccw,tape)
    require(result.keys()==keys.keys(),"missing capture reply wire")
    return result


def collect(out, cases, world_seed, reply_seed, locked=None):
    weights=model_roster()["historical"]["weights"]
    wire=[" ".join(map(str,weights))]
    for key,(kind,ply,ccw,tape) in cases.items():
        if locked is None or key in locked:
            target=-1 if locked is None else locked[key]
            wire.append(f"{kind} {key} {ply} {target} {world_seed} {reply_seed} {ccw} {tape}")
    roots={}
    for line in native(out,"capture-replies","\n".join(wire)+"\n").splitlines():
        tag,key,*data=line.split()
        row=list(map(int,data))
        if tag=="R":
            require(key in cases and key not in roots,"unexpected capture reply root")
            roots[key]=dict(zip(ROOT_FIELDS,row,strict=True),actions={})
        else:
            move,world,desk,outcome,plies=row
            root=roots[key]
            trials=root["actions"].setdefault(move,{})
            require(tag=="A" and 0<=world<root["worlds"] and desk in (0,1) and (world,desk) not in trials
                    and outcome in (-1,0,1,2) and 1<=plies<=140,"bad capture reply branch")
            trials[world,desk]=(outcome,plies)
    require(set(roots)==set(cases if locked is None else locked),"missing requested roots")
    for key,root in roots.items():
        actions=root["actions"]
        expected=(root["legal"] if locked is None else 2) if root["worlds"] else 0
        require(root["ply"]==cases[key][1] and len(actions)==expected and
                all(len(trials)==2*root["worlds"] for trials in actions.values()),"incomplete action panel")
    return roots


def comparisons(root, move):
    old=root["actions"][root["original"]]
    new=root["actions"][move]
    return [[sum(bound(old[w,d][0],new[w,d][0],False)[side] for w in range(root["worlds"]))
             for side in (0,1)] for d in (0,1)]


def choose(root):
    if root["worlds"]<4:
        return None
    options=[]
    for move in root["actions"]:
        if move==root["original"]:
            continue
        intervals=comparisons(root,move)
        lower=[a for a,_ in intervals]
        if min(b for _,b in intervals)>0:
            options.append((min(lower),sum(lower),-move))
    return -max(options)[2] if options else None


def summary(root, chosen):
    return {**{k:root[k] for k in ROOT_FIELDS}, "chosen":chosen,
            "rollouts":2*root["worlds"]*len(root["actions"]),
            "original_outcomes":dict(Counter(x[0] for x in root["actions"].get(root["original"],{}).values())),
            "difference_x2_by_desk":comparisons(root,chosen) if chosen in root["actions"] else None}


def confirm(out,cases,locked):
    roots=collect(out,cases,CONFIRMATION,CONFIRMATION,locked)
    require(roots==collect(out,cases,CONFIRMATION,CONFIRMATION,locked),"locked branches did not reproduce")
    results={key:summary(root,locked[key]) for key,root in roots.items()}
    sensitivity={}
    for name,seeds in (("world_only",(CONFIRMATION,DEVELOPMENT)),("reply_only",(DEVELOPMENT,CONFIRMATION))):
        altered=collect(out,cases,*seeds,locked)
        sensitivity[name]={key:summary(root,locked[key]) for key,root in altered.items()}
    return results,sensitivity


def main(audit):
    out=ROOT/"build/capture-replies"/os.environ["JUNQI_SERIAL_KEY"]
    out.mkdir(parents=True,exist_ok=True)
    cases=sources()
    if audit:
        prior=json.loads((out/"screen.json").read_bytes())
        locked={key:row["chosen"] for key,row in prior.items() if row["chosen"] is not None}
        report,sensitivity=confirm(out,cases,locked)
        qualified=sum(r["worlds"]==4 and min(lo for lo,_ in r["difference_x2_by_desk"])>0
                      for r in report.values() if r["difference_x2_by_desk"] is not None)
        write_receipt(out/"confirm.json",dict(qualified=qualified,results=report,sensitivity=sensitivity))
        print(json.dumps(dict(locked=len(locked),qualified=qualified,course=report.get(COURSE_KEY))))
        print("JUNQI_RESEARCH "+("supported" if qualified>=2 else "inconclusive"))
    else:
        build(out)
        controls=native(out,"prove-evaluator-controls","")
        require("E1_bait_after_reply" in controls and "E1_node_limit" in controls,"missing evaluator controls")
        (out/"controls.txt").write_text(controls,encoding="utf8")
        roots=collect(out,cases,DEVELOPMENT,DEVELOPMENT)
        report={key:summary(root,choose(root)) for key,root in roots.items()}
        write_receipt(out/"screen.json",report)
        print(json.dumps(dict(roots=len(report),rollouts=sum(r["rollouts"] for r in report.values()),
                              locked=sum(r["chosen"] is not None for r in report.values()),course=report[COURSE_KEY])))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected audit or no argument")
    main(bool(sys.argv[1:]))
