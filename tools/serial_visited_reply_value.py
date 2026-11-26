"""Calibrate short reply search on reached, independently reviewed full-information positions."""
from collections import Counter
from itertools import product
import json
from math import isfinite
import os
import sys

from serial_dynamic_reply import ROOT, build, frozen_wire, native, require
from serial_engineer_formation import human_partitions
from serial_formation_matrix import contrast, score_difference
from serial_full_student_evolution import digest
from serial_natural_routes import route_wire
from serial_quiet_options import FROZEN
from serial_reply_window import model_wire
from serial_worker import read, write_receipt

PREVIOUS = ROOT/"build/quiet-options/compare-quiet-options-with-local-search-e979a9c597e5cd1ed6bd"
SEEDS = (2831000,2831400)
SCORE_NAMES = ("six","critical","uniform","long")


def manifest(audit, *, seed_offset=0, archive_offset=0, include_witness=True, cuts=(1,19,36)):
    spec = {}
    rows = []
    if not audit and include_witness:
        witnesses = read(PREVIOUS/"development.json")["roots"]
        for seed in ("2825001","2825002"):
            w = witnesses[seed]
            spec[seed] = dict(policy=1, table=w["table"], cut=w["ply"], reference=1, witness=w["hash"])
            rows.append(f'R {seed} {seed} {w["ply"]} 1 {w["table"]} 1')
    for i in range(18):
        key = str(SEEDS[audit]+seed_offset+i)
        cut,policy,table,reference = cuts[i%len(cuts)],(i//3)%3,(i//2)%3,int(i in (0,5,10,15))
        spec[key] = dict(policy=policy,table=table,cut=cut,reference=reference,source="generated")
        rows.append(f"R {key} {key} {cut} {policy} {table} {reference}")
    start=8+archive_offset
    partition = tuple(human_partitions(max(32,start+6)))[audit]
    archives = {key:partition[key] for key in list(partition)[start:start+6]}
    for i,line in enumerate(route_wire(archives).splitlines()):
        tag,key,_,_,seed,tape = line.split(maxsplit=5)
        cut,policy,table = cuts[i%len(cuts)],i%3,(i+1)%3
        spec[key] = dict(policy=policy,table=table,cut=cut,reference=0,source="archived",family=archives[key]["family"])
        rows.append(f"{tag} {key} {seed} {cut} {policy} {table} 0 {tape}")
    return spec,"\n".join(rows)


def choose(actions, estimates, field="score"):
    best=actions[0]
    for action in actions[1:]:
        if estimates[action][field]>estimates[best][field]+1e-9:
            best=action
    return best


def collect(folder,wire,audit,sources=None):
    spec,tape=manifest(audit) if sources is None else sources
    roots,missing={},{}
    for tag,key,*value in map(str.split,native(folder,"visited-reply-value",wire+"\n"+tape).splitlines()):
        require(key in spec,"unexpected root")
        if tag=="M":
            missing[key]=value
        elif tag=="R":
            row=dict(zip(("ply","hash","seat","parent","child","count","reference"),map(int,value),strict=True))
            require(row["reference"]==spec[key]["reference"] and row["ply"]>=spec[key]["cut"] and
                    row["hash"]==spec[key].get("witness",row["hash"]),"root did not restore")
            roots[key]=dict(row, scores={}, review={})
        elif tag=="S":
            arm,action,score,term,moves,settled,seen,moved,ms=value
            row=dict(score=float(score),terminal_only=float(term),moves=int(moves),settled=int(settled),
                     inspected=int(seen),reassigned=int(moved),micros=int(ms))
            require(isfinite(row["score"]) and isfinite(row["terminal_only"]),"nonfinite teacher assessment")
            roots[key]["scores"].setdefault(SCORE_NAMES[int(arm)],{})[action]=row
        else:
            require(tag=="G","unexpected teacher calibration output")
            action,table,stream,cut,result,plies,final=map(int,value)
            root=roots[key]
            k=f"{action}/{table}/{stream}"
            require(k not in root["review"] and cut==root["ply"] and result in (-1,0,1,2),"invalid held-out reply")
            root["review"][k]=dict(result=result,plies=plies,hash=final)
    require(set(roots)|set(missing)==set(spec) and not any("witness" in spec[k] for k in missing),"unaccounted root")
    for key,root in roots.items():
        arms=root["scores"]
        actions=list(arms["six"])
        require(2<=len(actions)==root["count"]<=4 and set(actions)>={str(root["parent"]),str(root["child"])} and
                all(list(a)==actions for a in arms.values()) and len(arms)==3+root["reference"],"candidate lock mismatch")
        full_grid={f"{a}/{t}/{s}" for a,t,s in product(actions,range(3),range(8))}
        require(full_grid==set(root["review"]),"incomplete independent review")
        root["choice"]={name:choose(actions,estimates) for name,estimates in arms.items()}
        root["choice"]["no_leaf"]=choose(actions,arms["six"],"terminal_only")
        root["no_leaf_tied"]=len({e["terminal_only"] for e in arms["six"].values()})==1
        if "witness" in spec[key]:
            baseline=read(PREVIOUS/"development.json")["roots"][key]["arms"][0]["move"]
            require(root["choice"]["six"]==str(baseline),"six-ply baseline changed")
    return dict(roots=roots,missing=missing,spec=spec)


def root_difference(root,old,new,table,streams=range(8)):
    a,b=root["choice"][old],root["choice"][new]
    pairs=[(0.,0.) if a==b else score_difference(root["review"][f"{a}/{table}/{s}"]["result"],
            root["review"][f"{b}/{table}/{s}"]["result"]) for s in streams]
    return tuple(sum(v)/len(pairs) for v in zip(*pairs,strict=True))


def summarize(data):
    fresh=[r for k,r in data["roots"].items() if "witness" not in data["spec"][k]]
    trials=sum(len(r["review"]) for r in fresh)
    require(len(fresh)>=15 and any(p.get("source")=="archived" for k,p in data["spec"].items() if k in data["roots"]),
            "missing normal visited development positions")
    counts=Counter()
    for root in fresh:
        counts["roots_reassigned"]+=any(r["reassigned"] for r in root["scores"]["critical"].values())
        counts["six_leaf_indeterminate"]+=root["no_leaf_tied"]
        for arm,rows in root["scores"].items():
            for row in rows.values():
                for key in ("moves","settled","inspected","reassigned","micros"):
                    counts[f"{arm}_{key}"]+=row[key]
    estimates={}
    for label,old,new in (("critical_vs_six","six","critical"),("critical_vs_uniform","uniform","critical"),
                          ("uniform_vs_six","six","uniform"),("long_vs_six","six","long"),("no_leaf_vs_six","six","no_leaf")):
        use=[r for r in fresh if new in r["choice"]]
        estimates[label]=dict(roots=len(use),switched=sum(r["choice"][old]!=r["choice"][new] for r in use),
                              tables={str(t):contrast([root_difference(r,old,new,t) for r in use]) for t in range(3)})
    return dict(roots=len(fresh),review_trials=trials,counts=counts,estimates=estimates)


def main(audit):
    folder=ROOT/"build/visited-reply"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        folder.mkdir(parents=True,exist_ok=True)
        build(folder)
    wire,_=frozen_wire()
    wire+=" "+model_wire(read(FROZEN/"student.json")["models"]["window"],"window")
    data=collect(folder,wire,audit)
    summary=summarize(data)
    data.update(model_sha256=digest(wire),summary=summary)
    if audit:
        dev=read(folder/"development.json")
        require(set(dev["spec"]).isdisjoint(data["spec"]) and dev["model_sha256"]==data["model_sha256"] and
                {r["family"] for r in dev["spec"].values() if "family" in r}.isdisjoint(
                    r["family"] for r in data["spec"].values() if "family" in r), "confirmation input overlap")
        both=[v["interval"] for batch in (dev,data) for v in
              batch["summary"]["estimates"]["critical_vs_uniform"]["tables"].values()]
        print("JUNQI_RESEARCH "+("supported" if all(lo>0 for lo,_ in both) else
                                "refuted" if all(hi<0 for _,hi in both) else "inconclusive"))
    write_receipt(folder/("confirmation.json" if audit else "development.json"),data)
    print(json.dumps(summary))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
