"""Replay an early decision and the first allied replan as a paired 2x2 intervention."""
from collections import Counter
from itertools import product
import json
import os
import sys

from serial_evaluator_panel import ROOT, build, model_roster, native, require
from serial_formation_matrix import contrast, score_difference
from serial_natural_routes import route_wire
from serial_worker import read, write_receipt

RELAY = ROOT/"build/relay-setup/audit-full-mode-relay-setup-and-conversion-f1632fd88c0490777213"


def locked_roots():
    cases,expected={},{}
    for batch in ("development","confirmation"):
        evidence=read(RELAY/f"{batch}.json")["result"]
        for r in evidence["roots"]:
            if r["phase"]!=0 or not r["test"]:
                continue
            key=r["key"]
            require(key not in cases,"repeated source")
            cases[key]=dict(evidence["sources"][key],cap=r["cut"],original=r["base"],opening=r["take"],batch=batch)
            expected[key]={f"{table}/{stream}/{arm}":[score,finish]
                           for other,cut,arm,table,stream,action,score,finish,*_ in evidence["games"]
                           if other==key and cut==r["cut"]}
    require(len(cases)==10 and all(len(v)==16 for v in expected.values()),"historical relay panel drifted")
    return cases,expected


def chain_wire(cases,audit):
    wire=route_wire(cases)
    if not audit:
        return wire
    rows=[]
    for line in wire.splitlines():
        tag,key,direction,cut,seed,tape=line.split(maxsplit=5)
        rows.append(" ".join((tag,key,direction,cut,str(int(seed)+15607),tape)))
    return "\n".join(rows)+"\n"


def collect(folder,cases,weights,previous,audit=False):
    raw=native(folder,"relay-chain",weights+"\n"+chain_wire(cases,audit))
    rows,checked={},set()
    for tag,key,*fields in map(str.split,raw.splitlines()):
        require(key in cases,"new source entered retrospective panel")
        cut=int(fields[0])
        require(cut==cases[key]["cap"],"incorrect cut")
        if tag=="A":
            checked.add(key)
            continue
        table,stream,arm,first,score,finish,digest,handoff,cost,before,after=map(int,fields[1:])
        require(tag=="L" and table in range(2) and stream in range(4) and arm in range(4) and score in (-1,0,1,2) and
                first==cases[key]["opening" if arm in (1,2) else "original"] and finish>=cut and
                (0<cost<=96 if arm>=2 and handoff else cost==0),"invalid replan row")
        rows.setdefault(key,{})[f"{table}/{stream}/{arm}"]=[score,finish,digest,handoff,cost,before,after]
    require(checked==cases.keys() and rows.keys()==checked,"missing locked replay")
    expected={f"{t}/{s}/{a}" for t,s,a in product(range(2),range(4),range(4))}
    for key,root in rows.items():
        require(root.keys()==expected,"unpaired relay experiment")
        if not audit:
            require(all(root[k][:2]==v for k,v in previous[key].items()),"A/B no longer reproduce historical branch")
        for t,s,(base,trial) in product(range(2),range(4),((0,3),(1,2))):
            a,b=(root[f"{t}/{s}/{arm}"] for arm in (base,trial))
            require(a[2:4]==b[2:4] and a[5]==b[5],"eligible states diverged before the replan")
            if a[5]==b[6]:
                require(a[:2]==b[:2],"outcomes diverged despite no changed decision")
    return rows


def summarize(cases,rows):
    result={}
    for table in range(2):
        counts={arm:Counter() for arm in range(4)}
        paired={key:[] for key in ("opening","replan_after_opening","replan_without_opening","package")}
        for key in cases:
            root=rows[key]
            for stream,arm in product(range(4),range(4)):
                score,finish,digest,handoff,cost,before,after=root[f"{table}/{stream}/{arm}"]
                counts[arm][{-1:"unresolved",0:"loss",1:"draw",2:"win"}[score]]+=1
                counts[arm]["reach"]+=bool(handoff)
                counts[arm]["changed"]+=before!=after
                counts[arm]["cost"]+=cost
                counts[arm]["win140"]+=score==2 and finish<=140
            for name,(a,b) in zip(paired,((0,1),(1,2),(0,3),(0,2)),strict=True):
                pairs=[(root[f"{table}/{s}/{a}"],root[f"{table}/{s}/{b}"]) for s in range(4)]
                intervals=[(0.,0.) if (a,b) in ((0,3),(1,2)) and old[6]==new[6] else score_difference(old[0],new[0])
                           for old,new in pairs]
                paired[name].append(tuple(sum(x[i] for x in intervals)/4 for i in range(2)))
        result[str(table)]=dict(counts=counts,pairs={name:contrast(ds) for name,ds in paired.items()})
    return result


def main(audit):
    folder=ROOT/"build/relay-chain"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        folder.mkdir(parents=True,exist_ok=True)
        build(folder)
    roster=model_roster()["historical"]
    weights=" ".join(map(str,(*roster["weights"],0)))
    cases,previous=locked_roots()
    rows=collect(folder,cases,weights,previous,audit)
    summary=summarize(cases,rows)
    if audit:
        frozen=read(folder/"development.json")
        require(frozen["cases"]==cases and frozen["weights"]==roster["artifact_sha256"],
                "the new streams changed the root or model")
        print("JUNQI_RESEARCH inconclusive")
    write_receipt(folder/("new-streams.json" if audit else "development.json"),
                  dict(cases=cases,rows=rows,summary=summary,weights=roster["artifact_sha256"]))
    print(json.dumps(summary))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
