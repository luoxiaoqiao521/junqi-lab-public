"""Separate six-ply score terms while holding simulated branches constant."""
from collections import Counter, defaultdict
from itertools import product
import json
from math import isclose
import os
from statistics import mean
import sys

from full_mode_runtime import FullModeRun, sha
from serial_dynamic_reply import ROOT, build, require
from serial_formation_matrix import contrast, score_difference
from serial_full_mode_inheritance_confirm import frozen_inputs
from serial_full_scoring_audit import branches, collect, discover, run, verified_scope
from serial_worker import read, write_receipt

PREVIOUS=ROOT/"build/full-scoring-audit/confirm-actual-scoring-splits-and-reference-search-bbde4110d57d3af71fdb"
TERMS=("movable","distance","adjacency","early_finish")
NEW=range(2863100,2863112)


def prepare(folder):
    scope=FullModeRun(folder)
    frozen_inputs(scope)
    files=("engine/rules.json","engine/deadline_opening.exe","reset.json","inherited.json","bundle.json")
    require(all(sha(folder/name)==read(PREVIOUS/"source.json")[name] for name in files if not name.endswith(".exe")),
            "teacher comparator has different frozen policy or rules")
    build(scope.engine)
    write_receipt(folder/"source.json",{name:sha(folder/name) for name in files})
    write_receipt(folder/"manifest.json",[dict(seed=s,cut=(2,33,62,89)[i%4],table=2,group="new")
                                          for i,s in enumerate(NEW)])


def recompute(scope,wire,roots):
    tape="\n".join(f"{s} {r['table']} {r['request_cut']} 3" for s,r in roots.items())
    leaves=defaultdict(dict)
    for tag,seed,action,sample,digest,settlement,score,*tail in map(str.split,run(scope,wire,tape).splitlines()):
        require(tag=="L","unknown leaf receipt")
        s,a,i=map(int,(seed,action,sample))
        require(a in roots[s]["actions"] and (a,i) not in leaves[s],"unexpected six-ply sample")
        parts=tuple(map(float,tail[:4]))
        points=float(score)
        if int(settlement)==0:
            require(isclose(sum(parts),points,rel_tol=1e-12,abs_tol=1e-12),"ongoing score did not decompose")
        leaves[s][a,i]=dict(hash=digest,settlement=int(settlement),points=points,
                             terms=parts,plies=int(tail[4]))
    for s,root in roots.items():
        shortlist=sorted((a for a,x in root["actions"].items() if x["order"]>=0),
                          key=lambda a:root["actions"][a]["order"])
        require(leaves[s].keys()==set(product(shortlist,range(4))),"incomplete fixed six-ply candidate matrix")
        for a in shortlist:
            rows=[leaves[s][a,i] for i in range(4)]
            require(isclose(sum(x["points"] for x in rows),root["actions"][a]["six"],abs_tol=1e-12) and
                    sum(x["plies"] for x in rows)==root["actions"][a]["plies"],"old six-ply scores or budget changed")
        for j,term in enumerate(TERMS):
            root[term]=max(shortlist,key=lambda a:sum(leaves[s][a,i]["points"]-leaves[s][a,i]["terms"][j] for i in range(4)))
    return leaves


def compare_panel(scope,wire,roots,streams):
    arms=(*TERMS,"six","reference","inherited")
    choices={s:list(dict.fromkeys(r[x] for x in arms)) for s,r in roots.items()}
    outcomes,work=branches(scope,wire,roots,choices,(2,3,4),streams)
    tables={}
    for desk in (2,3,4):
        tables[desk]={}
        for arm in (*TERMS,"reference"):
            diffs=[]
            for s,r in roots.items():
                v=[(0,0) if r[arm]==r["six"] else score_difference(outcomes[s,r["six"],desk,i],outcomes[s,r[arm],desk,i])
                   for i in streams]
                diffs.append(tuple(map(mean,zip(*v,strict=True))))
            tables[desk][arm]=contrast(diffs)
    return dict(tables=tables,changed={a:sum(r[a]!=r["six"] for r in roots.values()) for a in (*TERMS,"reference")},
                work=work,runs=len(outcomes),truncations=sum(x<0 for x in outcomes.values()),
                outcomes=[[*k,v] for k,v in outcomes.items()])


def main(audit):
    folder=ROOT/"build/full-teacher-terms"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        prepare(folder)
    scope,wire=verified_scope(folder)
    if audit:
        roots=collect(scope,wire,read(folder/"manifest.json"))
        search=discover(scope,wire,roots)
    else:
        data=read(PREVIOUS/"development.json")
        roots={int(s):dict(r,actions={int(a):row for a,row in r["actions"].items()}) for s,r in data["roots"].items()}
        search=data["search"]
    leaves=recompute(scope,wire,roots)
    streams=range(19400,19406) if audit else range(19300,19306)
    result=compare_panel(scope,wire,roots,streams)
    stats=Counter(str(row["settlement"]) for values in leaves.values() for row in values.values())
    output=dict(result,leaves=[[s,a,i,v] for s,rows in leaves.items() for (a,i),v in rows.items()],
                roots={s:{k:v for k,v in r.items() if k!="actions"} for s,r in roots.items()},settlements=stats,
                search=search,early_fallbacks=[s for s,r in roots.items() if r["request_cut"]>r["cut"]])
    write_receipt(folder/("confirmation.json" if audit else "development.json"),output)
    if audit:
        winner=[term for term in TERMS if all(result["tables"][t][term]["interval"][0]>0 for t in (2,3,4))]
        print("JUNQI_RESEARCH "+("supported" if winner else "inconclusive"))
    print(json.dumps({k:v for k,v in output.items() if k not in ("roots","outcomes","leaves")}))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
