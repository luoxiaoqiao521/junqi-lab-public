"""Trace the historical extension gate and compare equal-budget alternatives."""
from collections import Counter
import hashlib
from itertools import product
import json
import os
import sys

from serial_evaluator_panel import ROOT, build, model_roster, native, require
from serial_formation_matrix import contrast, score_difference
from serial_natural_routes import candidate_replays, route_wire
from serial_relay_setup import SOURCE, archives, sources as relay_sources
from serial_worker import read, write_receipt

HISTORICAL = ROOT/"build/evaluator-panel/evaluator-panel-serial-calibration-079e33773261708c2e20/panel.txt"
ARMS = ("fixed", "wide", "extend", "leaf", "matched_short")


def trace_gate(folder, weights):
    raw = native(folder,"trace-evaluator-gate",weights)
    current=[row.split() for row in raw.splitlines() if row.startswith("panel ")]
    old=[row.split() for row in HISTORICAL.read_text(encoding="utf8").splitlines() if row.startswith("panel ")]
    counts=Counter()
    require(len(current)==len(old)==48,"missing historical panel")
    for cols,previous in zip(current,old,strict=True):
        require(len(cols)==54 and cols[:4]==previous[:4],"incomplete legacy trace")
        for i,j in ((4,4),(29,17)):
            require(cols[i]==previous[j] and list(map(float,cols[i+4:i+13]))==list(map(float,previous[j+4:j+13])),
                    "historical short forecasts changed")
            counts["proof_work_changes"]+=cols[i+3]!=previous[j+3]
            counts["proof_bound_changes"]+=cols[i+1:i+3]!=previous[j+1:j+3]
            visits,settled,unresolved,steps,*reasons=map(int,cols[i+13:i+25])
            require(visits+settled+unresolved==3 and sum(reasons)==visits,"missing gate trial")
            counts.update(dict(visits=visits,early_settled=settled,early_unresolved=unresolved,
                               extension_steps=steps,trials=3,positive_gate=sum(reasons[1:])))
    return dict(counts)


def samples(audit):
    already=(*archives()[0].values(),*candidate_replays().values(),*relay_sources(False).values(),*relay_sources(True).values())
    excluded={r["family"] for r in already}
    unique={}
    for r in sorted(read(SOURCE/"probe.json")["rows"],key=lambda x:(-x["verified_plies"],x["key"])):
        cap=min(r["verified_plies"],r["notice"]-1 if r["notice"] else 120)
        if cap<120 or r["family"] in excluded or r["prior_layout"] or r["prior_trajectory"] or r["direction"] not in (0,1):
            continue
        unique.setdefault(r["family"],r)
    selected={}
    for suffix in (".jqh",".jgs"):
        group=sorted((r for r in unique.values() if r["format"]==suffix),
                     key=lambda r:hashlib.sha256(("reply-budget:"+r["family"]).encode()).digest())
        require(len(group)>=16,"missing new archive families")
        for index,r in enumerate(group[int(audit):16:2]):
            selected[r["key"]]=dict(key=r["key"],family=r["family"],format=suffix,ccw=r["direction"],
                                    cap=(24,72,120)[index%3])
    start=2725000 if audit else 2724000
    generated={str(seed):dict(seed=seed,cap=(24,72,120)[i%3]) for i,seed in enumerate(range(start,start+12))}
    wire=route_wire(selected)+"".join(f'S {key} 0 {r["cap"]} {r["seed"]}\n' for key,r in generated.items())
    return dict(selected,**generated),wire


def collect(folder, weights, audit):
    sources,wire=samples(audit)
    raw=native(folder,"reply-budget",weights+"\n"+wire)
    controls, roots, coverage, reviews = [], {}, {}, {}
    for tag,key,*fields in map(str.split,raw.splitlines()):
        if tag=="C":
            controls.append([int(key),*map(int,fields)])
            continue
        require(key in sources,"unknown study root")
        if tag=="A":
            checked,diverged,reason,evaluated=fields
            coverage[key]=dict(checked=int(checked),diverged=int(diverged),reason=reason,evaluated=int(evaluated))
        elif tag=="R":
            cut,seat,legals,action=map(int,fields)
            roots[key]=dict(cut=cut,seat=seat,legals=legals,baseline=action,arms={})
        elif tag=="B":
            cut,arm,action,considered,order,plies,micros,visits,extended,extra,settled,unresolved,*reasons=map(int,fields)
            require(0<=order<considered<=8 and 0<plies<=96 and roots[key]["cut"]==cut and len(reasons)==8 and
                    visits==sum(reasons) and extended==sum(reasons[1:]) and extended<=extra and
                    (arm!=2 or visits+settled+unresolved==considered*3),"actual trial budget or trigger mismatch")
            roots[key]["arms"][ARMS[arm]]=dict(action=action,considered=considered,order=order,plies=plies,micros=micros,
                visits=visits,extended=extended,extra=extra,settled=settled,unresolved=unresolved,reasons=reasons)
        else:
            require(tag=="G" and len(fields)==9,"unexpected native record")
            cut,_,table,stream,action,score,finish,flag,digest=map(int,fields)
            require(roots[key]["cut"]==cut and score in (-1,0,1,2) and finish>=cut,"invalid review outcome")
            by_root=reviews.setdefault(key,{})
            name=f"{action}/{table}/{stream}"
            require(name not in by_root,"duplicate held-out outcome")
            by_root[name]=dict(score=score,finish=finish,flag=flag,digest=digest)
    require(len(controls)==2 and coverage.keys()==sources.keys() and
            set(roots)=={key for key,value in coverage.items() if value["evaluated"]==1},"missing study layer")
    for key,root in roots.items():
        arms=root["arms"]
        require(tuple(arms)==ARMS and arms["fixed"]["plies"]==arms["leaf"]["plies"],
                "leaf arm changed the trial trajectories or lost an arm")
        require(arms["extend"]["plies"]-arms["matched_short"]["plies"]==arms["extend"]["extra"],
                "matched extension changed work before the cutoff")
        selected={root["baseline"],*(item["action"] for item in arms.values())}
        require(set(reviews[key])=={f"{a}/{t}/{s}" for a,t,s in product(selected,range(2),range(4))},
                "incomplete independent action comparisons")
    return dict(controls=controls,sources=sources,coverage=coverage,roots=roots,reviews=reviews)


def summarize(data):
    roots=data["roots"]
    costs={}
    for arm in ARMS:
        costs[arm]=dict(Counter({field:sum(root["arms"][arm][field] for root in roots.values())
                                for field in ("considered","plies","micros","visits","extended","extra","settled","unresolved")}))
        costs[arm]["reasons"]=[sum(root["arms"][arm]["reasons"][i] for root in roots.values()) for i in range(8)]
        costs[arm]["picked_outside_narrow"]=sum(root["arms"][arm]["order"]>=4 for root in roots.values())
    strata={}
    for table in (0,1):
        for arm in ("product",*ARMS):
            scores=[[],[],[]]
            counts=Counter()
            for key,root in roots.items():
                new=root["baseline"] if arm=="product" else root["arms"][arm]["action"]
                bases=(root["baseline"],root["arms"]["fixed"]["action"],root["arms"]["matched_short"]["action"])
                ratings=data["reviews"][key]
                trial=[ratings[f"{new}/{table}/{s}"] for s in range(4)]
                counts["changed_from_product"]+=new!=bases[0]
                counts["changed_from_fixed"]+=new!=bases[1]
                for value in trial:
                    score,ply=value["score"],value["finish"]
                    counts[{-1:"unresolved",0:"loss",1:"draw",2:"win"}[score]]+=1
                    counts["win100"]+=score==2 and ply<=100
                    counts["win140"]+=score==2 and ply<=140
                    counts["loss140"]+=score==0 and ply<=140
                for i,base in enumerate(bases):
                    pairs=[]
                    for s,val in enumerate(trial):
                        prior=ratings[f"{base}/{table}/{s}"]["score"]
                        pairs.append((0.,0.) if new==base else score_difference(prior,val["score"]))
                    bound=tuple(sum(p[i] for p in pairs)/4 for i in range(2))
                    scores[i].append(bound)
                    names=("product","fixed","matched_short")
                    counts[f"improved_{names[i]}"]+=bound[0]>0
                    counts[f"worsened_{names[i]}"]+=bound[1]<0
            strata[f"{table}/{arm}"]=dict(outcomes=counts,vs_product=contrast(scores[0]),vs_fixed=contrast(scores[1]),
                vs_matched_short=contrast(scores[2]),
                plies=costs.get(arm,{}).get("plies",0))
    return dict(roots=len(roots),archives=sum("family" in data["sources"][key] for key in roots),
                stops=Counter(x["reason"] for x in data["coverage"].values()),costs=costs,strata=strata,
                reviews=sum(map(len,data["reviews"].values())))


def main(audit):
    folder=ROOT/"build/reply-budget"/os.environ["JUNQI_SERIAL_KEY"]
    roster=model_roster()["historical"]
    weights=" ".join(map(str,(*roster["weights"],0)))
    if not audit:
        folder.mkdir(parents=True,exist_ok=True)
        build(folder)
        legacy=trace_gate(folder,weights)
        write_receipt(folder/"legacy.json",legacy)
        print(json.dumps(dict(legacy=legacy)))
    data=collect(folder,weights,audit)
    stats=summarize(data)
    if audit:
        prior=read(folder/"development.json")
        require(set(data["sources"]).isdisjoint(prior["data"]["sources"]),"source overlap")
        require(prior["weights"]==roster["artifact_sha256"],"opponent weights changed")
        justified=[arm for arm in ("wide","extend","leaf") if all(
            batch["strata"][f"{table}/{arm}"][against]["interval"][0]>0
            for batch in (prior["stats"],stats) for table in (0,1) for against in ("vs_fixed","vs_product"))]
        print("JUNQI_RESEARCH "+("supported" if justified else "inconclusive"))
    write_receipt(folder/("confirmation.json" if audit else "development.json"),
                  dict(data=data,stats=stats,weights=roster["artifact_sha256"]))
    print(json.dumps(stats))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
