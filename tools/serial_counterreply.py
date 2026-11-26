"""Compare locked actions after natural play and an independently reviewed counterreply."""
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

PRIOR = ROOT/"build/reply-budget/audit-full-mode-reply-budget-allocation-a7af3315a27e0323736f"
WITNESSES = ("4d006216776b", "e3d110f645f4", "bdd329d7770b")


def sources(new):
    past=read(PRIOR/"confirmation.json")["data"]
    if not new:
        chosen={}
        for prefix in WITNESSES:
            key,=tuple(k for k in past["roots"] if k.startswith(prefix))
            root=past["roots"][key]
            chosen[key]=dict(past["sources"][key],cap=root["cut"],old=root["baseline"],alt=root["arms"]["wide"]["action"])
        return chosen
    sources=(*past["sources"].values(),*read(PRIOR/"development.json")["data"]["sources"].values(),
             *archives()[0].values(),*candidate_replays().values(),*relay_sources(False).values(),*relay_sources(True).values())
    excluded={row["family"] for row in sources if "family" in row}
    unique={}
    for row in sorted(read(SOURCE/"probe.json")["rows"],key=lambda x:(-x["verified_plies"],x["key"])):
        if row["family"] in excluded or row["prior_layout"] or row["prior_trajectory"] or row["direction"] not in (0,1):
            continue
        if min(row["verified_plies"],row["notice"]-1 if row["notice"] else 120)>=120:
            unique.setdefault(row["family"],row)
    chosen={}
    for suffix in (".jqh",".jgs"):
        ordered=sorted((r for r in unique.values() if r["format"]==suffix),
                       key=lambda r:hashlib.sha256(("counterreply:"+r["family"]).encode()).digest())
        require(len(ordered)>=10,"missing fresh replay families")
        for i,r in enumerate(ordered[:10]):
            chosen[r["key"]]=dict(key=r["key"],family=r["family"],format=suffix,ccw=r["direction"],cap=(24,72,120)[i%3],
                                  old=-1,alt=-1)
    return chosen


def manifest(selected):
    rows=[]
    for line in route_wire(selected).splitlines():
        tag,key,direction,cut,seed,tape=line.split(maxsplit=5)
        record=selected[key]
        rows.append(f"{tag} {key} {direction} {cut} {seed} {record['old']} {record['alt']} {tape}")
    return "\n".join(rows)+"\n"


def collect(folder,weights,fresh):
    selected=sources(fresh)
    payload=native(folder,"counterreply-audit",weights+"\n"+manifest(selected))
    roots,coverage,games,controls={},{},{},[]
    for line in payload.splitlines():
        tag,key,*fields=line.split()
        if tag=="C":
            controls.append([int(key),*map(int,fields)])
            continue
        require(key in selected,"unknown case")
        if tag=="A":
            verified,diverged,why,examined=fields
            coverage[key]=dict(verified=int(verified),diverged=int(diverged),why=why,examined=int(examined))
        elif tag=="R":
            cut,seat,base,alt,order,trial=map(int,fields)
            require(cut==selected[key]["cap"] and (not fresh or trial==int(order>=4)),"invalid opportunity")
            roots[key]=dict(cut=cut,seat=seat,base=base,alt=alt,order=order,trial=trial,replies={})
        elif tag=="P":
            arm,action,rank,target,actor,legal,team,flag,mover,steps,reply=map(int,fields)
            require(arm in (0,1) and max(team,flag,mover)<=legal<=steps and (reply>=0)==(legal>0),
                    "invalid opponent response census")
            roots[key]["replies"][str(arm)]=dict(action=action,rank=rank,target=target,actor=actor,legal=legal,
                                                team=team,flag=flag,mover=mover,steps=steps,reply=reply)
        else:
            require(tag=="G","invalid counterreply output")
            cut,arm,table,stream,action,outcome,ply,*extra=map(int,fields)
            require(cut==selected[key]["cap"] and outcome in (-1,0,1,2) and ply>=cut,"invalid rollout")
            bucket=games.setdefault(key,{})
            require(f"{table}/{arm}/{stream}" not in bucket,"duplicate rollout")
            bucket[f"{table}/{arm}/{stream}"]=dict(action=action,outcome=outcome,ply=ply)
    require(len(controls)==1 and selected.keys()==coverage.keys() and
            set(roots)=={k for k,r in coverage.items() if r["examined"]},"coverage mismatch")
    require(games.keys()=={k for k,r in roots.items() if r["trial"]},"incomplete trial selection")
    require(fresh or len(games)==3,"locked witness restoration failed")
    for key,by_game in games.items():
        require(roots[key]["cut"]<=coverage[key]["verified"] and
                set(by_game)=={f"{t}/{a}/{s}" for t,a,s in product(range(2),range(4),range(4))},"unpaired replies")
        for t,a,s in product(range(2),range(4),range(4)):
            row=by_game[f"{t}/{a}/{s}"]
            first=roots[key][("base","alt")[a//2]]
            require(row["action"]==first and roots[key]["replies"][str(a//2)]["action"]==first,
                    "review did not take locked action")
            if a%2 and roots[key]["replies"][str(a//2)]["reply"]<0:
                require(row==by_game[f"{t}/{a-1}/{s}"],"counterfactual changed a branch that had no response")
    return dict(controls=controls,roots=roots,sources=selected,coverage=coverage,games=games)


def summarize(data):
    by_root={}
    for key,rows in data["games"].items():
        table={}
        for t,pressure in product(range(2),range(2)):
            old,new=([rows[f"{t}/{arm}/{s}"]["outcome"] for s in range(4)] for arm in (pressure,2+pressure))
            diffs=[score_difference(a,b) for a,b in zip(old,new,strict=True)]
            table[f"{t}/{pressure}"]=dict(old=old,new=new,difference=[sum(d[i] for d in diffs)/4 for i in range(2)])
        by_root[key]=table
    effects={}
    for t,p in product(range(2),range(2)):
        name=f"{t}/{p}"
        ds=[r[name]["difference"] for r in by_root.values()]
        effects[name]=contrast(ds) if len(ds)>1 else None
    return dict(coverage=Counter(v["why"] for v in data["coverage"].values()), triggered=len(by_root),
                reviewed=sum(map(len,data["games"].values())),table=effects,by_root=by_root)


def main(audit):
    folder=ROOT/"build/counterreply"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        folder.mkdir(parents=True,exist_ok=True)
        build(folder)
    roster=model_roster()["historical"]
    weights=" ".join(map(str,(*roster["weights"],0)))
    data=collect(folder,weights,audit)
    stats=summarize(data)
    if audit:
        previous=read(folder/"locked.json")
        require(previous["weights"]==roster["artifact_sha256"] and set(previous["data"]["sources"]).isdisjoint(
            data["sources"]),"frozen model or source separation failed")
        print("JUNQI_RESEARCH inconclusive")
    write_receipt(folder/("new-sources.json" if audit else "locked.json"),
                  dict(data=data,stats=stats,weights=roster["artifact_sha256"]))
    print(json.dumps(stats))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
