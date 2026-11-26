"""Find natural relay setup and conversion, then verify matched one-action branches."""
from collections import Counter
import hashlib
from itertools import product
import json
import os
import sys

from measure_deadline_opening import ROOT, build, require
from serial_archive_pressure import SOURCE, archives
from serial_formation_matrix import contrast, score_difference
from serial_natural_routes import candidate_replays, route_wire
from serial_route_choice import native
from serial_worker import read, write_receipt


def sources(audit):
    archive,_=archives()
    excluded={x["family"] for x in (*archive.values(),*candidate_replays().values())}
    rows=read(SOURCE/"probe.json")["rows"]
    unique={}
    for row in sorted(rows,key=lambda r:(-r["verified_plies"],r["key"])):
        if row["family"] in excluded or row["prior_layout"] or row["prior_trajectory"] or row["direction"] not in (0,1):
            continue
        cap=min(row["verified_plies"],224,row["notice"]-1 if row["notice"] else 224)
        if cap>=72:
            unique.setdefault(row["family"],dict(key=row["key"],family=row["family"],format=row["format"],
                                                ccw=row["direction"],cap=cap))
    choices={}
    for suffix in (".jqh",".jgs"):
        group=sorted((r for r in unique.values() if r["format"]==suffix),
                     key=lambda r:hashlib.sha256(("natural-relay:"+r["family"]).encode()).digest())
        require(len(group)>=48,"insufficient independent replay families")
        choices.update((r["key"],r) for r in group[int(audit):48:2])
    return choices


def collect(folder,audit):
    selected=sources(audit)
    coverage, roots, games={}, [], []
    for tag,key,*fields in native(folder,"relay-setup",route_wire(selected)):
        require(key in selected,"unexpected source")
        if tag=="A":
            agreed,next_cut,reason,examined=fields
            require(0<=int(agreed)<=selected[key]["cap"] and int(next_cut)==(0 if reason=="matched" else int(agreed)+1),
                    "inconsistent replay boundary")
            coverage[key]=dict(agreed=int(agreed),reason=reason,examined=int(examined))
        elif tag=="P":
            phase,cut,seat,flag,own_turns,observed,base,take,test,spoil_ply,spoiler,spoil_action=map(int,fields)
            require(phase in (0,1) and own_turns in (1,2) and (phase==0 or observed in (1,2)) and (not test or base!=take),
                    "inconsistent relay intervention")
            roots.append(dict(key=key,phase=phase,cut=cut,seat=seat,flag=flag,own_turns=own_turns,
                              observed=observed,base=base,take=take,test=test,
                              first_opponent_interruption=[spoil_ply,spoiler,spoil_action] if spoil_ply else None))
        else:
            require(tag=="G","invalid relay record")
            games.append([key,*map(int,fields)])
    require(coverage.keys()==selected.keys() and all(r["cut"]<=coverage[r["key"]]["agreed"] for r in roots),
            "relay cut outside verified replay")
    expected=[[r["key"],r["cut"],a,t,s,r[("base","take")[a]]]
              for r in roots if r["test"] for t,s,a in product(range(2),range(4),range(2))]
    require([row[:6] for row in games]==expected and all(r[6] in (-1,0,1,2) for r in games),
            "incomplete team outcomes")
    return dict(sources=selected,coverage=coverage,roots=roots,games=games)


def summary(data):
    summary={}
    for phase in (0,1):
        for table in (0,1):
            scores={}
            label={-1:"unresolved",0:"loss",1:"draw",2:"win"}
            counts=[Counter(),Counter()]
            for root in data["roots"]:
                if root["phase"]==phase and root["test"]:
                    scores[(root["key"],root["cut"])]={}
            for key,cut,arm,profile,stream,action,score,ply,*tail in data["games"]:
                if profile==table and (key,cut) in scores:
                    scores[key,cut][arm,stream]=score
                    counts[arm][label[score]]+=1
                    counts[arm]["win140"]+=score==2 and ply<=140
            differences=[tuple(sum(col)/4 for col in zip(*(score_difference(x[0,s],x[1,s]) for s in range(4)),strict=True))
                         for x in scores.values()]
            summary[f"{phase}/{table}"]=dict(roots=len(scores),results=counts,
                                            team_delta=contrast(differences) if len(differences)>1 else None)
    return dict(plies=sum(x["agreed"] for x in data["coverage"].values()),
                examined=sum(x["examined"] for x in data["coverage"].values()),
                prefix=Counter(x["reason"] for x in data["coverage"].values()),
                setup_turns=Counter(r["own_turns"] for r in data["roots"] if r["phase"]==0),
                first_reply=Counter(r["observed"] for r in data["roots"] if r["phase"]==0),
                comparisons=summary,rollouts=len(data["games"]))


def main(audit):
    folder=ROOT/"build/relay-setup"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        folder.mkdir(parents=True,exist_ok=True)
        build(folder)
    result=collect(folder,audit)
    stats=summary(result)
    if audit:
        prior=read(folder/"development.json")
        require({r["family"] for r in prior["result"]["sources"].values()}.isdisjoint(
                r["family"] for r in result["sources"].values()),"source leakage")
        evidence=[block["team_delta"] for batch in (prior["stats"],stats) for block in batch["comparisons"].values()]
        print("JUNQI_RESEARCH "+("supported" if all(v and v["interval"][0]>0 for v in evidence) else "inconclusive"))
    write_receipt(folder/("confirmation.json" if audit else "development.json"),dict(result=result,stats=stats))
    print(json.dumps(stats))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected audit or no argument")
    main(bool(sys.argv[1:]))
