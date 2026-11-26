"""Evaluate one-ply response risks and paired team results on unused replay families."""
from collections import Counter
import hashlib
from itertools import product
import json
import os
import sys

from serial_archive_pressure import SOURCE, archives
from serial_formation_matrix import contrast, score_difference
from serial_route_choice import ROOT, build, compare_replays, native, require, write_receipt

PER_FORMAT = 24
KEYS = ("ply", "seat", "unassisted", "prior_sector", "base_sector", "alternative_sector", "base", "alternative",
        "human", "risk", "alternative_risk", "human_risk", "reply", "alternative_reply", "earlier_risk", "attacker", "target")
PRIOR = ROOT/"build/full-route-choice/audit-full-mode-defense-team-value-6cca9328583b68e11cd4/development.json"


def candidate_replays():
    earlier,_ = archives()
    used = {r["family"] for r in earlier.values()}
    census = json.loads((SOURCE/"probe.json").read_bytes())["rows"]
    unique = {}
    for r in sorted(census,key=lambda r:(r["format"]!=".jqh",-r["verified_plies"],r["key"])):
        if r["family"] in used or r["prior_layout"] or r["prior_trajectory"] or r["direction"] not in (0,1):
            continue
        cap = min(r["verified_plies"],128,r["notice"]-1 if r["notice"] else 128)
        if cap>=72:
            unique.setdefault(r["family"],dict(key=r["key"],family=r["family"],format=r["format"],ccw=r["direction"],cap=cap))
    selected = {}
    for suffix in (".jqh",".jgs"):
        group = sorted((r for r in unique.values() if r["format"]==suffix),
                       key=lambda r:hashlib.sha256(("natural-routes:"+r["family"]).encode()).digest())
        require(len(group)>=PER_FORMAT*2,"insufficient new replay families")
        for index,row in enumerate(group[:PER_FORMAT*2]):
            selected[row["key"]] = dict(row,audit=bool(index%2))
    return selected


def route_wire(selected):
    wires = {}
    with (SOURCE/"games.txt").open(encoding="utf8") as source:
        for line in source:
            tag,key,tape = line.split(maxsplit=2)
            if key not in selected:
                continue
            row = selected[key]
            require(tag==("H" if row["format"]==".jqh" else "G"),"route replay format mismatch")
            seed = int(key[:12],16)+920000
            wires[key] = f'{tag} {key} {row["ccw"]} {row["cap"]} {seed} {tape.rstrip()}\n'
    require(wires.keys()==selected.keys(),"replay wire does not match declared families")
    return "".join(wires[key] for key in selected)


def collect(directory, audit):
    selected = {key:row for key,row in candidate_replays().items() if row["audit"]==audit}
    raw = native(directory,"natural-routes",route_wire(selected))
    require(raw[0]==["C","direct-team-win"],"direct counterattack control did not run")
    coverage, roots, games = {}, {}, []
    for tag,key,*fields in raw[1:]:
        require(key in selected,"unknown replay family in native output")
        if tag=="A":
            agreed,first,reason,inspected,exposed,paired = fields
            agreed,first = int(agreed),int(first)
            require(0<=agreed<=selected[key]["cap"] and (first==0 if reason=="matched" else first==agreed+1),
                    "invalid strictly verified replay boundary")
            coverage[key] = dict(agreed=agreed,first=first,reason=reason,inspected=int(inspected),exposed=int(exposed),paired=int(paired))
        elif tag=="P":
            root = dict(zip(KEYS,map(int,fields),strict=True))
            require(key not in roots and 0<=root["alternative_risk"]<root["risk"]<=3 and root["reply"]>=0,
                    "invalid paired route exposure")
            require(0<=root["earlier_risk"]<root["risk"],"opponent move did not increase exposure in local comparison")
            roots[key] = root
        else:
            require(tag=="G","unexpected natural route output")
            games.append([key,*map(int,fields)])
    require(coverage.keys()==selected.keys() and set(roots)=={key for key,v in coverage.items() if v["paired"]==1},
            "unaccounted source or replay decision")
    expected = []
    for key,root in roots.items():
        require(root["ply"]<=coverage[key]["agreed"],"route cut after first profile mismatch")
        actions=(root["base"],root["alternative"],root["human"])
        expected.extend([key,root["ply"],arm,table,stream,actions[arm]]
                        for arm,table,stream in product(range(3),range(2),range(4)))
    require([r[:6] for r in games]==expected and all(r[6] in (-1,0,1,2) and r[7]>=r[1] for r in games),
            "missing or malformed paired strategy outcomes")
    return dict(sources=selected,coverage=coverage,roots=roots,games=games)


def summarize(data):
    coverage,roots = data["coverage"],data["roots"]
    tables = {}
    for table in (0,1):
        outcomes = {arm:Counter() for arm in (0,1,2)}
        trials = {key:{} for key in roots}
        for key,cut,arm,profile,stream,action,score,ply,flag,_ in data["games"]:
            if table!=profile:
                continue
            trials[key][arm,stream] = score
            outcomes[arm][{0:"loss",1:"draw",2:"win",-1:"unknown"}[score]] += 1
            outcomes[arm]["win140"] += score==2 and ply<=140
            outcomes[arm]["loss140"] += score==0 and ply<=140
            outcomes[arm]["flag"] += flag>0
        improvements = [tuple(sum(col)/4 for col in zip(*(score_difference(scores[0,s],scores[1,s])
                        for s in range(4)),strict=True)) for scores in trials.values()]
        tables[table] = dict(counts=outcomes,delta=contrast(improvements) if improvements else None)
    return dict(families=len(coverage),plies=sum(v["agreed"] for v in coverage.values()),paired=len(roots),
                inspected=sum(v["inspected"] for v in coverage.values()),exposed=sum(v["exposed"] for v in coverage.values()),
                reasons=dict(Counter(v["reason"] for v in coverage.values())),
                risk=dict(Counter(r["risk"] for r in roots.values())),
                exposed_pieces=dict(Counter(r["attacker"] for r in roots.values())),
                same_sector=sum(r["base_sector"]>=0 and r["base_sector"]==r["alternative_sector"] for r in roots.values()),
                unassisted=sum(r["unassisted"] for r in roots.values()),tables=tables)


def main(audit):
    folder = ROOT/"build/natural-routes"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        folder.mkdir(parents=True,exist_ok=True)
        build(folder)
    result = collect(folder,audit)
    stats = summarize(result)
    if audit:
        dev = json.loads((folder/"development.json").read_bytes())
        require(collect(folder,False)==dev["result"] and set(result["sources"]).isdisjoint(dev["result"]["sources"]),
                "prior evidence did not reproduce or partitions overlap")
        require(compare_replays(folder,False)==json.loads(PRIOR.read_bytes())["data"],"prior simulator interface changed outcomes")
        prior = dev["stats"]["tables"]
        supported = all(t["delta"] is not None and t["delta"]["interval"][0]>0
                        for t in [*prior.values(),*stats["tables"].values()])
        print("JUNQI_RESEARCH "+("supported" if supported else "inconclusive"))
    write_receipt(folder/("confirmation.json" if audit else "development.json"),dict(result=result,stats=stats))
    print(json.dumps(stats))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected audit or no argument")
    main(bool(sys.argv[1:]))
