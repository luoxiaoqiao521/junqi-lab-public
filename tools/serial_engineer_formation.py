"""Measure an opening engineer move opportunity with a frozen whole-game student."""
from collections import Counter
import hashlib
import json
import os
from statistics import mean, median
import sys

from serial_counterreply import sources as counter_sources
from serial_dynamic_reply import PRIOR, ROOT, build, native, require
from serial_formation_matrix import contrast, score_difference
from serial_full_student_evolution import GAME_KEYS, digest
from serial_natural_routes import SOURCE, candidate_replays, route_wire
from serial_relay_setup import archives, sources as relay_sources
from serial_reply_window import model_wire
from serial_worker import read, write_receipt

START = (2827100,2827400)
WINDOW = 32


def human_partitions(size=WINDOW):
    previous = (*archives()[0].values(),*candidate_replays().values(),*relay_sources(False).values(),
                *relay_sources(True).values(),*counter_sources(False).values(),*counter_sources(True).values())
    used = {r["family"] for r in previous}
    unique = {}
    for r in sorted(read(SOURCE/"probe.json")["rows"],key=lambda x:(-x["verified_plies"],x["key"])):
        if (r["family"] not in used and not r["prior_layout"] and not r["prior_trajectory"] and
                r["direction"] in (0,1) and r["verified_plies"]>=72 and (not r["notice"] or r["notice"]>72)):
            unique.setdefault(r["family"],dict(key=r["key"],family=r["family"],format=r["format"],ccw=r["direction"],cap=72))
    ordered = sorted(unique.values(),key=lambda r:hashlib.sha256(("engineer-formation:"+r["family"]).encode()).digest())
    require(len(ordered)>=2*size,"insufficient distinct source families for this split")
    return ({r["key"]:r for r in ordered[i:2*size:2]} for i in range(2))


def summarize(blocks):
    summaries = {}
    for name, group in (("overall",blocks),("generated",[b for b in blocks if b["source"]==0]),
                        ("replay",[b for b in blocks if b["source"]==1])):
        effects = [[],[]]
        interaction = []
        for b in group:
            differences=[score_difference(b["games"][a]["result"], b["games"][a+1]["result"]) for a in (0,2)]
            for j,d in enumerate(differences):
                effects[j].append(d)
            (a,bound_a),(c,d)=differences
            interaction.append((c-bound_a,d-a))
        cells = []
        for a in range(4):
            games = [b["games"][a] for b in group]
            exits = [g["engineer_exit"] for g in games if g["engineer_exit"]]
            cells.append(dict(
                outcomes=dict(Counter(str(g["result"]) for g in games)),
                mean_plies=mean(g["plies"] for g in games),
                wins100=sum(g["result"]==2 and g["plies"]<=100 for g in games),
                wins140=sum(g["result"]==2 and g["plies"]<=140 for g in games),
                losses140=sum(g["result"]==0 and g["plies"]<=140 for g in games),
                engineer_actions=sum(g["engineer_move"]>0 for g in games),
                engineer_exits=len(exits),median_first_exit=median(exits) if exits else None,
                collision_engineer_lost_before_exit=sum(g["engineer_loss"]>0 and
                    (not g["engineer_exit"] or g["engineer_loss"]<g["engineer_exit"]) for g in games),
                first_exits=dict(Counter(f'{g["exit_seat"]}/{g["exit_reason"]}' for g in games)),
                **{k:sum(g[k] for g in games) for k in ("turns","used","changed","micros","support_open","support_used")}))
        summaries[name]=dict(count=len(group),cells=cells,route_shift=contrast(effects[0]),
                             student_shift=contrast(effects[1]),interaction=contrast(interaction))
    return summaries


def formation_payload(student, audit):
    selected = tuple(human_partitions())[audit]
    generated = {str(s):dict(source=0,seed=s) for s in range(START[audit],START[audit]+WINDOW)}
    inputs = dict(generated)
    manifest = [f"R {key} {r['seed']}" for key,r in generated.items()]
    for row in route_wire(selected).splitlines():
        tag,key,_,_,seed,tape=row.split(maxsplit=5)
        manifest.append(f"{tag} {key} {seed} {tape}")
        inputs[key]=dict(selected[key],source=1,seed=int(seed))
    payload = model_wire(student,"window")+"\n"+"\n".join(manifest)
    return inputs,payload


def collect(folder, student, audit):
    inputs,payload = formation_payload(student,audit)
    raw = native(folder,"engineer-student-formation",payload)
    tried = {}
    accepted = {}
    for tag,key,*parts in map(str.split,raw.splitlines()):
        require(key in inputs,"unexpected opening source")
        if tag=="G":
            row=dict(zip(GAME_KEYS[1:],map(int,parts),strict=True))
            episode=accepted[key]
            require(row["arm"]==len(episode["games"]) and row["table"]==episode["table"] and
                    row["result"] in (-1,0,1,2) and row["used"]==(row["turns"] if row["arm"]>=2 else 0) and
                    0<=row["changed"]<=row["used"],"invalid full-opening crossover")
            episode["games"].append(row)
        else:
            require(key not in tried and tag in ("A","M","S") and int(parts[0])==inputs[key]["source"],
                    "opening failed integrity or duplication check")
            tried[key]=dict(status=tag,fields=list(map(int,parts)))
            if tag=="A":
                block=dict(zip(("source","learner","first","table","from","to","rank","old_exits","new_exits",
                                "original_hash","shifted_hash"),map(int,parts),strict=True))
                require(block["new_exits"]>block["old_exits"] and block["original_hash"]!=block["shifted_hash"],
                        "movement opportunity or state did not change")
                accepted[key]=dict(inputs[key],**{k:v for k,v in block.items() if k!="source"},games=[])
    require(tried.keys()==inputs.keys() and Counter(b["source"] for b in accepted.values())=={0:8,1:8} and
            all(len(a["games"])==4 for a in accepted.values()),"incomplete four-cell grid")
    require(len({b[k] for b in accepted.values() for k in ("original_hash","shifted_hash")})==32,
            "duplicate complete opening state")
    return dict(accepted=accepted,intake=tried,manifest_sha256=digest(payload),summary=summarize(list(accepted.values())))


def main(audit):
    folder = ROOT/"build/engineer-formation"/os.environ["JUNQI_SERIAL_KEY"]
    student = read(PRIOR/"students.json")["models"]["window"]
    model = digest(model_wire(student,"window"))
    if not audit:
        folder.mkdir(parents=True,exist_ok=True)
        build(folder)
    result = collect(folder,student,audit)
    result["student_sha256"] = model
    if audit:
        development = read(folder/"development.json")
        a,b = development["accepted"],result["accepted"]
        require(set(a).isdisjoint(b) and development["student_sha256"]==model and
                {r.get("family") for r in a.values() if r["source"]}.isdisjoint(
                    r.get("family") for r in b.values() if r["source"]), "confirmation reused development opening families")
        intervals=[r["summary"]["overall"]["interaction"]["interval"] for r in (development,result)]
        outcome = "supported" if all(lo>0 for lo,_ in intervals) else (
            "refuted" if all(hi<0 for _,hi in intervals) else "inconclusive")
        print("JUNQI_RESEARCH "+outcome)
    write_receipt(folder/("confirmation.json" if audit else "development.json"),result)
    print(json.dumps(result["summary"]))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
