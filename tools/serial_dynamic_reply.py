"""Audit frozen student rankings, then compare bounded takeover schedules in actual games."""
from collections import Counter
from datetime import datetime
import hashlib
from itertools import product
import json
import os
import sys
from time import perf_counter

from serial_counterreply import manifest, sources
from serial_evaluator_panel import ROOT, build, model_roster, native, require
from serial_formation_matrix import contrast, score_difference
from serial_reply_window import model_wire
from serial_worker import read, write_receipt

PRIOR = ROOT/"build/reply-window/audit-action-conditioned-reply-learning-9b9ca437ef846ea08211"
ARMS = ("path","fixed","dynamic","fixed_hybrid","dynamic_hybrid")
ROOT_KEYS = ("cut seat hash base witness s44 s52 wide52 all52 extension n4 n8 legal").split()
GAME_KEYS = ("seed table arm plies result scanned used late changed searches search_steps feature_steps "
             "micros first hash old new final_hash extended override_student").split()


def frozen_wire():
    frozen=read(PRIOR/"students.json")
    weights=model_roster()["historical"]
    wire=" ".join(map(str,(*weights["weights"],0)))
    wire+=" "+model_wire(frozen["models"]["board"],"board")
    wire+=" "+model_wire(frozen["models"]["window"],"window")
    return wire,weights["artifact_sha256"]


def audit_manifest():
    old=read(PRIOR/"games-confirmation.json")["rows"]
    seeded={}
    for seed in (2767006,2767014):
        arm,=tuple(r for r in old if r["seed"]==seed and r["table"]==1 and r["arm"]==2)
        seeded[str(seed)]=dict(hash=arm["root_hash"],base=arm["old_move"],witness=arm["new_move"],
                              cut=arm["first_change"],seed=seed)
    rows="".join(f"S {key} 1 {x['cut']} {x['seed']} {x['base']} {x['witness']}\n" for key,x in seeded.items())
    archived=sources(False)
    return seeded,archived,rows+manifest(archived)


def audit_candidates(folder,wire):
    seeded,archived,rows=audit_manifest()
    expected=set(seeded)|set(archived)
    raw=native(folder,"frozen-reply-audit",wire+"\n"+rows)
    data={}
    for tag,key,*fields in map(str.split,raw.splitlines()):
        require(key in expected,"unexpected decision source")
        if tag=="R":
            root=dict(zip(ROOT_KEYS,map(int,fields),strict=True))
            require(key not in data and root["n4"]<=root["n8"]<=root["legal"],"invalid frozen root")
            if key in seeded:
                require(all(root[k]==seeded[key][k] for k in ("hash","base","witness","cut")) and
                        root["witness"]==root["s44"],"the real losing fork did not restore")
            data[key]=dict(root=root,rank=[],games={})
        elif tag=="S":
            i,action,board,window=fields
            data[key]["rank"].append(dict(order=int(i),action=int(action),board=float(board),window=float(window)))
        else:
            require(tag=="G","invalid audit output")
            cut,index,table,stream,action,result,plies,flag,digest=map(int,fields)
            require(cut==data[key]["root"]["cut"] and result in (-1,0,1,2),"invalid audit rollout")
            name=f"{action}/{table}/{stream}"
            require(name not in data[key]["games"],"duplicate continuation")
            data[key]["games"][name]=dict(outcome=result,plies=plies,final_flag=flag)
    require(data.keys()==expected,"missing frozen decision")
    for item in data.values():
        r,rows,games=item["root"],item["rank"],item["games"]
        require([a["order"] for a in rows]==list(range(r["legal"])) and rows[0]["action"]==r["base"],"incomplete ranking")
        by_action={a["action"]:a for a in rows}
        require(len(by_action)==r["legal"],"duplicate candidate")
        require(set(games)=={f"{a}/{t}/{s}" for a,t,s in product(
            {r[k] for k in ("base","witness","s44","s52","wide52","all52","extension")},range(2),range(8))},
            "incomplete action-by-stream matrix")
        known=by_action[r["witness"]]
        item["witness_rank"]=dict(path=known["order"]+1,board=sum(a["board"]>known["board"]+1e-12 for a in rows)+1,
                                  window=sum(a["window"]>known["window"]+1e-12 for a in rows)+1,
                                  score_gap=by_action[r["s52"]]["window"]-known["window"])
        comparisons={}
        for name in ("witness","s44","s52","wide52","all52","extension"):
            for table in range(2):
                a,b=r["base"],r[name]
                pair=[score_difference(games[f"{a}/{table}/{s}"]["outcome"],games[f"{b}/{table}/{s}"]["outcome"]) if a!=b
                      else (0.,0.) for s in range(8)]
                comparisons[f"{name}/{table}"]=dict(
                    points=[sum(values)/8 for values in zip(*pair,strict=True)],
                    results=[games[f"{b}/{table}/{s}"]["outcome"] for s in range(8)])
        item["comparison"]=comparisons
    return data


def scheduled_games(folder,wire,audit,coverage=False):
    if coverage:
        start=2817800 if audit else 2817400
        ids=(0,1,5,6)
        arms=("path","fixed","wide","wide_if_close")
        keys=GAME_KEYS+["widened","tail_selected"]
        mode=3
    else:
        start=2813400 if audit else 2813000
        ids=tuple(range(len(ARMS)))
        arms=ARMS
        keys=GAME_KEYS
        mode=0
    count=24
    raw=native(folder,"scheduled-reply-games",f"{start} {count} {mode} "+wire)
    rows=[]
    for tag,*fields in map(str.split,raw.splitlines()):
        require(tag=="G","misplaced scheduled game output")
        rows.append(dict(zip(keys,map(int,fields),strict=True)))
    require([(r["seed"],r["table"],r["arm"]) for r in rows]==[
        (s,t,a) for s,t,a in product(range(start,start+count),range(2),ids)],"invalid matched opening grid")
    grouped={}
    for row in rows:
        require(0<=row["changed"]<=row["used"]<=2 and row["late"]<=row["used"]<=row["scanned"] and
                row["search_steps"]<=96*row["searches"] and row["searches"]<=row["used"],"invalid intervention cost")
        key=(row["seed"],row["table"])
        grouped.setdefault(key,[]).append(row)
    for group in grouped.values():
        for r in group:
            if r["changed"]==0:
                require(r["final_hash"]==group[0]["final_hash"],"no-change pairing diverged")
        if coverage:
            for r in group[2:]:
                require(0<=r["tail_selected"]<=r["widened"]<=r["used"],"invalid selective widening trace")
                if not r["tail_selected"]:
                    require(r["final_hash"]==group[1]["final_hash"],"no outside selection drifted from four candidates")
    report={}
    for table in range(2):
        groups=[v for (s,t),v in grouped.items() if t==table]
        for index,name in enumerate(arms):
            counts=Counter()
            chosen=[g[index] for g in groups]
            for r in chosen:
                counts[{-1:"unresolved",0:"loss",1:"draw",2:"win"}[r["result"]]]+=1
                counts["win100"]+=r["result"]==2 and r["plies"]<=100
                counts["win140"]+=r["result"]==2 and r["plies"]<=140
                counts["loss140"]+=r["result"]==0 and r["plies"]<=140
                fields=("scanned","used","late","changed","searches","micros","search_steps","feature_steps",
                        "extended","override_student")
                if coverage:
                    fields+=("widened","tail_selected")
                for field in fields:
                    counts[field]+=r[field]
            details={}
            for base in range(index):
                pairs=[score_difference(g[base]["result"],g[index]["result"]) if g[base]["final_hash"]!=g[index]["final_hash"]
                       else (0.,0.) for g in groups]
                details[arms[base]]=dict(**contrast(pairs),better=sum(lo>0 for lo,_ in pairs),worse=sum(hi<0 for _,hi in pairs))
            report[f"{table}/{name}"]=dict(counts=counts,comparisons=details,mean_plies=sum(r["plies"] for r in chosen)/count)
    return dict(rows=rows,summary=report)


def main(audit):
    began=perf_counter()
    key=os.environ["JUNQI_SERIAL_KEY"]
    out=ROOT/"build/dynamic-reply"/key
    wire,peer=frozen_wire()
    model=hashlib.sha256((PRIOR/"students.json").read_bytes()).hexdigest()
    if not audit:
        out.mkdir(parents=True,exist_ok=True)
        build(out)
        probe=audit_candidates(out,wire)
        write_receipt(out/"frozen-ranks.json",probe)
        first_verified=round(perf_counter()-began,3)
        kickoff=read(ROOT/"ops/serial.json")["research"]["observed_kickoff"]
        wall_to_first=round((datetime.now().astimezone()-datetime.fromisoformat(kickoff)).total_seconds(),3)
        print(json.dumps(dict(frozen_roots=len(probe),matched_continuations=sum(len(i["games"]) for i in probe.values()),
                              first_verified_seconds=first_verified,wall_to_first_seconds=wall_to_first)))
    result=scheduled_games(out,wire,audit)
    result.update(model=model,peer=peer,native_stage_seconds=round(perf_counter()-began,3))
    if audit:
        old=read(out/"development.json")
        require(old["model"]==model and old["peer"]==peer and
                {r["seed"] for r in old["rows"]}.isdisjoint(r["seed"] for r in result["rows"]),"opening holdout drift")
        print("JUNQI_RESEARCH inconclusive")
    else:
        result["first_verified_seconds"]=first_verified
        result["wall_to_first_seconds"]=wall_to_first
    path=out/("confirmation.json" if audit else "development.json")
    write_receipt(path,result)
    print(json.dumps(result["summary"]))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
