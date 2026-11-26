"""Measure paired public learning labels against two seats of active pressure."""
from collections import Counter
from hashlib import sha256
import json
import os
import subprocess
import sys

from audit_deadline_long_draw import TAIL
from export_deadline_learning_pairs import NODES, action_label, identities, index
from measure_deadline_opening import COLUMNS, ROOT, RULES, build, model_roster, require
from measure_deadline_rollout import TRACES, WORK

SEED, COUNT = 1091300, 32
MATCH = ("offered_at", "actor", "original_move", "proposed_move", "offer_trace")


def decode_root(record, original, alternative, partition):
    seed, seat, table, trace, ply, actor, clock, *tail = record
    require(trace==original["offer_trace"] and ply+1==original["offered_at"] and actor==seat
            and 0<=clock<=69 and len(tail)>4*NODES, "invalid or mismatched public root")
    view=[tail[i:i+4] for i in range(0,4*NODES,4)]
    identities(view,actor)
    legal=tail[4*NODES:]
    moves=(original["original_move"],original["proposed_move"])
    require(len(set(legal))==len(legal) and 0<=min(legal)<=max(legal)<NODES*NODES and
            moves[0]!=moves[1] and all(move in legal for move in moves), "illegally labeled move")
    return {
        "version":"junqi-public-paired-1", "family":seed, "example":[seat,table], "partition":partition,
        "observation":{"actor":actor,"ply":ply,"static_plies":clock,"left_100":max(0,100-ply),
                       "left_140":max(0,140-ply),"board":view,"legal":legal},
        "labels":[[move,action_label(row,ply)] for move,row in zip(moves,(original,alternative),strict=True)],
        "diagnostics":[{k:r[k] for k in ("first_result","partner_offset","partner_contact","contacts_next12",
                                        "friend_exits_next12","enemy_exits_next12")} for r in (original,alternative)]}


def calculate(directory):
    roster=model_roster()
    weights=[w for name in ("candidate","historical") for w in roster[name]["weights"]]
    run=subprocess.run([directory/"deadline_opening.exe",RULES,"learning-hard-outcomes"],
                       input=" ".join(map(str,(SEED,COUNT,*weights))),capture_output=True,text=True,
                       encoding="utf8",errors="replace",timeout=240)
    require(run.returncode==0,f"pressure branch failed: {run.stderr.strip()}")
    roots={}
    arms={14:{},21:{}}
    for line in run.stdout.splitlines():
        values=list(map(int,(line[2:] if line.startswith("P ") else line).split()))
        if line.startswith("P "):
            key=tuple(values[:3])
            require(key not in roots,"duplicate public root")
            roots[key]=values
        else:
            row=dict(zip((*COLUMNS,*WORK,*TRACES,*TAIL,"partner"),values,strict=True))
            key=index(row)
            require(key not in arms[row["arm"]] and row["partner"]==1,"duplicate game or wrong partner")
            arms[row["arm"]][key]=row
    grid={(s,j,t) for s in range(SEED,SEED+COUNT) for j in range(4) for t in range(2)}
    require(set(arms[21])==set(arms[14])==grid,"incomplete pressure matrix")
    require(set(roots)=={key for key,r in arms[14].items() if r["offered_at"]},"missing or extra public root")
    data=[]
    checks={}
    for key in sorted(grid):
        original,alternative=(arms[a][key] for a in (21,14))
        require(all(original[k]==alternative[k] for k in MATCH) and original["applied"]==0
                and alternative["applied"]==bool(original["offered_at"]),"more than one decision changed")
        if key not in roots:
            require(original["event_trace"]==alternative["event_trace"],"changed an unoffered game")
            continue
        partition="developer_validation" if (key[0]-SEED)%4==0 else "developer_train"
        data.append(decode_root(roots[key],original,alternative,partition))
    for partition in ("developer_train","developer_validation"):
        selected=[r for r in data if r["partition"]==partition]
        labels=[r["labels"] for r in selected]
        checks[partition]={
            "families":len({r["family"] for r in selected}),"pairs":len(selected),
            "final":dict(Counter(label["final"] or "unresolved" for pair in labels for _,label in pair)),
            "loss_families":len({r["family"] for r in selected if any(l["final"]=="loss" for _,l in r["labels"])}),
            "win_to_loss":sum(a["final"]=="win" and b["final"]=="loss" for (_,a),(_,b) in labels),
            "loss_to_win":sum(a["final"]=="loss" and b["final"]=="win" for (_,a),(_,b) in labels),
            "opposite_final":sum(a["final"]!=b["final"] for (_,a),(_,b) in labels),
            "different_140_win":sum(a["win_140"]!=b["win_140"] for (_,a),(_,b) in labels),
            "partner_first_reply_without_contact":sum(d["partner_offset"]>0 and not d["partner_contact"]
                for r in selected for d in r["diagnostics"])}
    wire="".join(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n" for row in data)
    summary={"pairs":len(data),"games_per_arm":len(grid),"partitions":checks,"sha256":sha256(wire.encode()).hexdigest(),
             "source":"fresh simulated layouts; two pressure opponents and a fixed historical partner",
             "treatment":"one candidate applied or skipped; all subsequent policies match"}
    return wire,summary


def main(audit):
    directory=ROOT/"build/deadline-opening"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        build(directory)
    wire,summary=calculate(directory)
    for filename,content in (("paired-public.jsonl",wire),("summary.json",json.dumps(summary,sort_keys=True,indent=2)+"\n")):
        target=directory/filename
        if audit:
            require(target.read_bytes()==content.encode(),f"changed pressure evidence: {filename}")
        else:
            pending=target.with_suffix(".tmp")
            pending.write_bytes(content.encode())
            pending.replace(target)
    print(json.dumps(summary,sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    if sys.argv[1:] not in ([],["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(sys.argv[1:]==["audit"])
