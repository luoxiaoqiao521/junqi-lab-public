"""Export observable decision roots and their matched natural-game outcome labels."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
import subprocess
import sys

from audit_deadline_long_draw import TAIL, SOURCE as COMBINED
from measure_deadline_opening import COLUMNS, ROOT, RULES, build, model_roster, require
from measure_deadline_rollout import TRACES, WORK


EVIDENCE=ROOT/"build/deadline-opening/audit-deadline-win-to-draw-785168f53ce9efc59109/summary.json"
NODES=129
OUTCOME={-1:None,0:"loss",1:"draw",2:"win"}


def identities(view,actor):
    require(all(-1<=owner<4 and -1<=rank<12 and 0<=ranks<4096 and moved in (0,1) and
                ((owner==-1 and rank==-1 and ranks==0 and not moved) or
                 (owner==actor and rank>=0 and ranks==(1<<rank)) or (owner>=0 and owner!=actor and rank==-1 and ranks>0))
                for owner,rank,ranks,moved in view), "hidden truth or invalid public rank mask in export")


def by_deadline(row,horizon,outcome):
    if row["outcome"]==-1 and row["plies"]<horizon:
        return None
    return row["outcome"]==outcome and row["plies"]<=horizon


def action_label(row,turn):
    remainder=row["plies"]-turn
    require(remainder>0,"label precedes decision")
    return {"final":OUTCOME[row["outcome"]], "observed_remaining":remainder,
            "censored":row["outcome"]==-1,"win_100":by_deadline(row,100,2),
            "win_140":by_deadline(row,140,2),"loss_140":by_deadline(row,140,0)}


def index(row):
    return row["seed"],row["seat"],row["table"]


def compile_pairs(stdout,reference,first_seed):
    arms={a:{index(r):r for r in reference if r["arm"]==a} for a in (11,13,14)}
    data=[]
    groups=defaultdict(set)
    snapshots=set()
    replays=set()
    for line in stdout.splitlines():
        if line.startswith("P "):
            seed,seat,table,trace,ply,actor,clock,*tail=map(int,line[2:].split())
            key=(seed,seat,table)
            baseline,proposal,first_only=(arms[a][key] for a in (11,13,14))
            require(key not in snapshots and trace==proposal["offer_trace"]==first_only["offer_trace"] and
                    ply+1==proposal["offered_at"] and actor==proposal["actor"] and 0<=clock<=69,"wrong matched root")
            snapshots.add(key)
            require(len(tail)>4*NODES,"incomplete public board or legal actions")
            view=[tail[i:i+4] for i in range(0,4*NODES,4)]
            identities(view,actor)
            legal=tail[4*NODES:]
            moves=(proposal["original_move"],proposal["proposed_move"])
            require(len(set(legal))==len(legal) and 0<=min(legal)<=max(legal)<NODES*NODES and
                    moves[0]!=moves[1] and all(m in legal for m in moves),"action label is not legal")
            # Source/layout fields remain envelope metadata, never features.
            partition=("developer_validation" if (seed-first_seed)%4==0 else "developer_train")
            groups[partition].add(seed)
            data.append({"version":"junqi-public-paired-1","family":seed,"example":[seat,table],
                         "partition":partition,"observation":{"actor":actor, "ply":ply,"static_plies":clock,
                            "left_100":max(0,100-ply),"left_140":max(0,140-ply),"board":view,"legal":legal},
                         "labels":[[move,action_label(row,ply)] for move,row in zip(moves,(baseline,first_only),strict=True)]})
        else:
            row=dict(zip((*COLUMNS,*WORK,*TRACES,*TAIL),map(int,line.split()),strict=True))
            key=index(row)
            require(key not in replays and arms[13][key]==row,"replayed combination changed the tested evidence")
            replays.add(key)
    require(replays==set(arms[13]) and snapshots=={k for k,r in arms[13].items() if r["offered_at"]},
            "incomplete learning export")
    require(len(groups)==2 and not set.intersection(*groups.values()),"layout crossed an internal partition")
    return data,groups


def calculate(directory):
    source=json.loads(COMBINED.read_text(encoding="utf8"))
    reference=json.loads(EVIDENCE.read_text(encoding="utf8"))["rows"]
    roster=model_roster()
    weights=[weight for name in ("candidate","historical") for weight in roster[name]["weights"]]
    stdin=" ".join(map(str,(source["seed"],source["count"],*weights)))
    proc=subprocess.run([directory/"deadline_opening.exe",RULES,"learning-roots"],input=stdin,capture_output=True,
                        text=True,encoding="utf8",errors="replace",timeout=240)
    require(proc.returncode==0,f"learning root replay failed: {proc.stderr.strip()}")
    data,groups=compile_pairs(proc.stdout,reference,source["seed"])
    wire="".join(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n" for row in data)
    final_labels=[label for row in data for _,label in row["labels"]]
    receipt={"examples":len(data),"labels":len(final_labels),"sha256":sha256(wire.encode()).hexdigest(),
             "partitions":{k:{"families":len(v),"examples":sum(r["partition"]==k for r in data)} for k,v in groups.items()},
             "natural_results":dict(Counter(label["final"] or "unresolved" for label in final_labels)),
             "source":str(EVIDENCE.relative_to(ROOT)),"continuation":"fixed continued pressure after the first move"}
    return wire,receipt


def main(audit):
    censored={"outcome":-1,"plies":120}
    require(by_deadline(censored,100,2) is False and by_deadline(censored,140,2) is None,
            "early censoring must not label a missed future win")
    directory=ROOT/"build/deadline-opening"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        build(directory)
    wire,receipt=calculate(directory)
    for name,content in (("paired-public.jsonl",wire),("summary.json",json.dumps(receipt,sort_keys=True,indent=2)+"\n")):
        target=directory/name
        if audit:
            require(target.read_bytes()==content.encode(),f"changed reproduced file: {name}")
        else:
            pending=target.with_suffix(".tmp")
            pending.write_bytes(content.encode())
            pending.replace(target)
    print(json.dumps(receipt,sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    if sys.argv[1:] not in ([],["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(sys.argv[1:]==["audit"])
