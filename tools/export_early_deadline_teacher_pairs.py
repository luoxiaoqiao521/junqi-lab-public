"""Export bounded candidate comparisons at two predeclared early-game windows."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
import subprocess
import sys

from export_deadline_learning_pairs import NODES, action_label, identities
from measure_deadline_opening import ROOT, RULES, build, model_roster, require


SEED, COUNT = 1092700, 24
ACTION_FIELDS = ("ply", "move", "plies", "outcome", "end_reason", "terminal_trace", "next_actor",
                 "partner_turn", "partner_contact", "enemy_turn", "enemy_contact",
                 "friendly_exit", "friendly_reason", "enemy_exit", "enemy_reason")
POLICIES = ("pressure", "bounded_reply", "joint_routes")


def parse(stdout, start=SEED, count=COUNT):
    roots, teachers, games = {}, {}, {}
    branches = defaultdict(dict)
    for line in stdout.splitlines():
        tag, *columns = line.split()
        seed, actor, context, *values = map(int, columns)
        key = (seed, actor, context)
        if tag == "P":
            require(key not in roots, "duplicate public root")
            roots[key] = values
        elif tag == "T":
            require(key not in teachers and len(values)==5, "invalid teacher selections")
            teachers[key] = values
        elif tag == "A":
            prefix, *payload = values
            row = dict(zip(ACTION_FIELDS, payload, strict=True)) | {"root_trace":prefix}
            require(row["move"] not in branches[key], "duplicate branch")
            branches[key][row["move"]] = row
        else:
            require(tag=="G" and key not in games and len(values)==2 and set(values)<={0,1}, "bad opening ledger")
            games[key] = values
    grid = {(s,j,t) for s in range(start,start+count) for j in range(4) for t in (0,1)}
    planned = {(s,j,2*t+w) for (s,j,t), flags in games.items() for w,flag in enumerate(flags) if flag}
    require(set(games)==grid and planned==set(roots)==set(teachers)==set(branches), "missing roots or opening ledger")
    return roots, teachers, branches, len(games)


def public_record(key, root, selections, branches, start=SEED):
    seed, learner, context = key
    trace, ply, actor, clock, *nodes = root
    boundary, close = (24,56) if context%2==0 else (56,100)
    require(boundary<=ply<close and actor==learner and 0<=clock<70 and len(nodes)>4*NODES, "wrong public state")
    view = [nodes[i:i+4] for i in range(0,4*NODES,4)]
    legal = nodes[4*NODES:]
    identities(view,actor)
    *choices, samples, count = selections
    require(len(legal)==len(set(legal)) and all(0<=m<NODES*NODES for m in legal)
            and len(branches)==count and 1<=count<=6 and set(choices)<=branches.keys()<=set(legal)
            and samples in (0,4*count), "nonlegal selection or changed candidate budget")
    require(all(r["root_trace"]==trace and r["ply"]==ply for r in branches.values()), "different branch roots")
    labels = {m: action_label(r,ply) for m,r in branches.items()}
    return {
        "version":"junqi-early-deadline-outcomes-1", "family":seed, "actor_seat":actor,
        "phase":context%2, "table":("historical","active_pressure")[context//2],
        "partition":"development_holdout" if (seed-start)%4==0 else "development_train",
        "observation":{"actor":actor,"ply":ply,"static_plies":clock,"left_100":100-ply,
                       "left_140":140-ply,"board":view,"legal":legal},
        "choices":dict(zip(POLICIES,choices,strict=True)),
        "branches":{m:{"outcome":labels[m], "replay":r} for m,r in branches.items()}
    }


def describe(cases):
    panel = {}
    for table in ("historical","active_pressure"):
        local = [r for r in cases if r["table"]==table]
        table_stats = {"roots":len(local), "candidates":sum(len(r["branches"]) for r in local)}
        for p in POLICIES:
            cnt = Counter()
            for r in local:
                selected=r["branches"][r["choices"][p]]
                row, outcome=selected["replay"],selected["outcome"]
                cnt["different_first"]+=r["choices"][p]!=r["choices"]["pressure"]
                cnt["flag_first_then_loss_by_140"]+=bool(outcome["loss_140"] and row["friendly_reason"]==1
                    and row["friendly_exit"]>0 and (row["enemy_exit"]==0 or row["friendly_exit"]<row["enemy_exit"]))
                cnt["any_early_loss"]+=outcome["loss_140"] is True
                cnt["win_100"]+=outcome["win_100"] is True
                cnt["win_140"]+=outcome["win_140"] is True
                cnt[outcome["final"] or "unresolved"]+=1
                cnt["first_enemy_by_100"]+=0<row["enemy_exit"]<=100
                cnt["partner_did_not_contact_on_first_turn"]+=row["partner_turn"]>0 and not row["partner_contact"]
            table_stats[p]=dict(cnt)
        avoidable = [r for r in local if r["branches"][r["choices"]["pressure"]]["outcome"]["loss_140"] is True
                     and any(b["outcome"]["final"] in ("win","draw") for b in r["branches"].values())]
        table_stats["early_loss_with_nonlosing_alternative"] = len(avoidable)
        panel[table]=table_stats
    return panel


def data(directory, mode="early-teacher-labels"):
    roster = model_roster()
    weights = [w for n in ("candidate","historical") for w in roster[n]["weights"]]
    run = subprocess.run([directory/"deadline_opening.exe",RULES,mode],
                         input=" ".join(map(str,(SEED,COUNT,*weights))),capture_output=True,text=True,
                         encoding="utf8",errors="replace",timeout=200)
    require(run.returncode==0,f"early branches failed: {run.stderr.strip()}")
    roots,teachers,branches,tracked=parse(run.stdout)
    cases = [public_record(k,roots[k],teachers[k],branches[k]) for k in sorted(roots)]
    baseline = {}
    for row in cases:
        selected = row["branches"][row["choices"]["pressure"]]["replay"]
        key = (row["family"],row["actor_seat"],row["table"])
        receipt = (selected["plies"],selected["outcome"],selected["terminal_trace"])
        require(baseline.setdefault(key,receipt)==receipt,"unchanged policy differed across two prefix windows")
    wire = "".join(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n" for row in cases)
    summary = {"opening_exposures":tracked,"roots":len(cases),"candidate_outcomes":sum(len(r["branches"]) for r in cases),
               "families":COUNT,"sha256":sha256(wire.encode()).hexdigest(),"tables":describe(cases)}
    return wire, summary


def main(audit):
    directory = ROOT/"build/deadline-opening"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        build(directory)
    wire,summary = data(directory)
    for name,content in (("early-public.jsonl",wire),("summary.json",json.dumps(summary,sort_keys=True,indent=2)+"\n")):
        receipt = directory/name
        if audit:
            require(receipt.read_bytes()==content.encode(),f"early evidence differed: {name}")
        else:
            pending = receipt.with_suffix(".tmp")
            pending.write_bytes(content.encode())
            pending.replace(receipt)
    print(json.dumps(summary,sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    if sys.argv[1:] not in ([],["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(sys.argv[1:]==["audit"])
