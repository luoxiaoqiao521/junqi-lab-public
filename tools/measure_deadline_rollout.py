"""Compare the same eligible decision stream with and without public rollouts."""
import json
import os
from statistics import mean, median
import subprocess
import sys

from measure_deadline_opening import COLUMNS, ROOT, RULES, TABLES, build, measure, model_roster, paired, require


SEED, COUNT = 1090100, 32
WORK = ("first_window", "evaluations", "rollout_changes", "early_changes", "absent",
        "simulations", "imagined_wins", "imagined_losses")
TRACES = ("before_forecast", "event_trace")


def win_curve_area(rows,horizon):
    reached = sum(horizon + 1 - r["plies"] for r in rows if r["outcome"] == 2 and r["plies"] <= horizon)
    missing = sum(horizon - r["plies"] for r in rows if r["outcome"] == -1 and r["plies"] < horizon)
    return [reached/len(rows), (reached+missing)/len(rows)]


def comparison(rows,first_exit,arms):
    report={}
    required=("first_window","before_forecast") if first_exit else (
        "first_window","before_forecast","pre_exit_trace","enemy_retired")
    for table,label in enumerate(TABLES):
        base,new=([r for r in rows if r["table"]==table and r["arm"]==arm] for arm in arms)
        for a,b in zip(base,new,strict=True):
            require(all(a[k]==b[k] for k in required),
                    "first evaluation diverged before rollout selection")
            if not a["rollout_changes"] and not b["rollout_changes"]:
                require(all(a[k]==b[k] for k in (*COLUMNS,"event_trace") if k!="arm"),
                        "a no-change forecast affected real play")
        old,measured=map(measure,(base,new))
        joint=[b["plies"]-a["plies"] for a,b in zip(base,new,strict=True) if a["outcome"]==b["outcome"]==2]
        record={"baseline":old,"response":measured,"pair_effect":paired(base,new,old,measured),
                "matched_wins":len(joint),"matched_win_mean_difference":mean(joint) if joint else None,
                "matched_win_median_difference":median(joint) if joint else None,
                "win_to_loss":sum(a["outcome"]==2 and b["outcome"]==0 for a,b in zip(base,new,strict=True)),
                "win_curve_area":{str(h):[win_curve_area(block,h) for block in (base,new)] for h in (100,140)},
                "prior_exit_windows":{str(h):sum(0<r["first_window"]<=h for r in new) for h in (100,140,280)},
                "optimistic_mean_floor":mean(r["first_window"] or r["plies"] for r in base),
                "affected_games":sum(r["rollout_changes"]>0 for r in new),
                "work":{k:sum(r[k] for r in new) for k in WORK if k!="first_window"}}
        report[label]=record
    return report


def calculate(directory,mode):
    roster=model_roster()
    weights=[weight for name in ("candidate","historical") for weight in roster[name]["weights"]]
    seed,arms={"response":(SEED,(9,10)),"first-exit":(SEED+100,(11,12)),
               "combined":(SEED+200,(11,12,13))}[mode]
    first_exit=mode!="response"
    run=subprocess.run([directory/"deadline_opening.exe",RULES,mode],
                       input=" ".join(map(str,(seed,COUNT,*weights))),capture_output=True,text=True,
                       encoding="utf8",errors="replace",timeout=240)
    require(run.returncode==0,f"public response rollout failed: {run.stderr.strip()}")
    rows=[dict(zip((*COLUMNS,*WORK,*TRACES),map(int,line.split()),strict=True)) for line in run.stdout.splitlines()]
    grid=[(t,a,s,j) for t in range(2) for a in arms for s in range(seed,seed+COUNT) for j in range(4)]
    require([(r["table"],r["arm"],r["seed"],r["seat"]) for r in rows]==grid,"incomplete response grid")
    require(all(0<=r["rollout_changes"]<=r["evaluations"]<=(14 if r["arm"]==13 else 7)
                and 0<=r["early_changes"]<=r["rollout_changes"]
                and 0<=r["absent"]<=r["evaluations"] and r["simulations"]<=24*r["evaluations"]
                and ((not r["first_window"] and not r["evaluations"]) or
                     (first_exit and 12<r["first_window"]<=min(100,r["enemy_retired"] or r["plies"],r["plies"])) or
                     (not first_exit and 0<r["enemy_retired"]<r["first_window"]<=min(280,r["plies"]))) for r in rows),
            "invalid forecast counts")
    report=comparison(rows,first_exit,arms[:2])
    if mode=="combined":
        report={"early_vs_frozen":report,
                "combined_vs_early":comparison(rows,False,(12,13)),
                "combined_vs_frozen":comparison(rows,True,(11,13))}
    return {"seed":seed,"count":COUNT,"report":report,"rows":rows}


def main(audit,mode="response"):
    directory=ROOT/"build/deadline-opening"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        build(directory)
    value=calculate(directory,mode)
    receipt=directory/"summary.json"
    wire=json.dumps(value,sort_keys=True,indent=2)+"\n"
    if audit:
        require(receipt.read_text(encoding="utf8")==wire,"public rollout replay differed")
    else:
        pending=receipt.with_suffix(".tmp")
        pending.write_text(wire,encoding="utf8")
        pending.replace(receipt)
    print(json.dumps(value["report"],sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    if sys.argv[1:] not in ([],["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(sys.argv[1:]==["audit"])
