"""Validate rollback and crossed partner styles on fresh complete games."""
from collections import Counter
import json
import os
from statistics import mean
import subprocess
import sys

from audit_deadline_long_draw import TAIL
from measure_deadline_opening import COLUMNS, ROOT, RULES, TABLES, build, measure, model_roster, paired, require
from measure_deadline_rollout import TRACES, WORK, win_curve_area


SEED,COUNT=1090500,24
PARTNERS=("pressure_pair","historical_partner","heuristic_partner")
ARMS=((11,"baseline"),(13,"combined"),(15,"skip_first"))


def comparisons(rows):
    tables={}
    for index,partner in enumerate(PARTNERS):
        for t,enemy in enumerate(TABLES):
            cells={label:[r for r in rows if r["partner"]==index and r["table"]==t and r["arm"]==mode]
                   for mode,label in ARMS}
            baseline=cells["baseline"]
            scores={name:measure(group) for name,group in cells.items()}
            for name,group in cells.items():
                scores[name]["past_280"]=sum(r["plies"]>280 for r in group)
                scores[name]["win_curve_area_140"]=win_curve_area(group,140)
                if name=="baseline":
                    continue
                paired_games=list(zip(baseline,group,strict=True))
                for a,b in paired_games:
                    require(a["first_window"]==b["first_window"] and a["before_forecast"]==b["before_forecast"],
                            "partner comparison starts diverged")
                    if not b["applied"]:
                        require(a["event_trace"]==b["event_trace"],"reverted moves altered a no-change trace")
                scores[name]["vs_baseline"]=paired(baseline,group,scores["baseline"],scores[name])
                scores[name]["result_transitions"]=dict(Counter(f"{a['outcome']}/{b['outcome']}" for a,b in paired_games))
                both_win=[b["plies"]-a["plies"] for a,b in paired_games if a["outcome"]==b["outcome"]==2]
                scores[name]["both_win_turn_difference"]=mean(both_win) if both_win else None
            tables[f"{partner}/{enemy}"]=scores
    return tables


def calculate(directory):
    roster=model_roster()
    weights=[weight for name in ("candidate","historical") for weight in roster[name]["weights"]]
    stdin=" ".join(map(str,(SEED,COUNT,*weights)))
    proc=subprocess.run([directory/"deadline_opening.exe",RULES,"partner-mix"],input=stdin,capture_output=True,
                        text=True,encoding="utf8",errors="replace",timeout=240)
    require(proc.returncode==0,f"unfamiliar partner replay failed: {proc.stderr.strip()}")
    keys=(*COLUMNS,*WORK,*TRACES,*TAIL,"partner")
    rows=[dict(zip(keys,map(int,line.split()),strict=True)) for line in proc.stdout.splitlines()]
    require([(r["partner"],r["table"],r["arm"],r["seed"],r["seat"]) for r in rows]==
            [(p,t,a,s,j) for p in range(3) for t in range(2) for a,_ in ARMS
             for s in range(SEED,SEED+COUNT) for j in range(4)],"incomplete crossed partner sample")
    return {"seed":SEED,"count":COUNT,"report":comparisons(rows),"rows":rows}


def main(audit):
    directory=ROOT/"build/deadline-opening"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        build(directory)
    value=calculate(directory)
    receipt=directory/"summary.json"
    wire=json.dumps(value,sort_keys=True,indent=2)+"\n"
    if audit:
        require(receipt.read_text(encoding="utf8")==wire,"crossed partner audit differed")
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
