"""Hold the shortlist and rollout budget fixed and change the opponent response."""
import json
import os
import subprocess
import sys

from compare_deadline_two_targets import extra
from measure_deadline_opening import COLUMNS, ROOT, RULES, build, measure, model_roster, paired, require
from measure_deadline_rollout import TRACES, WORK


SEED,COUNT=1093000,24


def compare(directory):
    roster=model_roster()
    weights=[w for n in ("candidate","historical") for w in roster[n]["weights"]]
    run=subprocess.run([directory/"deadline_opening.exe",RULES,"active-counterplay-forecast"],
                       input=" ".join(map(str,(SEED,COUNT,*weights))),capture_output=True,text=True,
                       encoding="utf8",errors="replace",timeout=200)
    require(run.returncode==0,f"counterplay skipped: {run.stderr.strip()}")
    keys=(*COLUMNS,*WORK,*TRACES,"partner")
    rows=[dict(zip(keys,map(int,line.split()),strict=True)) for line in run.stdout.splitlines()]
    grid=[(p,a,s,j) for p in (1,2) for a in (17,22) for s in range(SEED,SEED+COUNT) for j in range(4)]
    require([(r["partner"],r["arm"],r["seed"],r["seat"]) for r in rows]==grid
            and all(r["table"]==2 for r in rows),"incomplete or mismatched opponent grid")
    report={}
    for p,partner in ((1,"historical_partner"),(2,"heuristic_partner")):
        before,after=([r for r in rows if r["partner"]==p and r["arm"]==arm] for arm in (17,22))
        pairs=list(zip(before,after,strict=True))
        for x,y in pairs:
            require(x["before_forecast"]==y["before_forecast"] and x["first_window"]==y["first_window"],
                    "opening prefixes diverged before forecast")
        old,new=measure(before)|extra(before),measure(after)|extra(after)
        report[partner]={"before":old,"after":new,"delta":paired(before,after,old,new),
            "different_traces":sum(x["event_trace"]!=y["event_trace"] for x,y in pairs),
            "new_loss_by_140":sum(early_loss(y) and not early_loss(x) for x,y in pairs),
            "prevented_loss_by_140":sum(early_loss(x) and not early_loss(y) for x,y in pairs),
            "first_friendly_then_loss": [sum(early_loss(r) and r["friendly_retired"]>0
                and (not r["enemy_retired"] or r["friendly_retired"]<r["enemy_retired"]) for r in panel)
                for panel in (before,after)],
            "imagined_losses_selected":[sum(r["imagined_losses"] for r in panel) for panel in (before,after)]}
    return {"seed":SEED,"families":COUNT,"report":report,"rows":rows}


def early_loss(row):
    return row["outcome"]==0 and row["plies"]<=140


def main(audit):
    directory=ROOT/"build/deadline-opening"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        build(directory)
    result=compare(directory)
    wire=json.dumps(result,sort_keys=True,indent=2)+"\n"
    receipt=directory/"summary.json"
    if audit:
        require(receipt.read_bytes()==wire.encode(),"active counterplay reproduction differed")
    else:
        pending=receipt.with_suffix(".tmp")
        pending.write_bytes(wire.encode())
        pending.replace(receipt)
    print(json.dumps(result["report"],sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    if sys.argv[1:] not in ([],["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(sys.argv[1:]==["audit"])
