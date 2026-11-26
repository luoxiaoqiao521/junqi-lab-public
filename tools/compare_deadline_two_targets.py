"""Compare joint enemy-seat readiness under the same sampled continuation budget."""
import json
import os
from statistics import mean, median
import subprocess
import sys

from measure_deadline_opening import COLUMNS, ROOT, RULES, TABLES, build, measure, model_roster, paired, require
from measure_deadline_rollout import TRACES, WORK


SEED,COUNT=1090900,16


def extra(rows):
    gaps=[r["plies"]-r["enemy_retired"] for r in rows if r["outcome"]==2 and r["enemy_retired"]]
    return {"winning_gap_median":median(gaps) if gaps else None,
            "past_280":sum(r["plies"]>280 for r in rows),
            "first_exit_floor":mean(min(r["plies"],r["enemy_retired"]+1) if r["enemy_retired"] else r["plies"] for r in rows)}


def compare(rows,arms=(15,17)):
    out={}
    for p,partner in enumerate(("pressure_pair","historical_partner")):
        for t,table in enumerate(TABLES):
            base,test=([r for r in rows if (r["partner"],r["table"],r["arm"])==(p,t,arm)] for arm in arms)
            pairs=list(zip(base,test,strict=True))
            require(all(a["first_window"]==b["first_window"] and a["before_forecast"]==b["before_forecast"]
                        for a,b in pairs),"different starting trace before joint forecast")
            before,after=measure(base),measure(test)
            out[f"{partner}/{table}"]={"before":before|extra(base),"after":after|extra(test),
                "delta":paired(base,test,before,after),
                "different_games":sum(a["event_trace"]!=b["event_trace"] for a,b in pairs),
                "win_to_draw":sum(a["outcome"]==2 and b["outcome"]==1 for a,b in pairs),
                "win_to_loss":sum(a["outcome"]==2 and b["outcome"]==0 for a,b in pairs)}
    return out


def calculate(directory,staged):
    roster=model_roster()
    weights=[w for name in ("candidate","historical") for w in roster[name]["weights"]]
    mode,seed,arms=("staged-target-path",SEED+100,(15,17,18)) if staged else ("two-target-path",SEED,(15,17))
    proc=subprocess.run([directory/"deadline_opening.exe",RULES,mode],
                         input=" ".join(map(str,(seed,COUNT,*weights))),capture_output=True,text=True,
                         encoding="utf8",errors="replace",timeout=240)
    require(proc.returncode==0,f"joint target experiment failed: {proc.stderr.strip()}")
    rows=[dict(zip((*COLUMNS,*WORK,*TRACES,"partner"),map(int,line.split()),strict=True))
          for line in proc.stdout.splitlines()]
    grid=[(p,t,a,s,j) for p in range(2) for t in range(2) for a in arms
          for s in range(seed,seed+COUNT) for j in range(4)]
    require([(r["partner"],r["table"],r["arm"],r["seed"],r["seat"]) for r in rows]==grid,"incomplete joint matrix")
    report=compare(rows)
    if staged:
        report={"joint_vs_baseline":report,"staged_vs_baseline":compare(rows,(15,18)),
                "staged_vs_joint":compare(rows,(17,18))}
    return {"seed":seed,"count":COUNT,"report":report,"rows":rows}


def main(audit,staged=False):
    directory=ROOT/"build/deadline-opening"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        build(directory)
    result=calculate(directory,staged)
    wire=json.dumps(result,sort_keys=True,indent=2)+"\n"
    receipt=directory/"summary.json"
    if audit:
        require(receipt.read_text(encoding="utf8")==wire,"joint target recalculation differed")
    else:
        pending=receipt.with_suffix(".tmp")
        pending.write_text(wire,encoding="utf8")
        pending.replace(receipt)
    print(json.dumps(result["report"],sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    if sys.argv[1:] not in ([],["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(sys.argv[1:]==["audit"])
