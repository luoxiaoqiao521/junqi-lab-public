"""Test a public bomb-resource decision on paired complete games."""
import json
import os
import subprocess
import sys

from measure_deadline_opening import COLUMNS, ROOT, RULES, TABLES, build, measure, model_roster, paired, require
from measure_deadline_rollout import TRACES, WORK


BOMB=("first_bomb","bomb_prefix","offers","considered","critical", "changed_bomb","contacts", "low", "large", "mine", "flag")
SEED,COUNT=1090700,24


def comparison(rows):
    output={}
    for p,partner in enumerate(("pressure_pair","historical_partner")):
        for t,table in enumerate(TABLES):
            base,new=([r for r in rows if (r["partner"],r["table"],r["arm"])==(p,t,arm)] for arm in (15,16))
            for a,b in zip(base,new,strict=True):
                require(all(a[k]==b[k] for k in ("first_bomb","bomb_prefix")),"bomb roots diverged before the decision")
                if not b["changed_bomb"]:
                    require(a["event_trace"]==b["event_trace"],"unchanged bomb choice altered trajectory")
            before,after=measure(base),measure(new)
            output[f"{partner}/{table}"]={"before":before,"after":after,"delta":paired(base,new,before,after),
                "bomb_before":{k:sum(x[k] for x in base) for k in BOMB[2:]},
                "bomb_after":{k:sum(x[k] for x in new) for k in BOMB[2:]},
                "first_bomb_by_140":sum(0<b["first_bomb"]<=140 for b in new),
                "affected_games":sum(bool(b["changed_bomb"]) for b in new),
                "win_to_draw":sum(a["outcome"]==2 and b["outcome"]==1 for a,b in zip(base,new,strict=True)),
                "win_to_loss":sum(a["outcome"]==2 and b["outcome"]==0 for a,b in zip(base,new,strict=True))}
    return output


def calculate(directory):
    roster=model_roster()
    weights=[w for name in ("candidate","historical") for w in roster[name]["weights"]]
    trial=subprocess.run([directory/"deadline_opening.exe",RULES,"bomb-balance"],
                         input=" ".join(map(str,(SEED,COUNT,*weights))),capture_output=True,text=True,
                         encoding="utf8",errors="replace",timeout=240)
    require(trial.returncode==0,f"bomb experiment failed: {trial.stderr.strip()}")
    rows=[dict(zip((*COLUMNS,*WORK,*TRACES,*BOMB,"partner"),map(int,line.split()),strict=True))
          for line in trial.stdout.splitlines()]
    grid=[(p,t,a,s,j) for p in range(2) for t in range(2) for a in (15,16)
          for s in range(SEED,SEED+COUNT) for j in range(4)]
    require([(r["partner"],r["table"],r["arm"],r["seed"],r["seat"]) for r in rows]==grid,"incomplete bomb matrix")
    require(all(0<=r["changed_bomb"]<=r["considered"]<=7 and r["contacts"]==r["offers"]-r["changed_bomb"]
                for r in rows),"invalid bomb accounting")
    return {"seed":SEED,"count":COUNT,"report":comparison(rows),"rows":rows}


def main(audit):
    directory=ROOT/"build/deadline-opening"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        build(directory)
    result=calculate(directory)
    wire=json.dumps(result,sort_keys=True,indent=2)+"\n"
    receipt=directory/"summary.json"
    if audit:
        require(receipt.read_text(encoding="utf8")==wire,"bomb recalculation differed")
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
