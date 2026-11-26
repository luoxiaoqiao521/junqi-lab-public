"""Compare one textbook-inspired candidate slot with the original shortlist."""
from collections import Counter
import json
import os
import subprocess
import sys

from measure_deadline_opening import COLUMNS, ROOT, RULES, TABLES, build, model_roster, require
from measure_deadline_rollout import TRACES, WORK
from compare_deadline_two_targets import compare


SEED, COUNT = 1091100, 24
BOOK = ("book_offered", "book_selected", "book_applied", "first_book")


def calculate(directory,support=False):
    seed=SEED+100 if support else SEED
    arms=(17,20) if support else (17,19)
    mode="book-cover" if support else "book-candidate"
    roster=model_roster()
    weights=[w for name in ("candidate","historical") for w in roster[name]["weights"]]
    proc=subprocess.run([directory/"deadline_opening.exe",RULES,mode],
                        input=" ".join(map(str,(seed,COUNT,*weights))),capture_output=True,text=True,
                        encoding="utf8",errors="replace",timeout=240)
    require(proc.returncode==0,f"book candidate failed: {proc.stderr.strip()}")
    keys=(*COLUMNS,*WORK,*TRACES,*BOOK,"partner")
    rows=[dict(zip(keys,map(int,line.split()),strict=True)) for line in proc.stdout.splitlines()]
    grid=[(p,t,a,s,j) for p in range(2) for t in range(2) for a in arms
          for s in range(seed,seed+COUNT) for j in range(4)]
    require([(r["partner"],r["table"],r["arm"],r["seed"],r["seat"]) for r in rows]==grid,"incomplete book matrix")
    report=compare(rows,arms)
    for p,partner in enumerate(("pressure_pair","historical_partner")):
        for t,table in enumerate(TABLES):
            base,new=([r for r in rows if (r["partner"],r["table"],r["arm"])==(p,t,arm)] for arm in arms)
            for a,b in zip(base,new,strict=True):
                require(all(a[k]==0 for k in BOOK) and 0<=b["book_applied"]<=b["book_selected"]<=b["book_offered"]
                        <=b["evaluations"] and (0<b["first_book"]<=100 if b["book_applied"] else b["first_book"]==0),
                        "invalid book candidate funnel")
                if not b["book_offered"]:
                    require(a["event_trace"]==b["event_trace"],"no book candidate affected the game")
            cell=report[f"{partner}/{table}"]
            cell["nominations"]={k:sum(r[k] for r in new) for k in BOOK[:-1]}
            applied=[(a,b) for a,b in zip(base,new,strict=True) if b["book_applied"]]
            cell["first_enemy_with_applied"]={
                "games":len(applied),"by_140":[sum(0<r[i]["enemy_retired"]<=140 for r in applied) for i in (0,1)],
                "candidate_final":dict(Counter(b["outcome"] for _,b in applied))}
    return {"seed":seed,"count":COUNT,"report":report,"rows":rows}


def main(audit,support=False):
    directory=ROOT/"build/deadline-opening"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        build(directory)
    result=calculate(directory,support)
    wire=json.dumps(result,sort_keys=True,indent=2)+"\n"
    receipt=directory/"summary.json"
    if audit:
        require(receipt.read_text(encoding="utf8")==wire,"book candidate recalculation differed")
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
