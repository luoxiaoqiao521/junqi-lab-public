"""Replay the measured first-retirement continuations with public repetition counters."""
from collections import Counter
import json
import os
import subprocess
import sys

from measure_deadline_opening import COLUMNS, ROOT, RULES, TABLES, build, model_roster, require


REFERENCE = ROOT / "build/deadline-opening/confirm-first-retirement-conversion-d4ca9b15013bfcec3a80/summary.json"
COUNTERS = ("post_turns", "quiet", "reversals", "repeat", "persistent", "near_draw")


def calculate(directory):
    previous=json.loads(REFERENCE.read_text(encoding="utf8"))
    reference=[row for row in previous["rows"] if row["arm"]==2]
    weights=[weight for name in ("candidate", "historical") for weight in model_roster()[name]["weights"]]
    arguments=[directory/"deadline_opening.exe", RULES, "repeat"]
    trial=subprocess.run(arguments, input=" ".join(map(str, (*previous["inputs"]["seeds"], *weights))),
                         capture_output=True, text=True, encoding="utf8", errors="replace", timeout=220)
    require(trial.returncode==0, f"public repeat replay failed: {trial.stderr.strip()}")
    rows=[dict(zip((*COLUMNS, *COUNTERS), map(int,line.split()), strict=True))
          for line in trial.stdout.splitlines()]
    for a,b in zip(reference, rows, strict=True):
        require(b["arm"]==5 and all(a[k]==b[k] for k in COLUMNS if k!="arm"),
                "instrumented continuation altered the baseline")
        require(0<=b["near_draw"]<=b["persistent"]<=b["repeat"]<=b["quiet"]<=b["post_turns"]<=b["plies"]
                and 0<=b["reversals"]<=b["quiet"], "invalid public repeat counts")
    report={}
    for table,label in enumerate(TABLES):
        block=[row for row in rows if row["table"]==table]
        totals={key:sum(row[key] for row in block) for key in COUNTERS}
        verdicts=Counter(row["outcome"] for row in block if row["persistent"]>row["near_draw"])
        report[label]={"totals":totals,"candidate_games_by_result":dict(verdicts),
                       "games_reaching_first_retirement":sum(bool(row["enemy_retired"]) for row in block)}
    return {"report":report,"rows":rows}


def main(audit):
    key=os.environ["JUNQI_SERIAL_KEY"]
    directory=ROOT/"build/deadline-opening"/key
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        build(directory)
    value=calculate(directory)
    receipt=directory/"summary.json"
    if audit:
        require(receipt.read_text(encoding="utf8")==json.dumps(value,sort_keys=True,indent=2)+"\n",
                "public repeat audit differed")
    else:
        pending=receipt.with_suffix(".tmp")
        pending.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n",encoding="utf8")
        pending.replace(receipt)
    print(json.dumps(value["report"],sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    if sys.argv[1:] not in ([],["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(sys.argv[1:]==["audit"])
