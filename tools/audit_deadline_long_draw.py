"""Reconstruct the first override and measure its direct and later consequences."""
from collections import Counter
import json
import os
from statistics import mean, median
import subprocess
import sys

from measure_deadline_opening import COLUMNS, ROOT, RULES, TABLES, build, model_roster, require
from measure_deadline_rollout import TRACES, WORK


SOURCE = ROOT / "build/deadline-opening/confirm-combined-deadline-response-232de7daee14ee0335db/summary.json"
TAIL = ("offered_at", "actor", "original_move", "proposed_move", "applied", "offer_trace", "first_result",
        "partner_offset", "partner_contact", "contacts_next12", "friend_exits_next12", "enemy_exits_next12", "quiet_peak")
ARMS = ((11,"frozen"),(13,"combined"),(14,"first_only"),(15,"skip_first"))


def summarize(pairs):
    metrics={}
    for _,label in ARMS:
        rows=[p[label] for p in pairs]
        if not rows:
            continue
        offers=[r for r in rows if r["offered_at"]]
        metrics[label]={"outcomes":dict(Counter(r["outcome"] for r in rows)),
                        "mean_observed":mean(r["plies"] for r in rows),
                        "wins_within_140":sum(r["outcome"]==2 and r["plies"]<=140 for r in rows),
                        "beyond_280":sum(r["plies"]>280 for r in rows),
                        "first_enemy_by_140":sum(0<r["enemy_retired"]<=140 for r in rows),
                        "first_enemy_med":median(r["enemy_retired"] for r in rows if r["enemy_retired"])
                        if any(r["enemy_retired"] for r in rows) else None}
        if offers:
            metrics[label]["first_result"] = dict(Counter(r["first_result"] for r in offers))
            metrics[label]["mean_contacts_12"] = mean(r["contacts_next12"] for r in offers)
            metrics[label]["partner_first_contacts"] = sum(r["partner_contact"] for r in offers)
    compare=Counter(f"{int(p['first_only']['outcome']==2)}/{int(p['skip_first']['outcome']==2)}" for p in pairs)
    return {"pairs":len(pairs),"panels":metrics,"first_only_win_vs_skip_first_win":dict(compare)}


def report(rows):
    tests={}
    for table,name in enumerate(TABLES):
        groups={label:[r for r in rows if r["table"]==table and r["arm"]==arm] for arm,label in ARMS}
        pairs=[dict(zip(groups,seq,strict=True)) for seq in zip(*groups.values(),strict=True)]
        for p in pairs:
            original=p["combined"]
            for label in ("first_only","skip_first"):
                altered=p[label]
                keys=("offered_at","actor","original_move","proposed_move","offer_trace")
                require(all(original[k]==altered[k] for k in keys),"first decision root changed before the intervention")
            require(p["first_only"]["applied"]==bool(original["offered_at"]),"first-only intervention count")
            require(p["skip_first"]["applied"]==p["skip_first"]["rollout_changes"]-bool(original["offered_at"]),
                    "first override was not precisely omitted")
            require(original["first_result"]==p["first_only"]["first_result"],"shared first action led to a different collision")
        affected=[p for p in pairs if p["frozen"]["outcome"]==2 and p["combined"]["outcome"]==1]
        fast=[p for p in pairs if p["combined"]["outcome"]==2 and p["combined"]["plies"]<=140 and
              (p["frozen"]["outcome"]!=2 or p["frozen"]["plies"]>140)]
        tests[name]={"all":summarize(pairs),"win_to_draw":summarize(affected),"new_early_wins":summarize(fast)}
    require(sum(t["win_to_draw"]["pairs"] for t in tests.values())==28,"changed win-to-draw set")
    return tests


def calculate(directory):
    source=json.loads(SOURCE.read_text(encoding="utf8"))
    roster=model_roster()
    weights=[weight for name in ("candidate","historical") for weight in roster[name]["weights"]]
    start,count=source["seed"],source["count"]
    stdin=" ".join(map(str,(start,count,*weights)))
    result=subprocess.run([directory/"deadline_opening.exe",RULES,"long-draw"],input=stdin,capture_output=True,
                          text=True,encoding="utf8",errors="replace",timeout=240)
    require(result.returncode==0,f"long-draw replay failed: {result.stderr.strip()}")
    keys=(*COLUMNS,*WORK,*TRACES)
    rows=[dict(zip((*keys,*TAIL),map(int,line.split()),strict=True)) for line in result.stdout.splitlines()]
    grid=[(t,a,s,j) for t in range(2) for a,_ in ARMS for s in range(start,start+count) for j in range(4)]
    require([(r["table"],r["arm"],r["seed"],r["seat"]) for r in rows]==grid,"incomplete long-draw sample")
    for old,new in zip((r for r in source["rows"] if r["arm"] in (11,13)),
                       (r for r in rows if r["arm"] in (11,13)),strict=True):
        require(all(old[k]==new[k] for k in keys),"instrumentation altered an existing trajectory")
    return {"report":report(rows),"rows":rows}


def main(audit):
    directory=ROOT/"build/deadline-opening"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        build(directory)
    value=calculate(directory)
    receipt=directory/"summary.json"
    wire=json.dumps(value,sort_keys=True,indent=2)+"\n"
    if audit:
        require(receipt.read_text(encoding="utf8")==wire,"long-draw audit differed")
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
