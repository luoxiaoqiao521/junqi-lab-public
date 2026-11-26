"""Measure one public post-retirement preparation against continued pressure."""
from collections import Counter
import json
import os
from statistics import median
import subprocess
import sys

from measure_deadline_opening import COLUMNS, ROOT, RULES, TABLES, build, measure, model_roster, paired, require


SEED, COUNT = 1089900, 32
GATE = ("offered_at", "reply", "preparation_prefix")
REPLY = ("no_opportunity", "no_partner_turn", "closed", "available_unused", "used")


def comparison(rows):
    report = {}
    for table,label in enumerate(TABLES):
        base,new = ([row for row in rows if row["table"] == table and row["arm"] == code] for code in (7, 8))
        for a,b in zip(base,new,strict=True):
            require(all(a[k] == b[k] for k in ("offered_at", "preparation_prefix", "pre_exit_trace", "enemy_retired")),
                    "comparison diverged before the offer")
            if not a["offered_at"]:
                require(all(a[k] == b[k] for k in (*COLUMNS, *GATE) if k != "arm"), "no-opportunity pair diverged")
        old_score,new_score = map(measure, (base,new))
        offered = [(a,b) for a,b in zip(base,new,strict=True) if b["offered_at"]]
        matched_wins = [b["plies"] - a["plies"] for a,b in offered if a["outcome"] == b["outcome"] == 2]
        report[label] = {"baseline": old_score, "prepare_once": new_score,
                         "pair_effect": paired(base,new,old_score,new_score),
                         "matched_wins": len(matched_wins),
                         "median_paired_turn_difference_in_wins": median(matched_wins) if matched_wins else None,
                         "offered_layouts": len({b["seed"] for a,b in offered}),
                         "before_partner_turn": dict(Counter(REPLY[a["reply"]] for a,b in offered)),
                         "after_prepare": dict(Counter(REPLY[b["reply"]] for a,b in offered))}
    return report


def calculate(directory):
    roster = model_roster()
    weights = [weight for name in ("candidate", "historical") for weight in roster[name]["weights"]]
    result = subprocess.run([directory / "deadline_opening.exe", RULES, "gate"],
                            input=" ".join(map(str, (SEED, COUNT, *weights))), capture_output=True,
                            text=True, encoding="utf8", errors="replace", timeout=220)
    require(result.returncode == 0, f"post-retirement gate check failed: {result.stderr.strip()}")
    rows = [dict(zip((*COLUMNS, *GATE), map(int,line.split()), strict=True)) for line in result.stdout.splitlines()]
    require([(r["table"],r["arm"],r["seed"],r["seat"]) for r in rows] ==
            [(t,a,s,j) for t in range(2) for a in (7,8) for s in range(SEED,SEED+COUNT) for j in range(4)],
            "incomplete post-retirement gate sample")
    require(all(0 <= r["offered_at"] <= min(140,r["plies"]) and 0 <= r["reply"] < len(REPLY) and
                ((not r["offered_at"] and r["reply"] == 0) or
                 (0 < r["enemy_retired"] < r["offered_at"] and r["reply"] > 0)) for r in rows),
            "invalid gate evidence")
    return {"seed": SEED, "count": COUNT, "report": comparison(rows), "rows": rows}


def main(audit):
    directory = ROOT / "build/deadline-opening" / os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        build(directory)
    value = calculate(directory)
    receipt = directory / "summary.json"
    wire = json.dumps(value,sort_keys=True,indent=2) + "\n"
    if audit:
        require(receipt.read_text(encoding="utf8") == wire, "post-retirement gate audit differed")
    else:
        pending = receipt.with_suffix(".tmp")
        pending.write_text(wire,encoding="utf8")
        pending.replace(receipt)
    print(json.dumps(value["report"],sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__ == "__main__":
    if sys.argv[1:] not in ([],["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(sys.argv[1:] == ["audit"])
