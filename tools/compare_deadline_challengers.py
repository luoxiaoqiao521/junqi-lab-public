"""Compare existing teacher policies on fresh single-seat full openings."""
from collections import Counter
import json
import os
import subprocess
import sys

from compare_deadline_two_targets import extra
from measure_deadline_opening import COLUMNS, LABEL, ROOT, RULES, build, measure, model_roster, paired, require
from measure_deadline_rollout import TRACES, WORK


SEED, COUNT = 1092400, 24
ARMS = {0: "historical_reference", 2: "pressure", 15: "bounded_reply", 17: "joint_routes"}
PARTNERS = {1: "historical_partner", 2: "heuristic_partner"}
OPPONENTS = {0: "random", 1: "historical", 2: "active_pressure"}


def matches(directory):
    roster = model_roster()
    weights = [w for name in ("candidate", "historical") for w in roster[name]["weights"]]
    proc = subprocess.run([directory / "deadline_opening.exe", RULES, "challenger-race"],
                          input=" ".join(map(str, (SEED, COUNT, *weights))), capture_output=True,
                          text=True, encoding="utf8", errors="replace", timeout=200)
    require(proc.returncode == 0, f"challenger run failed: {proc.stderr.strip()}")
    rows = []
    for line in proc.stdout.splitlines():
        values = list(map(int, line.split()))
        suffix = (*WORK, *TRACES) if values[4] in (15, 17) else ()
        rows.append(dict(zip((*COLUMNS, *suffix, "partner"), values, strict=True)))
    grid = [(p,t,a,s,j) for p in PARTNERS for t in OPPONENTS for a in ARMS
            for s in range(SEED, SEED + COUNT) for j in range(4)]
    require([(r["partner"], r["table"], r["arm"], r["seed"], r["seat"]) for r in rows] == grid,
            "incomplete challenger matrix")
    require(all(r["outcome"] in LABEL and 0 < r["plies"] <= 3000 and
                0 <= r["changed100"] <= r["changed140"] <= r["changed"] <= r["plies"] for r in rows),
            "invalid game row")
    return rows


def cells(rows):
    report = {}
    for p, partner in PARTNERS.items():
        for t, table in OPPONENTS.items():
            groups = {a: [r for r in rows if (r["partner"], r["table"], r["arm"]) == (p,t,a)]
                      for a in ARMS}
            require(all(x["before_forecast"] == y["before_forecast"] and
                        x["first_window"] == y["first_window"]
                        for x,y in zip(groups[15], groups[17], strict=True)),
                    "route evaluator changed a shared forecast prefix")
            before = measure(groups[0])
            panel = {}
            for arm, name in ARMS.items():
                games = groups[arm]
                stats = measure(games)
                panel[name] = stats | extra(games)
                if arm:
                    panel[name]["relative_to_reference"] = paired(groups[0], games, before, stats)
                    panel[name]["outcome_changes"] = dict(Counter(
                        f"{LABEL[x['outcome']]}/{LABEL[y['outcome']]}"
                        for x,y in zip(groups[0], games, strict=True) if x["outcome"] != y["outcome"]))
            report[f"{partner}/{table}"] = panel
    return report


def main(audit):
    directory = ROOT / "build/deadline-opening" / os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True, exist_ok=True)
        build(directory)
    rows = matches(directory)
    report = cells(rows)
    wire = json.dumps({"seed": SEED, "layouts": COUNT, "report": report, "rows": rows},
                      sort_keys=True, indent=2) + "\n"
    receipt = directory / "summary.json"
    if audit:
        require(receipt.read_bytes() == wire.encode(), "challenger matrix did not reproduce")
    else:
        pending = receipt.with_suffix(".tmp")
        pending.write_bytes(wire.encode())
        pending.replace(receipt)
    for cell, panel in report.items():
        print(cell, json.dumps({name: {"mean": m["mean_complete"], "w140": m["before"]["140"].get("win",0),
                                     "outcome": m["outcomes"]} for name,m in panel.items()}, sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(sys.argv[1:] == ["audit"])
