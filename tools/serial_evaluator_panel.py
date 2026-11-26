"""Run the frozen E1 calibration panel and review the exact same evidence."""
from collections import Counter
import json
import math
import os
import subprocess
import sys

from measure_deadline_opening import ROOT, RULES, build, file_sha, model_roster, require
from full_mode_runtime import current_scope
from serial_worker import write_receipt

PROFILES = ("fixed_9_by_4", "fixed_12_by_3", "conditional_9_to_12_by_3")


def directory():
    return ROOT / "build/evaluator-panel" / os.environ["JUNQI_SERIAL_KEY"]


def native(path, mode, payload=""):
    scope=current_scope.get()
    if scope is not None:
        require(path.resolve()==scope.engine,"active research cannot launch a second engine directory")
        return scope.run(mode,payload)
    run = subprocess.run([path/"deadline_opening.exe",RULES,mode], input=payload, encoding="utf8",
                         cwd=ROOT, capture_output=True, timeout=95)
    require(run.returncode == 0, f"{mode} failed: {run.stderr.strip()}")
    return run.stdout


def decode_action(fields):
    move, lower, upper, nodes = map(int,fields[:4])
    require(0 <= lower <= upper <= 2 and 0 <= nodes <= 12000,"invalid proof bounds")
    profiles = {}
    for j, name in enumerate(PROFILES):
        value, plies, extended = fields[4+j*3:7+j*3]
        value, plies, extended = float(value), int(plies), int(extended)
        require(math.isfinite(value) and 0 < plies <= 36 and 0 <= extended <= (3 if j == 2 else 0), "budget mismatch")
        profiles[name] = {"value":value,"plies":plies,"extended":extended}
    return {"move":move,"bounds":[lower,upper],"nodes":nodes,"profiles":profiles}


def summarize(raw):
    panel = [row.split() for row in raw.splitlines() if row.startswith("panel ")]
    require(len(raw.splitlines()) == 12+48 and len(panel) == 48, "incomplete proof or panel")
    results = {}
    rows = []
    for _, family, remaining, rotation, *values in panel:
        family, remaining, rotation = map(int,(family,remaining,rotation))
        require(len(values) == 26,"missing paired scores")
        a, b = decode_action(values[:13]), decode_action(values[13:])
        require(a["move"] != b["move"], "same-action controls belong to a separate test")
        lo, hi = a["bounds"]
        other_lo, other_hi = b["bounds"]
        expected = 1 if lo > other_hi else -1 if hi < other_lo else 0 if lo == hi == other_lo == other_hi else None
        group = (family, "base" if remaining<=5 else "longer")
        cell = results.setdefault(group,{p:Counter() for p in PROFILES})
        for name in PROFILES:
            gap = a["profiles"][name]["value"]-b["profiles"][name]["value"]
            ranking = int(gap>1e-9)-int(gap< -1e-9)
            status = "unresolved" if expected is None else "correct" if expected == ranking else "false_equal" if ranking==0 \
                     else "false_distinction" if expected==0 else "reversed"
            cell[name][status]+=1
            for key in ("plies","extended"):
                cell[name][key]+=a["profiles"][name][key]+b["profiles"][name][key]
        rows.append(dict(family=family,remaining=remaining,rotation=rotation,reference=expected,actions=[a,b]))
    require([(r["family"],r["remaining"],r["rotation"]) for r in rows]==
            [(f,t,s) for f in (0,1) for t in (1,2,5,9,10,11) for s in range(4)], "panel order or coverage changed")
    return {"independent_templates":2,"cells":{f"{k[0]}/{k[1]}":v for k,v in results.items()},"cases":rows,
            "decision":"diagnostic_only","blockers":["Independent blind structures are absent.",
                "Full-information fixture outcomes do not certify four-dark decision quality."]}


def main(audit):
    path = directory()
    if not audit:
        path.mkdir(parents=True, exist_ok=False)
        build(path)
    roster = model_roster()["historical"]
    payload = " ".join(map(str, (*roster["weights"],0)))
    raw = native(path,"compare-evaluator-panel",payload)
    record = dict(engine=file_sha(path/"deadline_opening.exe"), rules=file_sha(RULES),
                  weights=roster["artifact_sha256"], **summarize(raw))
    if audit:
        require(raw==(path/"panel.txt").read_text(encoding="utf8") and
                record==json.loads((path/"panel.json").read_text(encoding="utf8")), "review did not reproduce raw panel")
    else:
        (path/"panel.txt").write_text(raw,encoding="utf8")
        write_receipt(path/"panel.json",record)
    print(json.dumps({key:record[key] for key in ("independent_templates","cells","decision","blockers")},indent=2))
    print("JUNQI_VERDICT accepted")


if __name__ == "__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
