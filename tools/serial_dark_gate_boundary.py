"""Replay the previous timing cohort and validate the exact horizon contract."""
from collections import Counter
import json
import os
import sys

from full_mode_runtime import FullModeRun, sha
from serial_dark_gate_timing import INPUTS, assert_layout_separation, prepare, run
from serial_dynamic_reply import ROOT, require
from serial_worker import read, write_receipt


PRIOR=ROOT/"build/dark-gate-timing/audit-four-dark-effective-gate-timing-2f86ae35a083fcae5d7a"


def prepare_boundary(scope):
    prepare(scope)
    folder=scope.folder
    grid={"development":read(PRIOR/"grid.json")["confirmation"],
          "confirmation":[(24157000+i,i%4,(i//4%2)*4,(12,24,36,60,88,96)[i%6]) for i in range(48)]}
    assert_layout_separation(grid)
    write_receipt(folder/"grid.json",grid)
    write_receipt(folder/"fingerprints.json",{n:sha(folder/n) for n in INPUTS})
    write_receipt(folder/"source.json",{"prior":sha(PRIOR/"confirmation.json")})


def repeatability(prior,current):
    old={tuple(r[:3]):r[-1] for r in prior["roots"]}
    seen=Counter()
    for seed,seat,table,row in current["roots"]:
        before=old[seed,seat,table]
        fields=("ply","status","actions","gates","games")
        require(json.dumps([row.get(k) for k in fields])==json.dumps([before.get(k) for k in fields]),
                "timing boundary changed the legal original or terminal comparison")
        seen["all"]+=1
        if row["status"]==3:
            seen["covered"]+=1
            seen["choice_changes"]+=row["choice"]!=before["choice"]
    return seen


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-gate-boundary"/os.environ["JUNQI_SERIAL_KEY"])
    folder=scope.folder
    if not audit:
        prepare_boundary(scope)
    require(all(sha(folder/n)==s for n,s in read(folder/"fingerprints.json").items()) and
            sha(PRIOR/"confirmation.json")==read(folder/"source.json")["prior"],"gate boundary inputs changed")
    phase="confirmation" if audit else "development"
    grid=[tuple(r) for r in read(folder/"grid.json")[phase]]
    models={n:read(folder/(n+".json")) for n in ("original","peer")}
    report=run(scope,grid,models)
    if not audit:
        report["summary"]["replay"]=repeatability(read(PRIOR/"confirmation.json"),report)
    else:
        print("JUNQI_RESEARCH supported")
    write_receipt(folder/(phase+".json"),report)
    print(json.dumps(report["summary"]))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected gate-window stage")
    main(bool(sys.argv[1:]))
