"""Check whether research students retain their gains with a frozen historical peer."""
from collections import Counter
from itertools import product
import json
import os
import shutil
import sys

from full_mode_runtime import FullModeRun, sha
from serial_dark_ordinary_inheritance import difference
from serial_dark_ordinary_learning import RULES, assert_layout_separation, run_native, summarize
from serial_dynamic_reply import ROOT, build, require
from serial_formation_matrix import contrast
from serial_worker import read, write_receipt


PEER=ROOT/"build/deadline-opening/train-early-deadline-student-3b863a7a46f46ca5f379/student.json"
STUDENTS=ROOT/"build/dark-phase-coverage/compare-four-dark-ordinary-phase-coverage-3438dbf87c2b225545be"
ARMS=("early","whole")
INPUTS=("grid.json","peer.json","parent.json","early.json","whole.json","engine/deadline_opening.exe","engine/rules.json")


def prepare(scope):
    folder=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    shutil.copyfile(PEER,folder/"peer.json")
    for key in ("parent",*ARMS):
        shutil.copyfile(STUDENTS/(key+".json"),folder/(key+".json"))
    grid={phase:[(start+i,i%4,t) for i,t in product(range(count),range(4))]
          for phase,start,count in (("development",24127000,12),("confirmation",24128000,32))}
    assert_layout_separation(grid)
    write_receipt(folder/"grid.json",grid)
    write_receipt(folder/"fingerprints.json",{name:sha(folder/name) for name in INPUTS})


def assess(rows,first):
    panel=summarize(rows,first)
    for table,detail in panel.items():
        runs=[r for r in rows if r["table"]==table]
        turns=sum(r["peer_turns"] for r in runs)
        changes=sum(r["peer_changes"] for r in runs)
        require((turns>0 and changes>0) if table else (turns==changes==0),"historical model was never used")
        detail.update(peer_turns=turns,peer_changes=changes)
    return panel


def evaluate(scope,grid,models):
    decorated={}
    frozen=None
    for name in ARMS:
        rows,_,first,trace=run_native(scope,"H",grid,(models["parent"],models[name]),peer=models["peer"])
        baseline,_,_,_=run_native(scope,"E",[r for r in grid if r[2]==0],(models["parent"],models[name]))
        control=[{k:v for k,v in r.items() if not k.startswith("peer_")} for r in rows if r["table"]==0]
        require(control==baseline,"historical entry changed a table with no assigned historical seats")
        if frozen is None:
            frozen=rows[::2]
        require(frozen==rows[::2],"frozen parent games changed between candidates")
        panel=assess(rows,first)
        pairs=list(zip(rows[::2],rows[1::2],strict=True))
        control_deltas={a["seed"]:difference(a,b) for a,b in pairs if not a["table"]}
        interaction={}
        for table in (1,2,3):
            shifted=[]
            for a,b in pairs:
                if a["table"]==table:
                    low,high=difference(a,b)
                    base_low,base_high=control_deltas[a["seed"]]
                    shifted.append((low-base_high,high-base_low))
            interaction[table]=contrast(shifted)
        decorated[name]=dict(games=rows,first=first,panel=panel,relative_to_control=interaction,**trace)
    return decorated


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-historical-peers"/os.environ["JUNQI_SERIAL_KEY"])
    if not audit:
        prepare(scope)
    folder=scope.folder
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"historical peer inputs changed")
    models={n:read(folder/(n+".json")) for n in ("parent","peer",*ARMS)}
    stage="confirmation" if audit else "development"
    grid=[tuple(r) for r in read(folder/"grid.json")[stage]]
    evaluated=evaluate(scope,grid,models)
    write_receipt(folder/(stage+".json"),evaluated)
    if audit:
        target=evaluated["early"]["panel"]
        verdict=("supported" if all(target[t]["interval"][0]>0 for t in (1,2,3)) else
                 "refuted" if any(target[t]["interval"][1]<0 for t in (1,2,3)) else "inconclusive")
        print("JUNQI_RESEARCH "+verdict)
    print(json.dumps({n:{k:v for k,v in arm.items() if k in ("panel","relative_to_control","seconds")}
                      for n,arm in evaluated.items()}))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected historical peer comparison stage")
    main(bool(sys.argv[1:]))
