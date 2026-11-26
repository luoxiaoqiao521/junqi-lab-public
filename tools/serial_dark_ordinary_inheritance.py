"""Compare continued and reset four-dark weights on the exact same new ordinary experience."""
from collections import Counter
from itertools import groupby, product
import json
import os
import shutil
from statistics import mean
import sys

from compare_book_learning import fit
from full_mode_runtime import FullModeRun, sha
from serial_dark_ordinary_learning import RATE, RULES, UPDATES, examples_xy, model_wire, run_native, summarize
from serial_dynamic_reply import ROOT, build, require
from serial_formation_matrix import contrast, score_difference
from serial_worker import read, write_receipt


PREVIOUS=ROOT/"build/dark-ordinary-learning/restore-four-dark-ordinary-learning-5051ce5317cedcbf2908"
ARMS=("inherit","restart")


def prepare(scope):
    folder=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    for old,new in (("parent.json","original.json"),("student.json","generation1.json")):
        shutil.copyfile(PREVIOUS/old,folder/new)
    grids=dict(train=[(24110000+i,(i//8)%4,(i//2)%4) for i in range(96)],
               development=[(24111000+i,i%4,t) for i,t in product(range(16),range(4))],
               confirmation=[(24112000+i,i%4,t) for i,t in product(range(32),range(4))])
    write_receipt(folder/"manifest.json",grids)
    write_receipt(folder/"fingerprints.json",{n:sha(folder/n) for n in
                  ("original.json","generation1.json","manifest.json","engine/deadline_opening.exe","engine/rules.json")})


def train(scope,first,original,grid):
    folder=scope.folder
    require(first["mean"]==original["mean"] and first["scale"]==original["scale"],"generation normalization drift")
    rows,examples,_,run=run_native(scope,"D",grid,(first,))
    x,y=examples_xy(examples,first)
    require(len({r["result"] for r in rows if r["result"]>=0})>=2,"new ordinary data lack outcome variation")
    write_receipt(folder/"training.json",dict(games=rows,examples=examples))
    source=sha(folder/"training.json")
    stats=dict(rows=len(x),ignored_rows=len(examples)-len(x),source=source,updates=UPDATES,rate=RATE,
               game_results=dict(Counter(r["result"] for r in rows)),**run)
    for name,parent in zip(ARMS,(first,original),strict=True):
        learned=fit(x,y,48120,24,parent["models"]["collision"],updates=UPDATES,rate=RATE)
        artifact=dict(parent,models={"collision":learned},training=stats,
                      parent_sha256=sha(folder/("generation1.json" if name=="inherit" else "original.json")))
        require(model_wire(artifact)!=model_wire(parent),"no parameter update in continuation control")
        write_receipt(folder/(name+".json"),artifact)
    write_receipt(folder/"next-fingerprints.json",{n:sha(folder/n) for n in ("training.json",*(a+".json" for a in ARMS))})


def difference(a,b):
    return (0,0) if a["hash"]==b["hash"] else score_difference(a["result"],b["result"])


def by_layout(pairs):
    return contrast([tuple(map(mean,zip(*(difference(a,b) for a,b in block),strict=True)))
                     for _,block in groupby(pairs,key=lambda r:r[0]["seed"])])


def evaluate(scope,grid,first):
    folder=scope.folder
    compare={}
    for name in ARMS:
        weights=read(folder/(name+".json"))
        rows,_,divergences,run=run_native(scope,"E",grid,(first,weights))
        compare[name]=dict(summary=summarize(rows,divergences),games=rows,first=divergences,**run)
    inherited=compare["inherit"]["games"]
    reset=compare["restart"]["games"]
    require(inherited[::2]==reset[::2],"the generation-one evaluation paths are not identical")
    against_first=list(zip(inherited[::2],inherited[1::2],strict=True))
    against_reset=list(zip(reset[1::2],inherited[1::2],strict=True))
    aggregate=dict(inherit_vs_first=by_layout(against_first),inherit_vs_reset=by_layout(against_reset))
    desks={t:contrast([difference(a,b) for a,b in against_reset if a["table"]==t]) for t in range(4)}
    return dict(arms=compare,aggregate=aggregate,inherit_vs_reset_by_desk=desks)


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-ordinary-inheritance"/os.environ["JUNQI_SERIAL_KEY"])
    folder=scope.folder
    if not audit:
        prepare(scope)
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"generation inputs changed")
    grid={k:[tuple(r) for r in v] for k,v in read(folder/"manifest.json").items()}
    first=read(folder/"generation1.json")
    if not audit:
        train(scope,first,read(folder/"original.json"),grid["train"])
    require(all(sha(folder/n)==v for n,v in read(folder/"next-fingerprints.json").items()),"generation outputs changed")
    stage="confirmation" if audit else "development"
    report=evaluate(scope,grid[stage],first)
    write_receipt(folder/(stage+".json"),report)
    if audit:
        aggregate=report["aggregate"]
        verdict="supported" if all(r["interval"][0]>0 for r in aggregate.values()) else (
            "refuted" if any(r["interval"][1]<0 for r in aggregate.values()) else "inconclusive")
        print("JUNQI_RESEARCH "+verdict)
    print(json.dumps(dict(aggregate=report["aggregate"],by_table=report["inherit_vs_reset_by_desk"],
                          training=read(folder/"inherit.json")["training"])))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected ordinary inheritance stage")
    main(bool(sys.argv[1:]))
