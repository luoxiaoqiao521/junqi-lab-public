"""Compare early-only and full-trajectory sampling without changing the played games."""
from collections import Counter
from itertools import product
import json
import os
import shutil
import sys

from compare_book_learning import fit
from full_mode_runtime import FullModeRun, sha
from serial_dark_ordinary_inheritance import PREVIOUS, by_layout, difference
from serial_dark_ordinary_learning import RATE, RULES, UPDATES, assert_layout_separation, examples_xy, model_wire, run_native, summarize, value_error
from serial_dynamic_reply import ROOT, build, require
from serial_formation_matrix import contrast
from serial_worker import read, write_receipt


ARMS=("early","whole")
INPUTS=("parent.json","grid.json","engine/deadline_opening.exe","engine/rules.json")


def prepare(scope):
    folder=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    shutil.copyfile(PREVIOUS/"student.json",folder/"parent.json")
    seeds={"training":(24123000,96),"value_holdout":(24124000,48)}
    grid={name:[(base+i,(i//8)%4,(i//2)%4) for i in range(count)] for name,(base,count) in seeds.items()}
    for name,base,count in (("development",24125000,16),("confirmation",24126000,32)):
        grid[name]=[(base+i,i%4,t) for i,t in product(range(count),range(4))]
    assert_layout_separation(grid)
    write_receipt(folder/"grid.json",grid)
    write_receipt(folder/"fingerprints.json",{p:sha(folder/p) for p in INPUTS})


def learn(scope,parent,grid):
    folder=scope.folder
    recorded={}
    for name,mode in zip(ARMS,("D","L"),strict=True):
        rows,examples,_,run=run_native(scope,mode,grid,(parent,))
        recorded[name]=dict(games=rows,examples=examples,runtime=run)
    require(recorded["early"]["games"]==recorded["whole"]["games"],
            "sampling changed played games, outcomes or per-game example counts")
    for name,group in recorded.items():
        examples=group["examples"]
        x,y=examples_xy(examples,parent)
        learned=fit(x,y,48120,24,parent["models"]["collision"],updates=UPDATES,rate=RATE)
        artifact=dict(parent,models={"collision":learned},parent_sha256=sha(folder/"parent.json"),
                      training=dict(rows=len(x),updates=UPDATES,mode=name,**group["runtime"]))
        require(model_wire(artifact)!=model_wire(parent),"phase experiment did not update")
        write_receipt(folder/(name+".json"),artifact)
    require(len({r["training"]["rows"] for r in (read(folder/(a+".json")) for a in ARMS)})==1,"sample sizes changed")
    write_receipt(folder/"training.json",recorded)
    write_receipt(folder/"trained.json",{n:sha(folder/n) for n in ("training.json",*(a+".json" for a in ARMS))})


def evaluate(scope,parent,grid):
    evaluated={}
    for name in ARMS:
        model=read(scope.folder/(name+".json"))
        games,_,first,run=run_native(scope,"E",grid,(parent,model))
        evaluated[name]=dict(games=games,first=first,panel=summarize(games,first),training=model["training"],**run)
    a,b=(evaluated[name]["games"] for name in ARMS)
    require(a[::2]==b[::2],"frozen evaluation reference changed")
    pairs=list(zip(a[1::2],b[1::2],strict=True))
    return dict(arms=evaluated,whole_vs_early=by_layout(pairs),
                by_desk={t:contrast([difference(a,b) for a,b in pairs if a["table"]==t]) for t in range(4)},
                whole_vs_parent=by_layout(list(zip(a[::2],b[1::2],strict=True))),
                early_vs_parent=by_layout(list(zip(a[::2],a[1::2],strict=True))),
                outcomes={k:dict(Counter(r["result"] for r in e["games"][1::2])) for k,e in evaluated.items()},
                layouts=len({s for s,*_ in grid}),games=4*len(grid))


def value_panel(scope,parent,grid):
    games,examples,_,run=run_native(scope,"L",grid,(parent,))
    groups={"before_144":[r for r in examples if r["features"][0]*140<143.5 and r["outcome"]>=0],
            "from_144":[r for r in examples if r["features"][0]*140>=143.5 and r["outcome"]>=0]}
    require(all(groups.values()),"value holdout has no rows in one stage")
    panel={name:dict(rows=len(rows),sources=len({r["seed"] for r in rows}),parent=value_error(rows,parent),
                     **{arm:value_error(rows,read(scope.folder/(arm+".json"))) for arm in ARMS})
           for name,rows in groups.items()}
    write_receipt(scope.folder/"heldout-values.json",dict(games=games,examples=examples,runtime=run,panel=panel))
    return panel


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-phase-coverage"/os.environ["JUNQI_SERIAL_KEY"])
    folder=scope.folder
    if not audit:
        prepare(scope)
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"phase comparison inputs changed")
    parent=read(folder/"parent.json")
    grid={k:[tuple(r) for r in v] for k,v in read(folder/"grid.json").items()}
    if not audit:
        learn(scope,parent,grid["training"])
    require(all(sha(folder/n)==v for n,v in read(folder/"trained.json").items()),"phase training outputs changed")
    stage="confirmation" if audit else "development"
    report=evaluate(scope,parent,grid[stage])
    if audit:
        report["value_panel"]=value_panel(scope,parent,grid["value_holdout"])
        low,high=report["whole_vs_early"]["interval"]
        print("JUNQI_RESEARCH "+("supported" if low>0 else "refuted" if high<0 else "inconclusive"))
    write_receipt(folder/(stage+".json"),report)
    print(json.dumps({k:v for k,v in report.items() if k!="arms"}))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected phase coverage stage")
    main(bool(sys.argv[1:]))
