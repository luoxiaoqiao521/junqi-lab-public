"""Compare ordinary terminal training with a small, known-probability legal exploration."""
from collections import Counter
from itertools import product
import json
import os
from random import Random
import shutil
import sys
from time import perf_counter

from compare_book_learning import fit
from full_mode_runtime import FullModeRun, isolated, sha
from serial_dark_ordinary_learning import (RATE, RULES, UPDATES, assert_layout_separation, data_health,
    examples_xy, model_wire, parse, run_native, summarize)
from serial_dark_partner_support import SOURCE
from serial_dynamic_reply import ROOT, build, require
from serial_worker import read, write_receipt


BUDGET=22000
INPUTS=("parent.json","peer.json","grid.json","engine/deadline_opening.exe","engine/rules.json")
WEIGHTS=("ordinary.json","explore.json")


def prepare(scope):
    folder=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    shutil.copyfile(SOURCE/"original.json",folder/"parent.json")
    shutil.copyfile(SOURCE/"peer.json",folder/"peer.json")
    grids=dict(train=[(24162000+i,i%4,(i//4)%2) for i in range(192)],
               controls=[(24163000+2*i,i%4,(i//4)%2) for i in range(8)],
               confirmation=[(24164000+i,i%4,t) for i,t in product(range(48),(0,2,3))])
    assert_layout_separation(grids)
    write_receipt(folder/"grid.json",grids)
    write_receipt(folder/"fingerprints.json",{n:sha(folder/n) for n in INPUTS})


def collect(scope,grid,parent,explore,budget):
    start=perf_counter()
    wire=["Y",model_wire(parent),f"{budget} {int(explore)}",*(" ".join(map(str,r)) for r in grid)]
    with isolated(scope,ROOT/"build"):
        raw=scope.run("dark-ordinary-learning","\n".join(wire)+"\n")
    records,decisions=[],[]
    left=None
    for line in raw.splitlines():
        tag,*fields=line.split()
        if tag=="B":
            require(left is None,"duplicate ordinary budget")
            left=int(fields[0])
        elif tag=="J":
            seed,seat,table,ply,base,chosen,alternatives,num,den=map(int,fields)
            expected=((1 if chosen!=base else 7*alternatives,8*alternatives) if explore else (1,1))
            require(8<=ply<120 and 1<=alternatives<=3 and (num,den)==expected and (explore or base==chosen),
                    "exploratory propensity disagrees with the assigned behavior")
            decisions.append(dict(seed=seed,seat=seat,table=table,ply=ply,base=base,
                                  chosen=chosen,alternatives=alternatives,probability=[num,den]))
        else:
            records.append(line)
    used=sum(line.startswith("G ") for line in records)
    games,examples,_=parse("\n".join(records),"Y",grid[:used])
    steps=sum(r["plies"] for r in games)
    positions={(r["seed"],r["seat"],r["table"]):r["plies"] for r in games}
    require(left is not None and steps+left==budget and len(positions)==used and
            all(r["ply"]<positions[r["seed"],r["seat"],r["table"]] for r in decisions),
            "ordinary experience budget or decision ledger does not balance")
    health=data_health(games,examples)
    return dict(games=games,examples=examples,decisions=decisions,seconds=perf_counter()-start,
                steps=steps,left=left,changed=sum(r["base"]!=r["chosen"] for r in decisions),health=health)


def balance(records,n):
    rows=[r for r in records["examples"] if r["outcome"]>=0]
    rng=Random(71922)
    rng.shuffle(rows)
    return rows[:n]


def learn(scope,grid,parent):
    folder=scope.folder
    control=collect(scope,grid["controls"],parent,False,8*720)
    previous,examples,_,_=run_native(scope,"D",grid["controls"],(parent,))
    require(previous==control["games"] and examples==control["examples"],
            "zero-exploration collection did not reproduce the ordinary path")
    logs={n:collect(scope,grid["train"],parent,i==1,BUDGET) for i,n in enumerate(("ordinary","explore"))}
    require({r["steps"] for r in logs.values()}=={BUDGET} and logs["explore"]["changed"]>0,
            "generation needs equal actual steps and real exploration")
    n=min(sum(r["outcome"]>=0 for r in a["examples"]) for a in logs.values())
    require(n>100,"both courses need sufficient completed outcome labels")
    metrics={}
    for name,records in logs.items():
        data=balance(records,n)
        x,y=examples_xy(data,parent)
        learned=fit(x,y,52122,24,parent["models"]["collision"],updates=UPDATES,rate=RATE)
        info=dict(rows=len(x),selected_returns=Counter(r["outcome"] for r in data),updates=UPDATES,rate=RATE)
        model=dict(parent,models={"collision":learned},parent_sha256=sha(folder/"parent.json"),training=info)
        require(model_wire(model)!=model_wire(parent),"ordinary learning did not update weights")
        write_receipt(folder/(name+".json"),model)
        metrics[name]=dict(**info,steps=records["steps"],games=len(records["games"]),
            full_returns=Counter(r["result"] for r in records["games"]),
            eligible=len(records["decisions"]),exploration_changes=records["changed"],seconds=records["seconds"])
    write_receipt(folder/"experience.json",logs)
    write_receipt(folder/"trained.json",{n:sha(folder/n) for n in (*WEIGHTS,"experience.json")})
    return dict(zero_control_games=len(previous),zero_control_examples=len(examples),courses=metrics)


def evaluate(scope,grid,parent,peer):
    folder=scope.folder
    ordinary,explore=(read(folder/n) for n in WEIGHTS)
    require(model_wire(ordinary)!=model_wire(explore),"two courses produced the same weights")
    require(all(m["parent_sha256"]==sha(folder/"parent.json") for m in (ordinary,explore)),
            "comparison does not descend from the assigned ordinary parent")
    base,_,base_first,a_time=run_native(scope,"X",grid,(parent,ordinary),peer=peer)
    experiment,_,first,b_time=run_native(scope,"X",grid,(ordinary,explore),peer=peer)
    require(base[1::2]==[dict(r,arm=3) for r in experiment[::2]],
            "ordinary student did not reproduce on the same seeds and seats")
    write_receipt(folder/"games.json",dict(parent_vs_ordinary=base,ordinary_vs_exploration=experiment,
                  ordinary_first=base_first,exploration_first=first,runtime=[a_time,b_time]))
    return dict(ordinary_vs_parent=summarize(base,base_first),exploration_vs_ordinary=summarize(experiment,first))


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-ordinary-exploration"/os.environ["JUNQI_SERIAL_KEY"])
    folder=scope.folder
    if not audit:
        prepare(scope)
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"exploration inputs changed")
    grids={k:[tuple(r) for r in v] for k,v in read(folder/"grid.json").items()}
    parent=read(folder/"parent.json")
    if audit:
        require(all(sha(folder/n)==v for n,v in read(folder/"trained.json").items()),"saved courses changed")
        result=evaluate(scope,grids["confirmation"],parent,read(folder/"peer.json"))
        panel=result["exploration_vs_ordinary"]
        verdict="supported" if min(panel[t]["interval"][0] for t in (0,2,3))>0 else (
            "refuted" if panel[0]["interval"][1]<0 else "inconclusive")
        print("JUNQI_RESEARCH "+verdict)
    else:
        result=learn(scope,grids,parent)
    write_receipt(folder/("confirmation.json" if audit else "training.json"),result)
    print(json.dumps(result))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected ordinary exploration stage")
    main(bool(sys.argv[1:]))
