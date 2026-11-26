"""Verify partner resource evidence and actual student learning on a proved local outcome."""
from collections import Counter
import json
from math import isfinite
import os
import shutil
import sys
from time import perf_counter

from compare_book_learning import fit, np
from full_mode_runtime import FullModeRun, isolated, sha
from serial_dark_broad_pairs import team_value
from serial_dark_ordinary_learning import RATE, RULES, UPDATES, assert_layout_separation, model_wire
from serial_dark_partner_support import SOURCE
from serial_dynamic_reply import ROOT, build, require
from serial_worker import read, write_receipt


INPUTS=("parent.json","grid.json","engine/rules.json","engine/deadline_opening.exe")


def prepare(scope):
    folder=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    shutil.copyfile(SOURCE/"original.json",folder/"parent.json")
    grids={phase:[(base+i,i%4,(i//4)%2,(24,48,72,96,120,144)[(i//8)%6])
                  for i in range(count)]
           for phase,base,count in (("train",24160000,112),("heldout",24161000,96))}
    assert_layout_separation(grids)
    write_receipt(folder/"grid.json",grids)
    write_receipt(folder/"fingerprints.json",{n:sha(folder/n) for n in INPUTS})


def parse(text,grid):
    rows=[]
    for tag,*fields in map(str.split,text.splitlines()):
        if tag=="F":
            seed,seat,table,ply,valid=map(int,fields)
            rows.append(dict(seed=seed,seat=seat,table=table,ply=ply,valid=valid,stock=[],actions=[]))
        elif tag=="S":
            rows[-1]["stock"].append(list(map(int,fields)))
        elif tag=="C":
            isolation,fallback,before,after,possible=map(int,fields)
            rows[-1].update(isolation=isolation,fallback=fallback,before=before,after=after,possible=possible)
        else:
            require(tag=="A","unknown partner diagnostic record")
            move,label,listed=map(int,fields[:3])
            before,after,*x=map(float,fields[3:])
            require(len(x)==44 and all(map(isfinite,[before,after,*x])) and label in (-1,0,1),
                    "invalid active student input or diagnostic")
            require(abs(21*x[34]-rows[-1]["possible"])<1e-8,
                    "student did not consume the counted possible engineer squares")
            rows[-1]["actions"].append(dict(move=move,label=label,listed=listed,before=before,after=after,x=x))
    require([(r["seed"],r["seat"],r["table"]) for r in rows]==[tuple(g[:3]) for g in grid],
            "opening assignments changed")
    for row in rows:
        require(bool(row["actions"])==bool(row["valid"]) and len(row["stock"])==4*row["valid"],
                "missing partner resources or legal moves")
        if row["valid"]:
            legal={a["move"] for a in row["actions"]}
            require(len(legal)==len(row["actions"]) and {row[n] for n in ("before","after","fallback")}<=legal,
                    "selected action is missing from the legal set")
    return rows


def run(scope,grid,parent,student):
    started=perf_counter()
    wire=[model_wire(parent),model_wire(student),*(" ".join(map(str,r)) for r in grid)]
    with isolated(scope,ROOT/"build"):
        raw=scope.run("dark-partner-evidence","\n".join(wire)+"\n")
    return parse(raw,grid),perf_counter()-started


def pairs(rows):
    for row in rows:
        wins=[a for a in row["actions"] if a["label"]==1]
        losses=[a for a in row["actions"] if a["label"]==0]
        if wins and losses:
            yield row,wins[0],losses[0]


def evaluate(rows,parent,student):
    counts=Counter()
    for row in rows:
        counts["openings"]+=1
        if not row["valid"]:
            continue
        counts["restored"]+=1
        counts["isolation"]+=row["isolation"]
        candidates={a["move"]:a for a in row["actions"]}
        counts["possible_positions_over_three"]+=row["possible"]>3
        for _,lo,hi,jlo,jhi,*_ in row["stock"]:
            counts["stock_intervals"]+=1
            counts["joint_interval_changed"]+=(lo,hi)!=(jlo,jhi)
        counts["legal"]+=len(candidates)
        counts["actual_choice_changed"]+=row["before"]!=row["after"]
        for name in ("before","after"):
            counts[name+"_known_loser_chosen"]+=candidates[row[name]]["label"]==0
            counts[name+"_known_survivor_chosen"]+=candidates[row[name]]["label"]==1
    chosen=list(pairs(rows))
    for row,a,b in chosen:
        counts["paired_roots"]+=1
        counts["same_features"]+=a["x"]==b["x"]
        counts["both_listed"]+=bool(a["listed"] and b["listed"])
        for n,m in (("before",parent),("after",student)):
            q=team_value(m,[a["x"],b["x"]])
            require(np.allclose(q,[a[n],b[n]],rtol=1e-10,atol=1e-12),
                    "Python training score differs from native selection")
            counts[n+"_correct"]+=a[n]>b[n]
    return dict(counts)


def learn(scope,rows,parent):
    usable=list(pairs(rows))
    require(len(usable)>=4,"insufficient proved same-root contact outcomes")
    raw=np.asarray([a["x"] for _,positive,negative in usable for a in (positive,negative)])
    x=np.clip((raw-parent["mean"])/parent["scale"],-6,6)
    y=np.full((len(raw),7),np.nan)
    y[:,:3]=np.eye(3)[[2,0]*len(usable)]
    initial=parent["models"]["collision"]
    child=fit(x,y,53122,24,initial,updates=UPDATES,rate=RATE)
    model=dict(parent,models={"collision":child},diagnostic="proved immediate collision outcome",
               updates=UPDATES,rows=len(raw))
    require(model_wire(model)!=model_wire(parent),"proved-label probe did not update weights")
    write_receipt(scope.folder/"student.json",model)
    write_receipt(scope.folder/"trained.json",{"student.json":sha(scope.folder/"student.json")})
    return model


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-partner-evidence"/os.environ["JUNQI_SERIAL_KEY"])
    folder=scope.folder
    if not audit:
        prepare(scope)
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"evidence input changed")
    grids=read(folder/"grid.json")
    parent=read(folder/"parent.json")
    if not audit:
        rows,_=run(scope,grids["train"],parent,parent)
        learn(scope,rows,parent)
    require(all(sha(folder/n)==v for n,v in read(folder/"trained.json").items()),"student file changed")
    student=read(folder/"student.json")
    phase="heldout" if audit else "train"
    rows,seconds=run(scope,grids[phase],parent,student)
    summary=evaluate(rows,parent,student)
    require(summary["restored"]>0 and summary.get("isolation",0)>0,"partner isolation was not tested")
    if audit:
        result="supported" if summary.get("after_correct",0)>summary.get("before_correct",0) else "inconclusive"
        print("JUNQI_RESEARCH "+result)
    write_receipt(folder/(phase+".json"),dict(summary=summary,rows=rows,seconds=seconds))
    print(json.dumps(dict(summary=summary,seconds=seconds)))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected partner evidence stage")
    main(bool(sys.argv[1:]))
