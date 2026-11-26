"""Separate world, continuation, and early-elimination effects under equal rollout counts."""
from collections import Counter
from itertools import product
import json
import os
import shutil
from statistics import mean
import sys
from time import perf_counter

from full_mode_runtime import FullModeRun, isolated, sha
from serial_dark_ordinary_learning import RULES, assert_layout_separation
from serial_dark_replanning import INPUTS, SOURCE, model_payload
from serial_dynamic_reply import ROOT, build, require
from serial_formation_matrix import contrast, score_bounds, score_difference
from serial_worker import read, write_receipt


FIELDS=("anchor","ply","legal","width","base","worlds","rejected","nodes","seat_control","isolation")
ARMS=("A","B","C","D")
ALLOCATIONS={
    "A":set(product((0,1),(0,2),(0,1))),
    "B":set(product(range(4),(0,2),(0,))),
    "C":set(product((0,1),(0,2,3,4),(0,))),
}
SCREEN=set(product((0,1),(0,2),(0,)))


def prepare(scope):
    folder=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    for n in ("original.json","mixed.json","peer.json"):
        shutil.copyfile(SOURCE/n,folder/n)
    grid={phase:[(start+i,i%4,4*(i//4%2),(16,28,44,60)[(i//2)%4])
                 for i in range(count)] for phase,start,count in (("development",24144000,12),("confirmation",24145000,32))}
    failures=[(r["seed"],r["seat"],r["table"],max(0,r["ply"]-4)) for r in read(SOURCE/"confirmation.json")["first"]
              if r["table"] in (0,4) and r["parent_result"]==2 and r["child_result"]==0 and r["ply"]>=4][:4]
    grid["earlier_losses"]=failures
    assert_layout_separation(grid)
    write_receipt(folder/"grid.json",grid)
    write_receipt(folder/"fingerprints.json",{n:sha(folder/n) for n in INPUTS})


def parse(raw,requests):
    roots={}
    root=None
    for tag,*fields in (line.split() for line in raw.splitlines()):
        data=list(map(int,fields))
        if tag=="F":
            key=tuple(data[:3])
            require(key not in roots,"duplicate distribution root")
            root=dict(zip(FIELDS,data[3:],strict=True),moves={},L={},D={},R={},G={})
            roots[key]=root
        elif tag=="S":
            a,move=data
            require(a not in root["moves"],"duplicate candidate index")
            root["moves"][a]=move
        else:
            a,w,t,i,score,plies=data
            cell=(a,w,t,i)
            require(tag in "LDRG" and cell not in root[tag] and score in (-1,0,1,2) and plies>0,"bad value cell")
            root[tag][cell]=(score,plies)
    require(set(roots)=={r[:3] for r in requests},"unexpected distribution root")
    return roots


def mean_value(rows):
    return tuple(map(mean,zip(*(score_bounds(v[0]) for v in rows),strict=True)))


def decisions(root,table):
    moves=root["moves"]
    count=root["width"]
    require(set(moves)==set(range(count)) and len(set(moves.values()))==count<=root["legal"],"incomplete legal portfolio")
    base=next(k for k,v in moves.items() if v==root["base"])
    l,d,r,g=(root[k] for k in ("L","D","R","G"))
    require(l.keys()=={(a,*w) for a in moves for w in set.union(*ALLOCATIONS.values())},"bad balanced cells")
    require(r.keys()==set(product(moves,range(4,8),(0,2,5),(60,61))) and
            g.keys()==set(product(moves,(-1,),(table,),range(62,66))),"missing unseen evaluation cell")
    survived=sorted(moves,key=lambda a:(a!=base,-sum(max(0,l[a,*x][0]) for x in SCREEN),a))[:max(1,count//2)]
    require(set(a for a,*_ in d)==set(survived) and len(d)==4*count,"early elimination spent the wrong budget")
    scores={arm:{a:[l[a,*ws] for ws in selected] for a in moves} for arm,selected in ALLOCATIONS.items()}
    scores["D"]={a:[l[a,*ws] for ws in SCREEN]+[cell for k,cell in d.items() if k[0]==a] for a in survived}
    choices={}
    costs={}
    for arm,cells in scores.items():
        limits={a:mean_value(vals) for a,vals in cells.items()}
        candidate=max(sorted(cells),key=lambda a:limits[a][0])
        choices[arm]=candidate if limits[candidate][0]>limits[base][1] else base
        costs[arm]=sum(v[1] for data in cells.values() for v in data)
    costs["D"]+=sum(l[a,*ws][1] for a in moves if a not in survived for ws in SCREEN)
    return base,choices,survived,costs


def effects(record,after,before):
    return tuple(map(mean,zip(*(score_difference(record[before,*ws][0],score)
                               for (a,*ws),(score,_) in record.items() if a==after),strict=True)))


def report(roots):
    labels=Counter()
    costs=Counter()
    comparisons={g:{a:[] for a in ARMS} for g in ("original","reference")}
    for (_,_,t),root in roots.items():
        if root["worlds"]!=8:
            for seq in comparisons["original"].values():
                seq.append((0,0))
            continue
        base,chosen,survived,work=decisions(root,t)
        ref=root["R"]
        widths={a:mean_value([x for (m,*_),x in ref.items() if m==a]) for a in root["moves"]}
        top=max(x[0] for x in widths.values())
        labels["reference_winner_eliminated"]+=max(widths[a][1] for a in survived)<top
        for arm,a in chosen.items():
            labels[arm+"_changed"]+=a!=base
            labels[arm+"_worse_than_base"]+=widths[a][1]<widths[base][0]
            labels[arm+"_better_than_base"]+=widths[a][0]>widths[base][1]
            labels[arm+"_better_third"]+=top>max(widths[a][1],widths[base][1])
            for group,rows in (("original",root["G"]),("reference",ref)):
                comparisons[group][arm].append(effects(rows,a,base) if a!=base else (0,0))
        directional_spread(root,base,labels)
        costs.update(work)
        root["chosen"]=chosen
    return dict(roots=len(roots),covered=sum(r["worlds"]==8 for r in roots.values()),
                isolation=sum(r["isolation"] for r in roots.values()),
                rejects=sum(r["rejected"] for r in roots.values()),seat_control=sum(r["seat_control"] for r in roots.values()),
                selection_rollouts=sum(8*r["width"] for r in roots.values() if r["worlds"]==8),
                reference_rollouts=sum(len(r["R"]) for r in roots.values()),costs=costs,cases=labels,
                comparisons={g:{a:contrast(v) for a,v in by_arm.items() if v} for g,by_arm in comparisons.items()})


def directional_spread(root,base,labels):
    ref=root["R"]
    flipped={k:False for k in ("world","ally","enemy")}
    for a in root["moves"]:
        if a==base:
            continue
        sign={}
        for w,t in product(range(4,8),(0,2,5)):
            lo,hi=tuple(map(mean,zip(*(score_difference(ref[base,w,t,i][0],ref[a,w,t,i][0])
                                      for i in (60,61)),strict=True)))
            sign[w,t]=1 if lo>0 else -1 if hi<0 else 0
        flipped["world"]|=any({sign[w,t] for w in range(4,8)}>={-1,1} for t in (0,2,5))
        for name,t in (("ally",2),("enemy",5)):
            flipped[name]|=any(sign[w,0]*sign[w,t]==-1 for w in range(4,8))
    labels.update({name+"_flip":int(v) for name,v in flipped.items()})


def run(scope,rows,models):
    start=perf_counter()
    roots={}
    for pos in range(0,len(rows),6):
        chunk=rows[pos:pos+6]
        wire=model_payload(models)+[" ".join(map(str,r)) for r in chunk]
        with isolated(scope,ROOT/"build"):
            raw=scope.run("dark-response-distribution","\n".join(wire)+"\n")
        roots.update(parse(raw,chunk))
    summary=report(roots)
    summary["seconds"]=perf_counter()-start
    cases=[]
    for key,root in roots.items():
        cases.append([*key,{k:[(*x,*v) for x,v in data.items()] if k in ("L","D","R","G") else data
                            for k,data in root.items()}])
    return dict(summary=summary,roots=cases)


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-response-distribution"/os.environ["JUNQI_SERIAL_KEY"])
    if not audit:
        prepare(scope)
    folder=scope.folder
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"distribution inputs changed")
    models={n:read(folder/(n+".json")) for n in ("original","peer")}
    grid={k:[tuple(r) for r in rows] for k,rows in read(folder/"grid.json").items()}
    stage="confirmation" if audit else "development"
    result=run(scope,grid[stage],models)
    if not audit:
        write_receipt(folder/"earlier-losses.json",run(scope,grid["earlier_losses"],models))
    else:
        require(result["summary"]["isolation"]>0,"no blind layout substitution control")
        outcomes=result["summary"]["comparisons"]
        same=outcomes["original"]["C"]["interval"]
        independent=outcomes["reference"]["C"]["interval"]
        verdict=("supported" if same[0]>0 and independent[0]>0 else
                 "refuted" if same[1]<0 or independent[1]<0 else "inconclusive")
        print("JUNQI_RESEARCH "+verdict)
    write_receipt(folder/(stage+".json"),result)
    print(json.dumps(result["summary"]))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected response distribution stage")
    main(bool(sys.argv[1:]))
