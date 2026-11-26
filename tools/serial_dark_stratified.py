"""Compare generic, stratified and coverage-triggered allocation with bounded work."""
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
from serial_dark_response_distribution import effects, mean_value
from serial_dynamic_reply import ROOT, build, require
from serial_formation_matrix import contrast
from serial_worker import read, write_receipt


ARMS=("ordinary","stratified","adaptive")
FIELDS=("ply","status","width","pivot","rejected","nodes","isolation")


def prepare(scope):
    folder=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    for n in ("original.json","mixed.json","peer.json"):
        shutil.copyfile(SOURCE/n,folder/n)
    grid={name:[(start+i,i%4,(i//4%2)*4,(16,28,44,60)[(i//2)%4]) for i in range(count)]
          for name,start,count in (("development",24150000,16),("confirmation",24151000,48))}
    assert_layout_separation(grid)
    write_receipt(folder/"grid.json",grid)
    write_receipt(folder/"fingerprints.json",{n:sha(folder/n) for n in INPUTS})


def decode(raw,requests):
    roots={}
    row=None
    for line in raw.splitlines():
        kind,*fields=line.split()
        nums=list(map(int,fields))
        if kind=="F":
            root=tuple(nums[:3])
            require(root not in roots,"duplicate stratification root")
            row=dict(zip(FIELDS,nums[3:],strict=True),U={},T={},R={},G={},moves={})
            roots[root]=row
        elif kind=="S":
            a,move=nums
            require(a not in row["moves"],"duplicate candidate key")
            row["moves"][a]=move
        else:
            a,w,g,t,score,steps=nums
            key=(a,w,g,t)
            require(kind in ("U","T","R","G") and key not in row[kind] and score in (-1,0,1,2) and steps>0,"bad strata cell")
            row[kind][key]=(score,steps)
    require(roots.keys()=={r[:3] for r in requests},"unregistered strata root")
    return roots


def world_groups(rows):
    groups={}
    for _,w,g,_ in rows:
        require(groups.setdefault(w,g)==g and g in (0,1),"world changed strata")
    return groups


def allocations(root):
    u,t,r=(root[n] for n in ("U","T","R"))
    groups=world_groups(u)
    require(set(groups)==set(range(4)) and world_groups(t)==world_groups(r)=={0:0,1:0,2:1,3:1},"unbalanced strata record")
    initial=[(u,w) for w in (0,1)]
    have=Counter(groups[w] for w in (0,1))
    picks=[]
    for g in (0,1):
        picks.extend((t,2*g+w) for w in range(2-have[g]))
    return dict(ordinary=[(u,w) for w in range(4)],stratified=[(t,w) for w in range(4)],adaptive=initial+picks)


def estimate(root,cells):
    indices=set(root["moves"])
    data={a:[] for a in indices}
    for rows,w in cells:
        for (a,i,g,t),v in rows.items():
            if i==w:
                data[a].append((g,v))
    require(all(len(v)==8 for v in data.values()),"rollout caps differ by candidate")
    require(all(plies<=180 for pairs in data.values() for g,(_,plies) in pairs),"selection exceeded horizon")
    scores={a:mean_value([v for g,v in pairs]) for a,pairs in data.items()}
    best=max(sorted(indices),key=lambda a:scores[a][0])
    choice=best if scores[best][0]>scores[0][1] else 0
    return choice,data


def conditional(data,a,g):
    low,high=mean_value([v for tag,v in data[a] if tag==g])
    base_lo,base_hi=mean_value([v for tag,v in data[0] if tag==g])
    return (low-base_hi,high-base_lo)


def sign(bounds):
    return 1 if bounds[0]>1e-9 else -1 if bounds[1]<-1e-9 else 0


def assess(roots):
    counts=Counter()
    steps=Counter()
    scores={a:[] for a in ARMS}
    factual={a:[] for a in ARMS}
    for root in roots.values():
        if root["status"]!=3:
            for arm in ARMS:
                factual[arm].append((0,0))
            continue
        a_set=root["moves"].keys()
        require(a_set==set(range(root["width"])) and root["pivot"] in root["moves"].values(),"strata omitted the public pivot")
        ref={(a,w,g,t):v for (a,w,g,t),v in root["R"].items()}
        require(ref.keys()=={(a,w,w//2,t) for a,w,t in product(a_set,range(4),(0,5))},"missing held reference")
        reference={a:[(g,v) for (i,w,g,t),v in ref.items() if i==a] for a in a_set}
        ref_outcomes={(a,w,t,0):v for (a,w,g,t),v in ref.items()}
        actual=root["G"]
        root["chosen"]={}
        for arm,cells in allocations(root).items():
            picked,data=estimate(root,cells)
            root["chosen"][arm]=picked
            counts[arm+"_changed"]+=picked!=0
            steps[arm]+=sum(p for pairs in data.values() for _,(_,p) in pairs)
            counts[arm+"_truncated"]+=sum(s<0 for pairs in data.values() for _,(s,p) in pairs)
            for a in sorted(a_set-{0}):
                for g in (0,1):
                    if not any(tag==g for tag,v in data[a]):
                        counts[arm+"_missing_condition"]+=1
                        continue
                    predicted=sign(conditional(data,a,g))
                    truth=sign(conditional(reference,a,g))
                    counts[arm+("_same" if predicted and truth==predicted else
                                "_reverse" if predicted and truth==-predicted else "_unresolved")]+=1
            a_ref=effects(ref_outcomes,picked,0) if picked else (0,0)
            a_true=effects(actual,picked,0) if picked else (0,0)
            scores[arm].append(a_ref)
            factual[arm].append(a_true)
            counts[arm+"_true_total_win_to_loss"]+=sum(
                actual[0,w,t,g][0]==2 and value[0]==0 for (a,w,t,g),value in actual.items() if a==picked)
    return dict(assigned=len(roots),status=Counter(r["status"] for r in roots.values()),counts=counts,steps=steps,
                isolation=sum(r["isolation"] for r in roots.values()),rejected=sum(r["rejected"] for r in roots.values()),
                rollout_allocation_cap=1440*sum(r["width"] for r in roots.values() if r["status"]==3),
                reference={a:contrast(v) for a,v in scores.items() if v},
                factual={a:contrast(v) for a,v in factual.items() if v})


def run(scope,requests,models):
    start=perf_counter()
    roots={}
    for i in range(0,len(requests),8):
        rows=requests[i:i+8]
        wire=model_payload(models)+[" ".join(map(str,r)) for r in rows]
        with isolated(scope,ROOT/"build"):
            raw=scope.run("dark-stratified","\n".join(wire)+"\n")
        roots.update(decode(raw,rows))
    result=assess(roots)
    result["seconds"]=perf_counter()-start
    records=[[*k,{name:[(*i,*v) for i,v in value.items()] if name in ("U","T","R","G") else value
                  for name,value in root.items()}] for k,root in roots.items()]
    return dict(summary=result,roots=records)


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-stratified"/os.environ["JUNQI_SERIAL_KEY"])
    folder=scope.folder
    if not audit:
        prepare(scope)
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"stratification inputs changed")
    phase="confirmation" if audit else "development"
    grid=[tuple(row) for row in read(folder/"grid.json")[phase]]
    models={n:read(folder/(n+".json")) for n in ("original","peer")}
    result=run(scope,grid,models)
    write_receipt(folder/(phase+".json"),result)
    if audit:
        summary=result["summary"]
        scope_interval=summary["reference"].get("adaptive",{}).get("interval",[-1,1])
        real=summary["factual"]["adaptive"]["interval"]
        verdict=("supported" if min(scope_interval[0],real[0])>0 else
                 "refuted" if max(scope_interval[1],real[1])<0 else "inconclusive")
        print("JUNQI_RESEARCH "+verdict)
    print(json.dumps(result["summary"]))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected stratification stage")
    main(bool(sys.argv[1:]))
