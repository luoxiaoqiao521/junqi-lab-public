"""Cross candidate width and terminal rollout at the first decision after fixed natural responses."""
from collections import Counter, defaultdict
from itertools import product
import json
import os
import shutil
from statistics import mean
import sys
from time import perf_counter

from full_mode_runtime import FullModeRun, isolated, sha
from serial_dark_ordinary_learning import RULES, assert_layout_separation, model_wire, run_native
from serial_dark_partner_support import SOURCE, parse as parse_support
from serial_dynamic_reply import ROOT, build, require
from serial_formation_matrix import contrast, score_difference
from serial_worker import read, write_receipt
from train_early_deadline_student import model_arguments


CONDITIONS=ROOT/"build/dark-partner-conditions/validate-four-dark-public-recapture-conditions-2a2cb9436f1c3725c102"
INPUTS=("original.json","mixed.json","peer.json","grid.json","engine/deadline_opening.exe","engine/rules.json")
COUNTERS=("anchor","ply","first","legal","short","wide","worlds","solved","rejected","duplicates","nodes","steps","isolation")
REASONS=("seat-control","collision","flag-event","route","actor-history","other")
ARMS=("A","B","C","D")
ORIGINAL_FORKS=(24136025,24136026)


def prepare(scope):
    folder=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    for name in ("original.json","mixed.json","peer.json"):
        shutil.copyfile(SOURCE/name,folder/name)
    prior=[(s,p,t,f) for s,p,t,f in read(CONDITIONS/"confirmation.json")["forks"] if s in ORIGINAL_FORKS and t==0]
    require(len(prior)==2,"known sacrifice and removal cases are absent")
    write_receipt(folder/"anchors.json",prior)
    grids={"historical":[(s,p,t,f["cut"],f["actions"][1]) for s,p,t,f in prior]}
    for stage,base,count in (("development",24137000,12),("confirmation",24138000,36)):
        grids[stage]=[(base+i,i%4,t,(12,24,40,56,76,96)[i%6],-1) for i,t in product(range(count),(0,2))]
    assert_layout_separation(grids)
    write_receipt(folder/"grid.json",grids)
    write_receipt(folder/"fingerprints.json",{n:sha(folder/n) for n in (*INPUTS,"anchors.json")})


def model_payload(models):
    return [model_wire(models["original"]),model_arguments(models["peer"],"small_nn")]


def verify_anchors(scope,models):
    expected=read(scope.folder/"anchors.json")
    grid=[(s,p,t) for s,p,t,f in expected]
    wire=[model_wire(models["original"]),model_wire(models["mixed"]),model_arguments(models["peer"],"small_nn")]
    wire+=[" ".join(map(str,(*r,4))) for r in grid]
    with isolated(scope,ROOT/"build"):
        raw=scope.run("dark-partner-support","\n".join(wire)+"\n")
    _,found=parse_support(raw,grid,4)
    for s,p,t,f in expected:
        root=found[s,p,t]
        require(all(root[k]==f[k] for k in ("cut","actions","evidence")) and
                [list((*k,*v)) for k,v in root["cells"].items()]==f["cells"],"historical counterfactual changed")


def parse(raw,grid,replicas):
    roots={}
    for tag,s,p,t,*tail in map(str.split,raw.splitlines()):
        key=tuple(map(int,(s,p,t)))
        fields=list(map(int,tail))
        if tag=="M":
            require(key not in roots and len(fields)==3,"invalid unreachable root")
            roots[key]=dict(skipped=fields)
        elif tag=="F":
            n=len(COUNTERS)
            row=dict(zip(COUNTERS,fields[:n],strict=True))
            require(key not in roots and row["anchor"]<row["ply"] and 1<=row["short"]<=row["wide"]<=row["legal"],
                    "invalid root and legal candidates")
            row.update(failures=dict(zip(REASONS,fields[n:n+6],strict=True)),moves=fields[n+6:n+10],
                       baseline=fields[n+10:],runs={})
            require(len(row["baseline"])==3 and sum(row["failures"].values())==row["rejected"] and
                    row["worlds"]+row["duplicates"]+row["rejected"]==row["solved"],"world counters differ")
            require(row["worlds"]==4 or row["moves"][:2]==row["moves"][2:],"coverage fallback changed selection")
            roots[key]=row
        else:
            d,i,a,result,steps,trace=fields
            require(tag=="R" and (d,i,a) not in roots[key]["runs"] and result in (-1,0,1,2) and steps>0,"bad paired run")
            roots[key]["runs"][d,i,a]=(result,steps,trace)
    require(roots.keys()=={r[:3] for r in grid},"unregistered replan root")
    for key,row in roots.items():
        if "runs" not in row:
            continue
        domains=range(3 if row["worlds"]==4 else 1)
        cells=row["runs"]
        require(cells.keys()==set(product(domains,range(replicas),range(4))),"missing replan branch")
        for d,i in product(domains,range(replicas)):
            results={}
            for a in range(4):
                move=row["moves"][a]
                require(results.setdefault(move,cells[d,i,a])==cells[d,i,a],"same first move diverged on same stream")
    return roots


def statistic(roots,domain,covered=False):
    groups=defaultdict(list)
    for (seed,_,_),row in roots.items():
        if covered and row.get("worlds")!=4:
            continue
        runs=row.get("runs",{})
        table=defaultdict(list)
        for (d,i,a),after in runs.items():
            if a==0 or (domain==0)!=(d==0):
                continue
            before=runs[d,i,0]
            delta=(0,0) if before==after else score_difference(before[0],after[0])
            table[ARMS[a]].append(delta)
        effects={arm:tuple(map(mean,zip(*values,strict=True))) for arm,values in table.items()}
        if not effects:
            effects=dict.fromkeys(ARMS[1:],(0,0))
        b,c,d=(effects[a] for a in ARMS[1:])
        effects["interaction"]=(d[0]-b[1]-c[1],d[1]-b[0]-c[0])
        groups[seed].append(effects)
    if not groups:
        return {}
    return {arm:contrast([tuple(map(mean,zip(*(r[arm] for r in rows),strict=True))) for rows in groups.values()])
            for arm in (*ARMS[1:],"interaction")}


def summarize(roots):
    reached=[row for row in roots.values() if "runs" in row]
    failures=Counter()
    for row in reached:
        failures.update(row["failures"])
    return dict(assigned=len(roots),reached=len(reached),world_counts=dict(Counter(r["worlds"] for r in reached)),
                skipped=dict(Counter(r["skipped"][2] for r in roots.values() if "skipped" in r)),
                world_rejections=failures,work={n:sum(r[n] for r in reached) for n in ("nodes","steps","isolation")},
                changed={name:sum(r["moves"][i]!=r["moves"][0] for r in reached) for i,name in enumerate(ARMS)},
                original_boards=statistic(roots,0),new_worlds=statistic(roots,1,True),
                continuation_runs=sum(len(r["runs"]) for r in reached))


def run(scope,rows,models,replicas):
    started=perf_counter()
    wire=model_payload(models)+[" ".join(map(str,(*r,replicas))) for r in rows]
    with isolated(scope,ROOT/"build"):
        raw=scope.run("dark-replanning","\n".join(wire)+"\n")
    roots=parse(raw,rows,replicas)
    info=summarize(roots)
    info["seconds"]=perf_counter()-started
    return roots,info


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-replanning"/os.environ["JUNQI_SERIAL_KEY"])
    if not audit:
        prepare(scope)
    folder=scope.folder
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"replan input changed")
    models={n:read(folder/(n+".json")) for n in ("original","mixed","peer")}
    grid={k:[tuple(r) for r in v] for k,v in read(folder/"grid.json").items()}
    if not audit:
        verify_anchors(scope,models)
        roots,info=run(scope,grid["historical"],models,4)
        write_receipt(folder/"historical.json",serialize(roots,info))
    stage="confirmation" if audit else "development"
    roots,info=run(scope,grid[stage],models,4)
    played,_,_,_=run_native(scope,"X",[r[:3] for r in grid[stage]],(models["original"],models["mixed"]),peer=models["peer"])
    for r in played[::2]:
        observed=roots[r["seed"],r["seat"],r["table"]]
        require("baseline" not in observed or observed["baseline"]==[r["result"],r["plies"],r["hash"]],
                "A did not reproduce the ordinary full game")
    if audit:
        require(info["work"]["isolation"]>0,"no valid alternate-order information isolation control")
        low,high=info["original_boards"]["D"]["interval"]
        reference=info["new_worlds"]["D"]["interval"]
        verdict=("supported" if low>0 and reference[0]>0 else "refuted" if high<0 or reference[1]<0 else "inconclusive")
        print("JUNQI_RESEARCH "+verdict)
    write_receipt(folder/(stage+".json"),serialize(roots,info))
    print(json.dumps(info))
    print("JUNQI_VERDICT accepted")


def serialize(roots,info):
    return dict(summary=info,roots=[[*key,{**row,"runs":[(*r,*v) for r,v in row.get("runs",{}).items()]}]
                                    for key,row in roots.items()])


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected replanning stage")
    main(bool(sys.argv[1:]))
