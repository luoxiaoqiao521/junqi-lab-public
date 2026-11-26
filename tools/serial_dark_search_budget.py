"""Compare fixed and adaptive spending on strict four-dark worlds."""
from hashlib import sha256
from itertools import product
import json
import os
import shutil
from statistics import mean
import sys
from time import perf_counter

from full_mode_runtime import FullModeRun,isolated,sha
from serial_dark_bomb_calibration import PREVIOUS as LADDER, SOURCE
from serial_dynamic_reply import ROOT,build,require
from serial_formation_matrix import contrast,score_difference
from serial_worker import read,write_receipt
from train_early_deadline_student import RULES,model_arguments

CALIBRATION=ROOT/"build/dark-bomb-calibration/calibrate-decision-relevant-dark-bomb-branches-00c98e19bee622cc6cc7"
ARMS=("fixed","worlds","replies","depth","focused")


def new_roots(blocked):
    rows=read(SOURCE/"probe.json")["rows"]
    pool=sorted((r for r in rows if r["format"]==".jgs" and r["result"] in (1,2,3) and r["mid"] and
                 not any(r[k] for k in ("prior_layout","prior_trajectory","stopped","notice"))),
                key=lambda r:sha256(("another-dark-budget-"+r["family"]).encode()).digest())
    picked=[]
    for r in pool:
        if r["family"] in blocked:
            continue
        blocked.add(r["family"])
        picked.append({k:r[k] for k in ("key","family","learner","mid","direction")})
        if len(picked)==8:
            break
    require(len(picked)==8,"eight independently held layouts are not available")
    return picked


def write_replays(folder,cohorts):
    keys={r["key"] for group in cohorts for r in group}
    tapes={}
    for line in (SOURCE/"games.txt").read_text(encoding="utf8").splitlines():
        kind,k,tape=line.split(maxsplit=2)
        if kind=="G" and k in keys:
            tapes[k]=tape
    require(tapes.keys()==keys,"dark search replay source missing")
    for i,cohort in enumerate(cohorts):
        (folder/f"replay-{i}.txt").write_text("\n".join(
            f"{r['key']} {r['learner']} {r['mid']-1} {r['direction']} {23100+100*i+j} {tapes[r['key']]}"
            for j,r in enumerate(cohort))+"\n",encoding="ascii")


def prepare(scope):
    folder=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    shutil.copyfile(CALIBRATION/"student.json",folder/"student.json")
    manifests=read(CALIBRATION/"manifest.json")
    active=set()
    for file in ("development.json","confirmation.json"):
        active.update(k for k,r in read(CALIBRATION/file)["roots"].items() if "truth" in r)
    previous={r["family"] for group in read(LADDER/"manifest.json") for r in group}
    cohorts=[[r for r in manifests if r["key"] in active],new_roots(previous|{r["family"] for r in manifests})]
    require(len(cohorts[0])==33,"unexpected number of two-sided prior roots")
    write_replays(folder,cohorts)
    write_receipt(folder/"manifest.json",cohorts)
    files=("engine/deadline_opening.exe","engine/rules.json","student.json","manifest.json","replay-0.txt","replay-1.txt")
    write_receipt(folder/"fingerprints.json",{name:sha(folder/name) for name in files})


def decode(text,cohort,arms=ARMS):
    registry={r["key"]:r for r in cohort}
    roots={}
    for tag,key,*fields in map(str.split,text.splitlines()):
        if tag=="M":
            roots[key]=dict(registry[key],skipped=fields[0])
        elif tag=="D":
            h,f,slots,nonbomb,bomb,nodes,cut,*moves=map(int,fields)
            require(cut==registry[key]["mid"]-1 and h in moves and f in moves,"frozen candidate list drift")
            roots[key]=dict(registry[key],history=h,focus=f,slots=slots,world_counts=[nonbomb,bomb],nodes=nodes,
                            candidates=moves,arms={},outcomes=[])
        elif tag=="K":
            arm,pivot=map(int,fields)
            roots[key]["arms"][arms[arm]]=dict(pivotal=bool(pivot),cells={},steps=0)
        elif tag=="B":
            arm,move,side,n,steps=map(int,fields[:5])
            values=list(map(float,fields[5:]))
            require(n>0 and 0<=values[1]<=values[0]+1e-12<=values[2]+1e-12 and values[2]<=n,
                    "invalid bound or search job")
            entry=roots[key]["arms"][arms[arm]]
            entry["cells"].setdefault(move,{})[side]=[n,steps,*values]
            entry["steps"]+=steps
        else:
            require(tag=="Y","unexpected dark search output")
            roots[key]["outcomes"].append(list(map(int,fields)))
    require(roots.keys()==registry.keys(),"lost decision roots")
    return roots


def finish_roots(roots,expect_breadth=True):
    for row in roots.values():
        if not row.get("arms"):
            continue
        require((("worlds" in row["arms"])==(expect_breadth and min(row["world_counts"])>=4)) and
                {tuple(r[:3]) for r in row["outcomes"]}==set(product(row["candidates"],range(4),range(4))),
                "stepped worlds or frozen outcomes were not fully measured")
        probability=2/row["slots"]
        order=[row["history"],*[m for m in row["candidates"] if m!=row["history"]]]
        for profile in row["arms"].values():
            require(set(profile["cells"])==set(order) and profile["steps"]<=192*len(order),
                    "equal rollout cap or fixed candidates violated")
            scores={}
            for m in order:
                parts=profile["cells"][m]
                scores[m]=[(1-probability)*parts[0][i]/parts[0][0]+probability*parts[1][i]/parts[1][0]
                           for i in range(2,5)]
            profile["choice"]=max(order,key=lambda m:scores[m][0])
            profile["values"]=scores
            profile["separated"]=any(all(m==alt or vals[1]>other[2] for alt,other in scores.items())
                                     for m,vals in scores.items())
    return [r for r in roots.values() if r.get("arms")]


def tally(rows,arms=ARMS):
    out={}
    for name in arms[1:]:
        supported=[r for r in rows if name in r["arms"]]
        if not supported:
            continue
        reference="fixed"
        tables={}
        for t in range(4):
            diffs=[]
            for r in supported:
                a,b=(r["arms"][i]["choice"] for i in (reference,name))
                values={(m,j):v for m,k,j,v,_ in r["outcomes"] if k==t}
                diffs.append(tuple(map(mean,zip(*[(0,0) if a==b else score_difference(values[a,i],values[b,i])
                                                 for i in range(4)],strict=True))))
            tables[t]=dict(**contrast(diffs),positive=sum(a>0 for a,b in diffs),negative=sum(b<0 for a,b in diffs))
        out[name]=dict(roots=len(supported),steps=sum(r["arms"][name]["steps"] for r in supported),
                       fixed_steps=sum(r["arms"][reference]["steps"] for r in supported),
                       changed=sum(r["arms"][name]["choice"]!=r["arms"][reference]["choice"] for r in supported),
                       separated=sum(r["arms"][name]["separated"] for r in supported),tables=tables)
    return out


def main(audit):
    folder=ROOT/"build/dark-search-budget"/os.environ["JUNQI_SERIAL_KEY"]
    scope=FullModeRun(folder)
    if not audit:
        prepare(scope)
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"dark budget inputs changed")
    i=int(audit)
    artifact=read(folder/"student.json")
    tape=model_arguments(artifact,"collision",artifact["encoding"])+"\n"+(folder/f"replay-{i}.txt").read_text(encoding="ascii")
    begin=perf_counter()
    with isolated(scope,ROOT/"build"):
        raw=scope.run("dark-search-budget",tape)
    roots=decode(raw,read(folder/"manifest.json")[i])
    active=finish_roots(roots)
    stats=dict(profile=tally(active),requested=len(roots),completed=len(active),
               pivotal=sum(r["arms"]["focused"]["pivotal"] for r in active),
               nodes=sum(r["nodes"] for r in roots.values() if "nodes" in r),
               validation_steps=sum(y[-1] for r in active for y in r["outcomes"]),
               validation_truncated=sum(y[-2]<0 for r in active for y in r["outcomes"]),
               seconds=perf_counter()-begin)
    write_receipt(folder/("confirmation.json" if audit else "development.json"),dict(roots=roots,summary=stats))
    if audit:
        reliable=any(all(c["interval"][0]>0 for c in p["tables"].values()) for p in stats["profile"].values())
        print("JUNQI_RESEARCH "+("supported" if reliable else "inconclusive"))
    print(json.dumps(stats))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no argument or audit")
    main(bool(sys.argv[1:]))
