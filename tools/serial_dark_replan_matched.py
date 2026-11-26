"""Compare candidate width and stability under matched rollout counts."""
from collections import defaultdict
from itertools import product
import json
import os
import shutil
from statistics import mean
import sys
from time import perf_counter

from full_mode_runtime import FullModeRun, isolated, sha
from serial_dark_ordinary_learning import RULES, assert_layout_separation, run_native
from serial_dark_replanning import INPUTS, SOURCE, model_payload, parse, serialize, summarize
from serial_dynamic_reply import ROOT, build, require
from serial_formation_matrix import contrast, score_difference
from serial_worker import read, write_receipt


def prepare(scope):
    folder=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    for name in ("original.json","mixed.json","peer.json"):
        shutil.copyfile(SOURCE/name,folder/name)
    grid={stage:[(base+i,i%4,t,(16,28,36,48,60,84)[i%6],-1)
                 for i,t in product(range(count),(0,4))]
          for stage,base,count in (("development",24139000,12),("confirmation",24140000,36))}
    assert_layout_separation(grid)
    write_receipt(folder/"grid.json",grid)
    write_receipt(folder/"fingerprints.json",{key:sha(folder/key) for key in INPUTS})


def paired(roots,old,new,domain):
    groups=defaultdict(list)
    for (seed,_,_),row in roots.items():
        if domain and row.get("worlds")!=4:
            continue
        runs=row.get("runs",{})
        both=[]
        for (d,i,a),after in runs.items():
            if a!=new or bool(d)!=bool(domain):
                continue
            before=runs[d,i,old]
            both.append((0,0) if before==after else score_difference(before[0],after[0]))
        groups[seed].append(tuple(map(mean,zip(*both,strict=True))) if both else (0,0))
    return contrast([tuple(map(mean,zip(*rows,strict=True))) for rows in groups.values()]) if groups else None


def matched_run(scope,rows,models):
    began=perf_counter()
    replicas=4
    wire=model_payload(models)+[" ".join(map(str,(*r,replicas))) for r in rows]
    with isolated(scope,ROOT/"build"):
        raw=scope.run("dark-replanning-matched","\n".join(wire)+"\n")
    work={}
    ordinary=[]
    for line in raw.splitlines():
        if line.startswith("K "):
            s,p,t,n,w,short,wide=map(int,line.split()[1:])
            require((s,p,t) not in work and n==w and (not n or min(short,wide)>0),"unequal logical rollout counts")
            work[s,p,t]=[n,w,short,wide]
        else:
            ordinary.append(line)
    roots=parse("\n".join(ordinary),rows,replicas)
    require(work.keys()=={k for k,v in roots.items() if "moves" in v},"missing matched work counter")
    for k,v in work.items():
        require(bool(v[0])==(roots[k]["worlds"]==4) and sum(v[2:])==roots[k]["steps"],"work/coverage mismatch")
        roots[k]["matched_work"]=v
    played,_,_,_=run_native(scope,"X",[r[:3] for r in rows],(models["original"],models["mixed"]),peer=models["peer"])
    for r in played[::2]:
        root=roots[r["seed"],r["seat"],r["table"]]
        require("baseline" not in root or root["baseline"]==[r["result"],r["plies"],r["hash"]],"matched A changed")
    report=summarize(roots)
    report.update(matched_work=list(map(sum,zip(*work.values(),strict=True))),
                  heads={str(t):{f"{b}-{a}/{domain}":paired({k:v for k,v in roots.items() if k[2]==t},a,b,domain)
                                 for a,b in ((0,1),(0,2),(0,3),(1,2),(2,3)) for domain in (0,1)}
                         for t in (0,4)},seconds=perf_counter()-began)
    # The third search arm is a guard on the same evaluated cells, not a crossed factor.
    for group in ("original_boards","new_worlds"):
        report[group].pop("interaction",None)
    return roots,report


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-matched-replan"/os.environ["JUNQI_SERIAL_KEY"])
    if not audit:
        prepare(scope)
    folder=scope.folder
    require(all(sha(folder/n)==s for n,s in read(folder/"fingerprints.json").items()),"matched replanning inputs changed")
    stage="confirmation" if audit else "development"
    rows=[tuple(r) for r in read(folder/"grid.json")[stage]]
    models={n:read(folder/(n+".json")) for n in ("original","mixed","peer")}
    roots,report=matched_run(scope,rows,models)
    write_receipt(folder/(stage+".json"),serialize(roots,report))
    if audit:
        guarded=report["heads"]["4"]["3-0/0"]["interval"]
        witness=report["heads"]["4"]["3-0/1"]
        verdict=("supported" if guarded[0]>0 and witness and witness["interval"][0]>0 else
                 "refuted" if guarded[1]<0 else "inconclusive")
        print("JUNQI_RESEARCH "+verdict)
    print(json.dumps(report))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected matched replanning stage")
    main(bool(sys.argv[1:]))
