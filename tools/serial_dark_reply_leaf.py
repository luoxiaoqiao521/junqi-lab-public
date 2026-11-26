"""Cross a single lawful enemy reply with an independent continuation of the same checkpoint."""
from collections import Counter
from hashlib import sha256
from itertools import product
import json
from math import isfinite
import os
from pathlib import Path
import shutil
from statistics import mean
import subprocess
import sys
from time import perf_counter

from audit_human_timing import sanitized_tape
from audit_qq_archive import INPUT, public_records
from full_mode_runtime import FullModeRun, isolated, sha
from serial_dark_formation_library import known_families
from serial_dark_formation_sources import PREVIOUS, ROOT
from serial_dark_search_budget import CALIBRATION
from serial_dynamic_reply import build, require
from serial_formation_matrix import contrast, score_difference
from serial_replay_intake import hashes, jgs
from serial_replay_phase import INTAKE
from serial_worker import read, write_receipt
from train_early_deadline_student import RULES, model_arguments


WORLD_STUDY=ROOT/"build/dark-formation-sources/compare-dark-formation-world-sources-570435cf4ad43d56fe34"
ANCHORS={"266df0449c6a":(7031,7026),"591bfd788c08":(530,4149)}
ARMS=("natural_leaf","natural_final","pressure_leaf","pressure_final")


def archived_holdout(scope,blocked):
    candidates={}
    for raw in public_records(Path(read(INPUT)["corpus"])):
        try:
            starting,moves,events=jgs(raw)
            if len(moves)<96:
                continue
            family,_=hashes(starting,moves)
        except ValueError:
            continue
        if family not in blocked and family not in candidates:
            candidates[family]=sanitized_tape(raw,events)
    selected=sorted(candidates,key=lambda f:sha256(("dark-reply-holdout:"+f).encode()).digest())[:160]
    request="\n".join(f"G {k} {candidates[k]}" for k in selected)+"\n"
    probe=subprocess.run([INTAKE/"replay_intake_probe.exe",scope.rules],input=request,encoding="ascii",
                         capture_output=True,check=True,timeout=45)
    accepted=[]
    for family,*fields in map(str.split,probe.stdout.splitlines()):
        direction,_,_,mid,_,notice,stopped,result,_,learner=map(int,fields)
        if direction>=0 and mid>1 and not (notice or stopped) and result in (1,2,3):
            accepted.append(dict(key=family,family=family,mid=mid,learner=learner,direction=direction))
    require(len(accepted)>=12,"fewer than twelve strict held-out archive families")
    return accepted[:12],candidates,dict(checked=len(selected),strict=len(accepted))


def prepare(scope):
    folder=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    shutil.copyfile(CALIBRATION/"student.json",folder/"student.json")
    earlier=[r for path in (PREVIOUS,WORLD_STUDY) for cohort in read(path/"manifest.json")["cohorts"] for r in cohort]
    anchors=[r for r in earlier if r["key"][:12] in ANCHORS]
    require(len(anchors)==2,"disputed archived decisions are missing")
    pilot=anchors+read(PREVIOUS/"manifest.json")["cohorts"][0][:10]
    held,bank,coverage=archived_holdout(scope,known_families()|{r["family"] for r in earlier})
    keys={r["key"] for r in pilot}
    archive={}
    for row in (INTAKE/"games.txt").read_text(encoding="utf8").splitlines():
        tag,key,tape=row.split(maxsplit=2)
        if tag=="G" and key in keys:
            archive[key]=tape
    require(archive.keys()==keys,"missing development tape")
    for fold,(cohort,source) in enumerate(((pilot,archive),(held,bank))):
        with (folder/f"replay-{fold}.txt").open("w",encoding="ascii") as sink:
            for n,r in enumerate(cohort):
                pair=ANCHORS.get(r["key"][:12],(-1,-1))
                print(r["key"],r["learner"],r["mid"]-1,r["direction"],877000+100*fold+n,*pair,source[r["key"]],file=sink)
    write_receipt(folder/"manifest.json",dict(cohorts=[pilot,held],holdout_preflight=coverage))
    paths=("engine/deadline_opening.exe","engine/rules.json","student.json","manifest.json","replay-0.txt","replay-1.txt")
    write_receipt(folder/"fingerprints.json",{p:sha(folder/p) for p in paths})


def parse(raw,cohort):
    roots={}
    for tag,key,*fields in map(str.split,raw.splitlines()):
        if tag=="M":
            roots[key]=dict(skipped=fields[0],details=list(map(int,fields[1:])))
        elif tag=="D":
            a,b,slots,nodes,micros=map(int,fields)
            require(a!=b and slots>2,"invalid frozen moves or focal prior")
            roots[key]=dict(moves=(a,b),slots=slots,nodes=nodes,micros=micros,cells={},referee={})
        elif tag=="C":
            side,table,arm,move,enemy,natural,reply,ally,plies,outcome=map(int,fields[:10])
            leaf=float(fields[10])
            trace,final_a,extra_a,final_b,extra_b=map(int,fields[11:])
            require(0<=leaf<=1 and 0<plies<=48 and all(-1<=v<=2 for v in (outcome,final_a,final_b)) and
                    plies+max(extra_a,extra_b)<=400 and (outcome<0 or outcome==final_a==final_b),"bad checkpoint or final")
            cells=roots[key]["cells"]
            index=(side,table,arm,move)
            require(index not in cells,"duplicate checkpoint")
            cells[index]=dict(enemy=enemy,natural=natural,reply=reply,ally=ally,plies=plies,outcome=outcome,leaf=leaf,
                              trace=trace,final=(final_a,final_b),extra=(extra_a,extra_b))
        elif tag=="I":
            roots[key]["isolation"]=list(map(int,fields))
        elif tag=="X":
            move,x=int(fields[0]),list(map(float,fields[1:]))
            features=roots[key].setdefault("features",{})
            require(move not in features and len(x)==44 and all(map(isfinite,x)),"invalid public course features")
            features[move]=x
        else:
            require(tag=="Y","unknown audit output")
            table,move,stream,score,plies=map(int,fields)
            roots[key]["referee"][table,move,stream]=(score,plies)
    require(roots.keys()=={r["key"] for r in cohort},"replay coverage changed")
    for row in roots.values():
        if "skipped" in row:
            continue
        cells=row["cells"]
        require("features" not in row or row["features"].keys()==set(row["moves"]),"features do not match both moves")
        require(cells.keys()==set(product(range(2),range(4),range(2),row["moves"])) and
                row["referee"].keys()==set(product(range(4),row["moves"],range(2))),"incomplete paired study")
        for s,t,m in product(range(2),range(4),row["moves"]):
            a,b=(cells[s,t,k,m] for k in (0,1))
            require(a["reply"]!=b["reply"] or a==b,"same-reply placebo changed a checkpoint or final")
    return roots


def rank_delta(row,table,pressure,use_final,p):
    old,new=row["moves"]
    parts=[]
    for side in range(2):
        a,b=(row["cells"][side,table,pressure,m] for m in (old,new))
        diff=(tuple(map(mean,zip(*(score_difference(x,y) for x,y in zip(a["final"],b["final"],strict=True)),strict=True)))
              if use_final else (b["leaf"]-a["leaf"],)*2)
        parts.append(diff)
    return [other*(1-p)+bomb*p for other,bomb in zip(*parts,strict=True)]


def pick(delta):
    if delta[0]>1e-8:
        return 1
    if delta[1]<-1e-8 or max(map(abs,delta))<=1e-8:
        return 0
    return None


def explain(row):
    old,new=row["moves"]
    p=2/row["slots"]
    detail={}
    for table in range(4):
        deltas={name:rank_delta(row,table,t//2,t%2,p) for t,name in enumerate(ARMS)}
        ref=tuple(map(mean,zip(*(score_difference(row["referee"][table,old,i][0],row["referee"][table,new,i][0])
                                 for i in range(2)),strict=True)))
        chosen={a:pick(d) for a,d in deltas.items()}
        sensitivity={name:dict(bomb=rank_delta(row,table,t//2,t%2,1),other=rank_delta(row,table,t//2,t%2,0))
                     for t,name in enumerate(ARMS)}
        detail[table]=dict(delta=deltas,choices=chosen,referee=ref,sensitivity=sensitivity,
                           leaf_disagrees=[chosen[ARMS[i]]!=chosen[ARMS[i+1]] for i in (0,2)],
                           pressure_changed=sum(row["cells"][s,table,1,m]["reply"]!=row["cells"][s,table,0,m]["reply"]
                                                for s,m in product(range(2),row["moves"])))
    return detail


def compare(row,table,arm):
    d=row["tables"][table]
    a,b=(d["choices"][name] for name in (ARMS[0],arm))
    if a==b or b is None:
        return (0,0)
    lo,hi=d["referee"]
    return (lo,hi) if b==1 else (-hi,-lo)


def summarize(roots):
    active=[r for r in roots.values() if "cells" in r]
    all_cells=[c for r in active for c in r["cells"].values()]
    panel={}
    for t in range(4):
        if not active:
            break
        panel[t]={a:dict(**contrast([compare(r,t,a) for r in active]),
                         changed=sum(r["tables"][t]["choices"][a] not in (None,r["tables"][t]["choices"][ARMS[0]])
                                     for r in active)) for a in ARMS[1:]}
    return dict(registered=len(roots),completed=len(active),skips=Counter(r["skipped"] for r in roots.values() if "skipped" in r),
                cells=len(all_cells),changed_replies=sum(t["pressure_changed"] for r in active for t in r["tables"].values()),
                ended_before_checkpoint=sum(c["outcome"]>=0 for c in all_cells),
                short_steps=sum(c["plies"] for c in all_cells),final_steps=sum(sum(c["extra"]) for c in all_cells),
                final_truncated=sum(v<0 for c in all_cells for v in c["final"]),
                referee_truncated=sum(v<0 for r in active for v,_ in r["referee"].values()),
                world_nodes=sum(r["nodes"] for r in active),world_micros=sum(r["micros"] for r in active),
                leaf_disagreement=[sum(t["leaf_disagrees"][i] for r in active for t in r["tables"].values()) for i in (0,1)],
                heldout_action_scores=panel)


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-reply-leaf"/os.environ["JUNQI_SERIAL_KEY"])
    folder=scope.folder
    if not audit:
        prepare(scope)
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"reply/leaf experiment changed inputs")
    model=read(folder/"student.json")
    fold=int(audit)
    tape=model_arguments(model,"collision",model["encoding"])+"\n"+(folder/f"replay-{fold}.txt").read_text(encoding="ascii")
    began=perf_counter()
    with isolated(scope,ROOT/"build"):
        raw=scope.run("dark-reply-leaf",tape)
    roots=parse(raw,read(folder/"manifest.json")["cohorts"][fold])
    for row in roots.values():
        if "cells" in row:
            row["tables"]=explain(row)
    summary=summarize(roots)
    summary["seconds"]=perf_counter()-began
    for row in roots.values():
        for name in ("cells","referee"):
            if name in row:
                row[name]={"/".join(map(str,k)):v for k,v in row[name].items()}
    write_receipt(folder/("confirmation.json" if audit else "development.json"),dict(roots=roots,summary=summary))
    if audit:
        panel=summary["heldout_action_scores"]
        supported=bool(panel) and any(all(r[a]["interval"][0]>0 for r in panel.values()) for a in ARMS[1:])
        print("JUNQI_RESEARCH "+("supported" if supported else "inconclusive"))
    print(json.dumps(summary))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no argument or audit")
    main(bool(sys.argv[1:]))
