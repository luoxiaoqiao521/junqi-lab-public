"""Compare equal-cap short forecasts against held-out four-dark continuations."""
from collections import Counter
import json
import math
import os
from statistics import mean
import subprocess
import sys
from time import perf_counter

from measure_deadline_opening import ROOT, RULES, SCORE, build, file_sha, interval, model_roster, require
from serial_worker import write_receipt

START, COUNT = 1203600, 12
PROFILES = ("fixed9", "fixed12", "conditional")


def action(fields):
    require(len(fields)==21,"incomplete action profiles or references")
    results = {}
    for j,name in enumerate(PROFILES):
        value,plies,extended,reasons=fields[1+4*j:5+4*j]
        value=float(value)
        plies,extended,reasons=map(int,(plies,extended,reasons))
        require(math.isfinite(value) and 0<plies<=36 and 0<=extended<=3 and 0<=reasons<=7 and
                (name=="conditional" or not extended) and bool(extended)==bool(reasons),"invalid metering or budget")
        results[name]=dict(value=value,plies=plies,extended=extended,reasons=reasons)
    reference=[list(map(int,fields[i:i+2])) for i in range(13,21,2)]
    require(all(score in SCORE and n>0 for score,n in reference),"invalid matched continuation endpoint")
    return dict(move=int(fields[0]),responses=results,reference=reference)


def direction(gap):
    return int(gap>1e-9)-int(gap<-1e-9)


def independent_reference(a,b):
    outcomes=[(SCORE[x[0]],SCORE[y[0]]) for x,y in zip(a["reference"],b["reference"],strict=True)]
    if any(a is None or b is None for a,b in outcomes):
        return "truncated", None
    signs=Counter(direction(a-b) for a,b in outcomes)
    if signs[1] and signs[-1]:
        return "sensitive", None
    if signs[0]==4:
        return "sample_tie", None
    sign=1 if signs[1] else -1
    return ("one_decisive",None) if signs[sign]==1 else ("supported",sign)


def decode(raw):
    lines=raw.splitlines()
    require(lines[0]=="controls 4","public cutoff or visibility controls failed")
    rows=[]
    for line in lines[1:]:
        fields=line.split()
        seed,seat,table,phase,ply,worlds=map(int,fields[:6])
        require(-1<=worlds<=8 and (worlds!=-1 or ply==0),"bad snapshot disposition")
        row=dict(seed=seed,seat=seat,table=table,phase=phase,ply=ply,worlds=worlds)
        if worlds==8:
            a,b=action(fields[6:27]),action(fields[27:])
            require(a["move"]!=b["move"],"same action in competing profiles")
            status,expected=independent_reference(a,b)
            chosen={p:direction(a["responses"][p]["value"]-b["responses"][p]["value"]) for p in PROFILES}
            row.update(actions=[a,b],reference=status,expected=expected,prediction=chosen)
        else:
            require(len(fields)==6,"unexpected profiles for unscored root")
        rows.append(row)
    expected=sorted((s,(s+j)%4,t,p) for s in range(START,START+COUNT) for j in (0,1) for t in (0,1) for p in (0,1))
    require(sorted((r["seed"],r["seat"],r["table"],r["phase"]) for r in rows)==expected,"missing predeclared root slots")
    return rows


def summarize(rows):
    scored=[r for r in rows if r["worlds"]==8]
    supported=[r for r in scored if r["expected"] is not None]
    totals={}
    for p in PROFILES:
        forecasts=[a["responses"][p] for r in scored for a in r["actions"]]
        totals[p]=dict(correct=sum(r["prediction"][p]==r["expected"] for r in supported),
                       unknown_tie=sum(r["prediction"][p]==0 for r in supported),
                       reversed=sum(r["prediction"][p]==-r["expected"] for r in supported),
                       plies=sum(f["plies"] for f in forecasts),extended=sum(f["extended"] for f in forecasts),
                       reasons=dict(Counter(str(f["reasons"]) for f in forecasts if f["reasons"])))
    per_layout={}
    for r in supported:
        delta=int(r["prediction"]["conditional"]==r["expected"])-int(r["prediction"]["fixed9"]==r["expected"])
        per_layout.setdefault(r["seed"],[]).append(delta)
    deltas=[mean(v) for v in per_layout.values()]
    changed=[r for r in scored if r["prediction"]["conditional"]!=r["prediction"]["fixed9"]]
    return dict(planned=len(rows),scored=len(scored),scored_layouts=len({r["seed"] for r in scored}),
                dispositions=dict(Counter(r.get("reference",str(r["worlds"])) for r in rows)),profiles=totals,
                ref_layouts=len(deltas),paired_accuracy_delta=dict(mean=mean(deltas),grouped_95=interval(deltas)) if deltas else None,
                different_order=len(changed),different_order_unresolved=sum(r["expected"] is None for r in changed),
                after_first_exit=sum(r["phase"] for r in scored))


def main(audit):
    path=ROOT/"build/blind-extension"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        path.mkdir(parents=True,exist_ok=False)
        build(path)
    exe=path/"deadline_opening.exe"
    roster=model_roster()
    weights=[w for name in ("candidate","historical") for w in roster[name]["weights"]]
    payload=" ".join(map(str,(START,COUNT,*weights)))
    start=perf_counter()
    run=subprocess.run([exe,RULES,"blind-extension"],cwd=ROOT,input=payload,encoding="utf8",capture_output=True,timeout=115)
    print(f"four-dark comparison seconds={perf_counter()-start:.2f}")
    require(run.returncode==0,f"blind probe failed: {run.stderr.strip()}")
    rows=decode(run.stdout)
    result=dict(engine=file_sha(exe),rules=file_sha(RULES),summary=summarize(rows),rows=rows)
    if audit:
        require(run.stdout==(path/"blind.txt").read_text(encoding="utf8") and
                result==json.loads((path/"blind.json").read_text(encoding="utf8")),"blind comparison did not replay")
    else:
        (path/"blind.txt").write_text(run.stdout,encoding="utf8")
        write_receipt(path/"blind.json",result)
    print(json.dumps(result["summary"],indent=2))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
