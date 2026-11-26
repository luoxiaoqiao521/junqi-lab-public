"""Check recoverable errors against the student's actual no-teacher choice."""
from collections import Counter
import json
import os
from statistics import mean
import subprocess
import sys
from time import perf_counter

from serial_conversion_labels import MODEL, ROOT, RULES, bounds, build, decode, file_sha, interval, model_arguments, model_roster, require
from serial_worker import write_receipt

LOCKED=(1203855,1)
FRESH=(1203950,16)
TABLES=tuple((p,t) for p in (1,3) for t in (0,1))
REFERENCE=ROOT/"build/conversion-label-transfer/confirm-conversion-label-transfer-3d143702bcde16b6bf67/transfer.json"


def compare_student(root, arm):
    pairs=root["pairs"]
    cells={f"{p}/{t}":bounds([(x[10],x[arm]) for x in pairs if x[0] and tuple(x[1:3])==(p,t)])
           for p,t in TABLES} if root["worlds"]>1 else {}
    worlds=[bounds([(x[10],x[arm]) for x in pairs if x[0]==w]) for w in range(1,root["worlds"])]
    rank="undecided"
    if len(worlds)==4:
        if all(lo>0 for lo,_ in cells.values()) and all(lo>=0 for lo,_ in worlds): rank="alternative"
        if all(hi<0 for _,hi in cells.values()) and all(hi<=0 for _,hi in worlds): rank="student"
    return dict(rank=rank, tables=cells, worlds=worlds)


def summarize(roots):
    groups={}
    for r in roots:
        if r["status"]==2:
            r["comparisons"]={name:compare_student(r,arm) for name,arm in (("ordinary",4),("teacher",7))}
        pairs=[(x[10],x[4]) for x in r.get("pairs",[]) if x[0]]
        bucket=groups.setdefault(r["seed"],[])
        bucket.append(bounds(pairs) if pairs else [-1,1] if r["status"]==2 else [0,0])
    lower,upper=zip(*(tuple(mean(v) for v in zip(*block)) for block in groups.values()))
    eligible=[r for r in roots if r["status"]==2]
    def keys_with(name):
        return {r["seed"] for r in eligible if any(x["rank"]==name for x in r["comparisons"].values())}
    return dict(openings=len(roots),sources=len(groups),dispositions=dict(Counter(str(r["status"]) for r in roots)),
                candidate_roots=len(eligible), teacher_agreed_ordinary=sum(r["features"][0]["move"]==r["features"][1]["move"] for r in eligible),
                direction_by_alternative={k:dict(Counter(r["comparisons"][k]["rank"] for r in eligible)) for k in ("ordinary","teacher")},
                distinct_helped_families=len(keys_with("alternative")),distinct_harmed_families=len(keys_with("student")),
                continuations=3*sum(len(r["pairs"]) for r in eligible),
                ordinary_minus_student=[mean(lower),mean(upper)],grouped_95=[interval(lower)[0],interval(upper)[1]])


def main(audit):
    path=ROOT/"build/student-anchored-conversion"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        path.mkdir(parents=True,exist_ok=False)
        build(path)
    model=json.loads(MODEL.read_text(encoding="utf8"))
    exe=path/"deadline_opening.exe"
    roster=model_roster()
    weights=[w for name in ("candidate","historical") for w in roster[name]["weights"]]
    raw={}; start=perf_counter()
    for name,grid in (("locked",LOCKED),("holdout",FRESH)):
        payload=" ".join(map(str,(*grid,*weights)))+" "+model_arguments(model,"replay_control")
        run=subprocess.run([exe,RULES,"student-label-"+name],cwd=ROOT,input=payload,encoding="utf8",capture_output=True,timeout=110)
        require(run.returncode==0,f"{name} failed: {run.stderr.strip()}")
        raw[name]=run.stdout
    locked=decode(raw["locked"],True,True)
    fresh=decode(raw["holdout"],True,True)
    require(set(fresh)=={(s,s%4,3,t,0) for s in range(FRESH[0],sum(FRESH)) for t in (0,1)},"incomplete student-centered openings")
    root=locked[(LOCKED[0],LOCKED[0]%4,3,0,0)]
    prior=next(r for r in json.loads(REFERENCE.read_text(encoding="utf8"))["rows"] if (r["seed"],r["table"])==(LOCKED[0],0))
    require(all(root[k]==prior[k] for k in ("checkpoint","ply","original","student_move")) and
            root["features"][:2]==prior["features"] and root["features"][2]["move"]==root["student_move"],"locked student error changed")
    root["comparisons"]={name:compare_student(root,arm) for name,arm in (("ordinary",4),("teacher",7))}
    report=summarize(list(fresh.values()))
    result=dict(engine=file_sha(exe),model=file_sha(MODEL),rules=file_sha(RULES),prior=file_sha(REFERENCE),
                locked=root,holdout=report,rows=list(fresh.values()))
    if audit:
        require(result==json.loads((path/"student.json").read_text(encoding="utf8")) and all(
            work==(path/(name+".txt")).read_text(encoding="utf8") for name,work in raw.items()),"student reprobe did not replay")
    else:
        for name,work in raw.items(): (path/(name+".txt")).write_text(work,encoding="utf8")
        write_receipt(path/"student.json",result)
    print(json.dumps({"locked":root["comparisons"],**report},indent=2))
    print(f"student-centered terminal comparisons seconds={perf_counter()-start:.2f}")
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
