"""Retest the locked hypothesis and locate decisions the current student misses."""
from collections import Counter
import json
import os
import subprocess
import sys
from time import perf_counter

from serial_conversion_labels import MODEL, ROOT, RULES, build, decode, file_sha, model_arguments, model_roster, require, summarize
from serial_worker import write_receipt

LOCKED = (1203762,1)
HOLDOUT = (1203850,24)
PRIOR = ROOT/"build/conversion-labels/verify-conditional-conversion-label-pairs-59c14ba449c00e9d2f53/labels.json"


def main(audit):
    path=ROOT/"build/conversion-label-transfer"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        path.mkdir(parents=True,exist_ok=False)
        build(path)
    exe=path/"deadline_opening.exe"
    model=json.loads(MODEL.read_text(encoding="utf8"))
    roster=model_roster()
    weights=[w for name in ("candidate","historical") for w in roster[name]["weights"]]
    raw={}; start=perf_counter()
    for name,grid in (("locked",LOCKED),("holdout",HOLDOUT)):
        payload=" ".join(map(str,(*grid,*weights)))+" "+model_arguments(model,"replay_control")
        run=subprocess.run([exe,RULES,"conversion-label-"+name],cwd=ROOT,input=payload,encoding="utf8",capture_output=True,timeout=110)
        require(run.returncode==0,f"{name} failed: {run.stderr.strip()}")
        raw[name]=run.stdout
    fixed=decode(raw["locked"],True)
    fresh=decode(raw["holdout"],True)
    require(set(fresh)=={(s,s%4,3,t,0) for s in range(HOLDOUT[0],sum(HOLDOUT)) for t in (0,1)},"incomplete held-out sources")
    root=fixed[(LOCKED[0],LOCKED[0]%4,3,0,0)]
    previous=next(item for item in json.loads(PRIOR.read_text(encoding="utf8"))["fresh"]
                  if (item["seed"],item["table"])==(LOCKED[0],0))
    require(previous["direction"]=="proposal" and all(root[k]==previous[k] for k in
            ("seed","table","ply","checkpoint","original","features")),
            "locked candidate or known base did not reproduce")
    proposals=[r for r in fresh.values() if r["status"]==2]
    resolved=[r for r in proposals if r["direction"]!="unresolved"]
    report=summarize(list(fresh.values()))
    report.update(students=dict(Counter(r["student_disposition"] for r in proposals)),
                  teacher_absent_from_student_shortlist=sum(not r["proposal_in_student_shortlist"] for r in proposals),
                  resolved_examples=[{k:r[k] for k in ("seed","table","ply","direction","student_disposition",
                                      "table_bounds","teacher_minus_student")} for r in resolved],
                  three_arm_continuations=3*sum(len(r["pairs"]) for r in proposals))
    result=dict(engine=file_sha(exe),rules=file_sha(RULES),model=file_sha(MODEL),prior=file_sha(PRIOR),
                locked=root, holdout=report,rows=list(fresh.values()))
    if audit:
        require(result==json.loads((path/"transfer.json").read_text(encoding="utf8")) and all(
            work==(path/(name+".txt")).read_text(encoding="utf8") for name,work in raw.items()),"lock and holdout did not replay")
    else:
        for name,work in raw.items(): (path/(name+".txt")).write_text(work,encoding="utf8")
        write_receipt(path/"transfer.json",result)
    print(json.dumps({"locked":{k:root[k] for k in ("direction","worlds","table_bounds","student_move",
                       "student_disposition","proposal_in_student_shortlist","teacher_minus_student")}, **report},indent=2))
    print(f"held-out continuations seconds={perf_counter()-start:.2f}")
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
