"""Falsify a proposed conditional conversion lesson before creating training labels."""
from collections import Counter
import json
import os
from statistics import mean
import subprocess
import sys
from time import perf_counter

from measure_deadline_opening import ROOT, RULES, SCORE, build, file_sha, interval, model_roster, require
from measure_human_winner_takeover import MODEL
from serial_conversion_history import CROSS
from serial_worker import write_receipt
from train_early_deadline_student import model_arguments

PILOT = (1203411,1)
FRESH = (1203750,16)
JOURNAL = ROOT/"build/conversion-history/verify-conversion-handoff-history-21405eae3c5d9a83e869/prefixes.txt"
TABLES = tuple((a,b) for a in (2,3) for b in (0,1))
KEY = ("seed","seat","partner","table","early")


def bounds(results):
    lo,hi=[],[]
    for a,b in results:
        a,b=SCORE[a],SCORE[b]
        lo.append((0 if b is None else b)-(1 if a is None else a))
        hi.append((1 if b is None else b)-(0 if a is None else a))
    return [mean(lo),mean(hi)]


def decode(raw, confirmation=False, anchored=False):
    tables=tuple((a,b) for a in ((1,3) if anchored else (1,2)) for b in (0,1)) if confirmation else TABLES
    cases={}
    for line in raw.splitlines():
        tag,*tokens=line.split()
        if tag=="R":
            key=tuple(map(int,tokens[:5]))
            row=dict(zip(KEY,key,strict=True),checkpoint=int(tokens[5]),status=int(tokens[6]),features=[],pairs=[])
            require(key not in cases,"duplicate conversion roots")
            cases[key]=row
            if row["status"]==2:
                metrics=("ply","legal","nominated","worlds","attempts","baseline_score","proposal_score","sample_count")
                if confirmation: metrics+=("student_move","proposal_in_student_shortlist")
                row.update(zip(metrics,map(int,tokens[7:]),strict=True))
        elif tag=="F":
            move=int(tokens[0]); score,*features=map(float,tokens[1:])
            require(len(features)==44,"unknown student encoding")
            row["features"].append(dict(move=move,team_estimate=score,inputs=features))
        elif tag=="O":
            row["original"]=list(map(int,tokens))
        else:
            require(tag=="A" and len(tokens)==(13 if confirmation else 10),"unrecognized label observation")
            row["pairs"].append(list(map(int,tokens)))
    for r in cases.values():
        if r["status"]!=2:
            require(not r["features"] and not r["pairs"],"unexpected continuations")
            continue
        require(2<=r["nominated"]<=6 and 1<=r["worlds"]<=5 and 0<=r["attempts"]<=128 and
                len(r["features"])==(3 if anchored else 2) and len(r["original"])==6,"incomplete legal-choice audit")
        expected=[(world,p,t,stream) for world in range(r["worlds"]) for p,t in tables for stream in range(4)]
        require([tuple(x[:4]) for x in r["pairs"]]==expected,"mismatched world, profile or stream")
        require(all(x[j] in SCORE and 0<x[j+1]<=3000 and 1<=x[j+2]<=5 for x in r["pairs"]
                    for j in ((4,7,10) if confirmation else (4,7))),
                "invalid referee terminal")
        r["table_bounds"]={f"{p}/{t}":bounds([(x[4],x[7]) for x in r["pairs"] if x[0] and tuple(x[1:3])==(p,t)])
                           for p,t in tables} if r["worlds"]>1 else {}
        r["world_bounds"]=[bounds([(x[4],x[7]) for x in r["pairs"] if x[0]==w]) for w in range(r["worlds"])]
        r["direction"]="unresolved"
        if r["worlds"]==5:
            bands=[*r["world_bounds"],*r["table_bounds"].values()]
            if all(lo>0 for lo,_ in r["table_bounds"].values()) and all(lo>=0 for lo,_ in bands):
                r["direction"]="proposal"
            if all(hi<0 for _,hi in r["table_bounds"].values()) and all(hi<=0 for _,hi in bands):
                r["direction"]="baseline"
        if confirmation:
            original,proposal=(f["move"] for f in r["features"][:2])
            choice=r["student_move"]
            if choice in (original,proposal):
                prior=4 if choice==original else 7
                require(all(x[prior:prior+3]==x[10:13] for x in r["pairs"]),"same student move changed outcomes")
            r["student_disposition"]="baseline" if choice==original else "proposal" if choice==proposal else "other"
            r["teacher_minus_student"]={f"{p}/{t}":bounds([(x[10],x[7]) for x in r["pairs"]
                                                           if x[0] and tuple(x[1:3])==(p,t)]) for p,t in tables} if r["worlds"]>1 else {}
    return cases


def verify_pilot(pilot):
    key=(1203411,3,3,1)
    expected={tuple(map(int,w[:5])):w for line in CROSS.read_text(encoding="utf8").splitlines()
              if tuple(map(int,(w:=line.split())[:4]))==key}
    journal={tuple(map(int,w[:5])):int(w[7]) for line in JOURNAL.read_text(encoding="utf8").splitlines()
             if tuple(map(int,(w:=line.split())[:4]))==key}
    require(set(pilot)==set(expected)==set(journal),"missing frozen reversal roots")
    for k,r in pilot.items():
        values=list(map(int,expected[k]))
        require(r["status"]==2 and r["checkpoint"]==journal[k]==values[8] and
                r["original"]==[values[i] for i in (23,22,24,33,32,34)] and
                [x["move"] for x in r["features"]]==[values[19],values[29]],"reversal checkpoint did not reproduce")
    shared=[r["features"][1] for r in pilot.values()]
    require(shared[0]["move"]==shared[1]["move"],"original same-move reversal lost")
    changes=[j for j,(a,b) in enumerate(zip(*(r["inputs"] for r in shared),strict=True)) if a!=b]
    return dict(shared_move=shared[0]["move"],different_features_41=[i for i in changes if i<41],
                different_features_44=changes,non_time_feature_changes=[i for i in changes if i>=3],
                estimates=[r["team_estimate"] for r in shared])


def summarize(fresh):
    proposed=[r for r in fresh if r["status"]==2]
    groups={}
    for r in fresh:
        # No intervention leaves the same trajectory. Unknown outcomes get their full score interval.
        band=bounds([(x[4],x[7]) for x in r["pairs"] if x[0]]) if r.get("worlds",0)>1 else [0,0] if r["status"]!=2 else [-1,1]
        groups.setdefault(r["seed"],[]).append(band)
    low,high=zip(*(tuple(mean(v) for v in zip(*group)) for group in groups.values()))
    # This is an experimental distribution at independently re-seeded handoffs, not a deployment gain estimate.
    return dict(openings=len(fresh),sources=len(groups),triggered=len(proposed),
                dispositions=dict(Counter(str(r["status"]) for r in fresh)), directions=dict(Counter(r["direction"] for r in proposed)),
                risk_filtered=sum(r["proposal_score"]==-10001 for r in proposed),
                alternative_worlds=sum(r["worlds"]-1 for r in proposed),paired_continuations=2*sum(len(r["pairs"]) for r in proposed),
                experimental_score_bounds=[mean(low),mean(high)],grouped_95=[interval(low)[0],interval(high)[1]],
                theoretical_trigger_limit=len(proposed)/len(fresh))


def main(audit):
    path=ROOT/"build/conversion-labels"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        path.mkdir(parents=True,exist_ok=False)
        build(path)
    exe=path/"deadline_opening.exe"
    model=json.loads(MODEL.read_text(encoding="utf8"))
    roster=model_roster()
    weights=[w for name in ("candidate","historical") for w in roster[name]["weights"]]
    raw={}; start=perf_counter()
    for name,grid in (("pilot",PILOT),("fresh",FRESH)):
        payload=" ".join(map(str,(*grid,*weights)))+" "+model_arguments(model,"replay_control")
        run=subprocess.run([exe,RULES,"conversion-label-"+name],cwd=ROOT,input=payload,encoding="utf8",capture_output=True,timeout=110)
        require(run.returncode==0,f"{name} failed: {run.stderr.strip()}")
        raw[name]=run.stdout
    pilot,fresh=decode(raw["pilot"]),decode(raw["fresh"])
    require(set(fresh)=={(s,s%4,3,t,0) for s in range(FRESH[0],sum(FRESH)) for t in (0,1)},"invalid unseen opening grid")
    feature_check=verify_pilot(pilot)
    result=dict(engine=file_sha(exe),rules=file_sha(RULES),model=file_sha(MODEL),controls=feature_check,
                summary=summarize(list(fresh.values())),pilot=list(pilot.values()),fresh=list(fresh.values()))
    if audit:
        require(result==json.loads((path/"labels.json").read_text(encoding="utf8")) and all(
            work==(path/(name+".txt")).read_text(encoding="utf8") for name,work in raw.items()),"qualification did not replay")
    else:
        for name,work in raw.items(): (path/(name+".txt")).write_text(work,encoding="utf8")
        write_receipt(path/"labels.json",result)
    print(json.dumps({**result["summary"],"pilot":[(r["direction"],r["table_bounds"],r["world_bounds"]) for r in pilot.values()],
                      "inputs":feature_check},indent=2))
    print(f"paired terminal and replays seconds={perf_counter()-start:.2f}")
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
