"""Fit a small bomb-branch reweighting model and audit independent chooser consequences."""
from hashlib import sha256
from itertools import product
import json
import os
import shutil
from statistics import mean
import sys
from time import perf_counter

from full_mode_runtime import FullModeRun,isolated,sha
from serial_dark_information_audit import SOURCE
from serial_dynamic_reply import ROOT,build,require
from serial_formation_matrix import contrast,score_difference
from serial_preference_retention import np
from serial_worker import read,write_receipt
from train_early_deadline_student import RULES,model_arguments

PREVIOUS=ROOT/"build/dark-information-audit/verify-four-dark-information-and-all-actor-controls-4735e68a487d6694f102"
FIELDS="history focus slots minimum maximum moved row cut nonbomb bomb no_excluded bomb_excluded nodes size".split()


def prepare(scope):
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    shutil.copyfile(PREVIOUS/"student.json",scope.folder/"student.json")
    blocked={r["family"] for group in read(PREVIOUS/"manifest.json") for r in group}
    rows=read(SOURCE/"probe.json")["rows"]
    eligible=sorted((r for r in rows if r["format"]==".jgs" and r["result"] in (1,2,3) and r["mid"] and
                    not any(r[k] for k in ("prior_layout","prior_trajectory","stopped","notice"))),
                    key=lambda r:sha256(("dark-bomb-probability-v1:"+r["family"]).encode()).digest())
    selected=[]
    for r in eligible:
        if r["family"] in blocked:
            continue
        blocked.add(r["family"])
        selected.append({k:r[k] for k in ("key","family","learner","mid","direction")})
        if len(selected)==44:
            break
    require(len(selected)==44,"probability cohorts need disjoint archive families")
    use={r["key"] for r in selected}
    tapes={}
    for line in (SOURCE/"games.txt").read_text(encoding="utf8").splitlines():
        kind,k,tape=line.split(maxsplit=2)
        if kind=="G" and k in use:
            tapes[k]=tape
    require(tapes.keys()==use,"probability audit sources unavailable")
    for i,cohort in enumerate((selected[:24],selected[24:])):
        sheet=scope.folder/f"replay-{i}.txt"
        sheet.write_text("\n".join(f"{r['key']} {r['learner']} {r['mid']-1} {r['direction']} {22000+j+100*i} {tapes[r['key']]}"
                                   for j,r in enumerate(cohort))+"\n",encoding="ascii")
    write_receipt(scope.folder/"manifest.json",selected)
    files=("engine/deadline_opening.exe","engine/rules.json","student.json","replay-0.txt","replay-1.txt","manifest.json")
    write_receipt(scope.folder/"fingerprints.json",{n:sha(scope.folder/n) for n in files})


def audit_rows(scope,split):
    split_path=scope.folder/f"replay-{split}.txt"
    rows=read(scope.folder/"manifest.json")
    req={r["key"]:r for r in (rows[24:] if split else rows[:24])}
    output={}
    artifact=read(scope.folder/"student.json")
    wire=model_arguments(artifact,"collision",artifact["encoding"])+"\n"+split_path.read_text(encoding="ascii")
    with isolated(scope,ROOT/"build"):
        raw=scope.run("dark-bomb-belief",wire)
    for tag,key,*fields in map(str.split,raw.splitlines()):
        require(key in req,"unregistered probability root")
        if tag=="M":
            output[key]=dict(req[key],missing=fields[0])
            continue
        v=list(map(int,fields))
        if tag=="P":
            output[key]=dict(req[key],**dict(zip(FIELDS,v,strict=True)),components={},outcomes=[])
        elif tag=="T":
            truth,=v
            r=output[key]
            require(not r["bomb_excluded" if truth else "no_excluded"],"joint candidate exclusion contradicted replay truth")
            r["truth"]=truth
        elif tag=="Q":
            action,side,samples,natural,lo,hi,work=v
            r=output[key]
            require(samples==2*r["bomb" if side else "nonbomb"] and 0<=natural<=samples and
                    0<=lo<=hi<=2*samples,"invalid joint conditional outcome bound")
            r["components"].setdefault(action,{})[side]=[samples,natural,lo,hi,work]
        else:
            require(tag=="Y","unknown conditional trace")
            output[key]["outcomes"].append(v)
    require(output.keys()==req.keys(),"missing offered or skipped root")
    for row in output.values():
        if "missing" in row:
            continue
        both=bool(row["nonbomb"] and row["bomb"])
        require(len(row["components"])==row["size"] and
                (set(product(row["components"],range(4),range(4))) if both else set())==
                {tuple(y[:3]) for y in row["outcomes"]},"conditional final-score audit misaligned")
    return output


def features(row):
    return [1., (row["minimum"]+row["maximum"])/2-1., float(row["moved"]), float(row["row"]>=4)]


def learned_probability(row,weights):
    prior=2/row["slots"]
    offset=np.log(prior/(1-prior))
    return float(np.exp(-np.logaddexp(0.,-(offset+np.asarray(features(row))@weights))))


def train(rows):
    x=np.asarray([features(r) for r in rows])
    y=np.asarray([r["truth"] for r in rows])
    require(len(rows)>=6,"not enough observed bomb decisions for a lightweight fit")
    offset=np.log(np.asarray([2/(r["slots"]-2) for r in rows]))
    weights=np.zeros(x.shape[1])
    for _ in range(100):
        p=np.exp(-np.logaddexp(0.,-(offset+x@weights)))
        weights-=.35*(x.T@(p-y)/len(y)+.12*weights)
    require(np.isfinite(weights).all() and np.linalg.norm(weights)>0,"bomb calibration did not update")
    return dict(encoding="four-dark-bomb-risk-v1",features=["bias","live-stock-center","moved","back-board-row"],
                updates=100,weights=weights.tolist(),development_families=[r["family"] for r in rows],
                positive=int(y.sum()),samples=len(y))


def choose(row,p):
    if not row["nonbomb"] or not row["bomb"]:
        return row["history"]
    best=-1.
    picked=row["history"]
    for action in [picked,*[a for a in row["components"] if a!=picked]]:
        no,yes=(row["components"][action][i] for i in (0,1))
        value=(1-p)*(no[2]+no[3])/(4*no[0])+p*(yes[2]+yes[3])/(4*yes[0])
        if value>best+1e-12:
            best=value
            picked=action
    return picked


def assess(rows,model):
    probs={"half":[.5 for _ in rows],"slots":[2/r["slots"] for r in rows],
           "learned":[learned_probability(r,model["weights"]) for r in rows]}
    actual=[r["truth"] for r in rows]
    proper={name:dict(brier=mean((a-b)**2 for a,b in zip(q,actual,strict=True)),
                      log_loss=-mean(float(np.log(p if y else 1-p)) for p,y in zip(q,actual,strict=True)))
            for name,q in probs.items()}
    chosen={k:[choose(r,p) for r,p in zip(rows,v,strict=True)] for k,v in probs.items()}
    panels={}
    for desk in range(4):
        panels[desk]={}
        for label in ("half","slots","history"):
            bases=chosen[label] if label in chosen else [r["history"] for r in rows]
            pairs=[]
            for r,base,test in zip(rows,bases,chosen["learned"],strict=True):
                trials={(a,j):score for a,t,j,score,_ in r["outcomes"] if t==desk}
                diffs=[(0.,0.) if base==test else score_difference(trials[base,j],trials[test,j]) for j in range(4)]
                pairs.append(tuple(map(mean,zip(*diffs,strict=True))))
            panels[desk][label]=contrast(pairs)
    return dict(proper=proper,table=panels,changed_from_slots=sum(a!=b for a,b in zip(chosen["slots"],chosen["learned"],strict=True)),
                changed_from_half=sum(a!=b for a,b in zip(chosen["half"],chosen["learned"],strict=True)),
                both_sides=sum(bool(r["bomb"] and r["nonbomb"]) for r in rows),positive=sum(actual),
                rates=list(zip(probs["slots"],probs["learned"],actual,strict=True)),actions=chosen)


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-bomb-calibration"/os.environ["JUNQI_SERIAL_KEY"])
    if not audit:
        prepare(scope)
    require(all(sha(scope.folder/n)==v for n,v in read(scope.folder/"fingerprints.json").items()),"bomb calibration source changed")
    begin=perf_counter()
    raw=audit_rows(scope,int(audit))
    rows=[r for r in raw.values() if "missing" not in r]
    path=scope.folder/"model.json"
    if not audit:
        write_receipt(path,train(rows))
        write_receipt(scope.folder/"model-integrity.json",{"model_sha":sha(path)})
    else:
        require(sha(path)==read(scope.folder/"model-integrity.json")["model_sha"],"reloaded risk model changed")
    model=read(path)
    require(model["encoding"]=="four-dark-bomb-risk-v1" and len(model["weights"])==4,"wrong risk model encoding")
    if audit:
        require(set(model["development_families"]).isdisjoint(r["family"] for r in rows),"calibration reused fitting layouts")
    result=assess(rows,model)
    summary=dict(registered=len(raw),eligible=len(rows),runtime_seconds=perf_counter()-begin,model_sha=sha(path),**result)
    write_receipt(scope.folder/("confirmation.json" if audit else "development.json"),dict(roots=raw,summary=summary))
    if audit:
        better=result["proper"]["learned"]["brier"]<result["proper"]["slots"]["brier"]
        decisive=all(t["slots"]["interval"][0]>0 for t in result["table"].values())
        print("JUNQI_RESEARCH "+("supported" if better and decisive else "inconclusive"))
    print(json.dumps({k:v for k,v in summary.items() if k not in ("rates","actions")}))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no args or audit")
    main(bool(sys.argv[1:]))
