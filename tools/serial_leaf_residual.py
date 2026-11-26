"""Use the shared leaf evaluation path for bounded residual calibration."""
import hashlib
import json
import os
import sys

from serial_dynamic_reply import ROOT, PRIOR, build, frozen_wire, require
from serial_full_student_evolution import digest
from serial_leaf_teacher import collect, fit_residual, summarize, train as train_value, value_rows
from serial_preference_retention import np, train as train_student, training_rows
from serial_quiet_options import FROZEN
from serial_reply_window import examples, model_wire, split
from serial_student_lesson_credit import PREVIOUS, measure_root_scores
from serial_visited_reply_value import manifest
from serial_worker import read,write_receipt

ALPHAS=(.25,.5,1.)


def lock_scale(data):
    """Use layout-block folds from development data only, keeping the holdout untouched."""
    keys=sorted(data["roots"],key=lambda k:hashlib.sha256(k.encode()).digest())
    losses={a:[] for a in ALPHAS}
    baseline=[]
    for i in range(4):
        test=set(keys[i::4])
        fit={"roots":{k:r for k,r in data["roots"].items() if k not in test}}
        probe=dict(data,roots={k:r for k,r in data["roots"].items() if k in test})
        artifact=train_value(fit)
        residual=fit_residual(fit,artifact["models"]["calibrated"])
        for alpha in ALPHAS:
            artifact["residual"]=dict(residual,alpha=alpha)
            forecasts,_=value_rows(probe,artifact)
            metrics=summarize(probe,forecasts)["calibration"]
            losses[alpha].append(float(np.mean([metrics[t]["residual"]["brier"] for t in range(3)])))
            if alpha==ALPHAS[0]:
                baseline.append(float(np.mean([metrics[t]["calibrated"]["brier"] for t in range(3)])))
    return min(ALPHAS,key=lambda a:np.mean(losses[a])),dict(folds=4,residual=losses,calibrated=baseline)


def diagnose_student(folder):
    parent=read(PRIOR/"students.json")["models"]["window"]
    screen=read(PREVIOUS/"screen.json")
    cases,_,_=examples(folder)
    old,_=split(cases)
    raw,y,pairs,retain=training_rows(screen,old)
    indexes=[index for a,b,_ in pairs for index in (a,b)]
    isolated_pairs=[(a,a+1,.99) for a in range(0,len(indexes),2)]
    hard=[(a,b,.99) for a,b,_ in pairs]
    opts=dict(steps=320,rate=.12,decay=0.)
    sets=dict(parent=parent,isolated=train_student(parent,(raw[indexes],y[indexes],isolated_pairs,[]),True,**opts),
              mixed=train_student(parent,(raw,y,hard,retain),True,**opts))
    report={m:measure_root_scores(folder,weight,screen) for m,weight in sets.items()}
    write_receipt(folder/"three-case-diagnostic.json",report)
    return {m:dict(pair_positive=sum(r["lesson_gap"]>0 for r in rows.values()),
                   chosen=sum(r["chosen"]==r["lesson"] for r in rows.values())) for m,rows in report.items()}


def main(audit):
    folder=ROOT/"build/leaf-residual"/os.environ["JUNQI_SERIAL_KEY"]
    wire=frozen_wire()[0]+" "+model_wire(read(FROZEN/"student.json")["models"]["window"],"window")
    if not audit:
        folder.mkdir(parents=True,exist_ok=True)
        build(folder)
    sources=manifest(audit,seed_offset=12100 if audit else 11300,archive_offset=24,
                     include_witness=False,cuts=(1,18,32,45))
    data=collect(folder,wire,audit,sources=sources)
    path=folder/"value.json"
    if not audit:
        alpha,cv=lock_scale(data)
        artifact=train_value(data)
        artifact.update(residual=fit_residual(data,artifact["models"]["calibrated"],alpha),
                        source_sha=digest(wire),training_ids=list(data["spec"]),
                        training_families=[r["family"] for r in data["spec"].values() if "family" in r],
                        cv=cv,case_probe=diagnose_student(folder))
        write_receipt(path,artifact)
    artifact=read(path)
    require(artifact["source_sha"]==digest(wire),"loaded model used different frozen policies")
    if audit:
        require(set(artifact["training_ids"]).isdisjoint(data["spec"]) and set(artifact["training_families"]).isdisjoint(
            r["family"] for r in data["spec"].values() if "family" in r),"residual confirmation source reused")
    forecasts,cost=value_rows(data,artifact)
    stats=summarize(data,forecasts)
    stats.update(cost=cost,actual_estimators=tuple(next(iter(forecasts.values()))["choice"]),
                 loaded_sha=hashlib.sha256(path.read_bytes()).hexdigest(),alpha=artifact["residual"]["alpha"],
                 truncated_fallbacks=sum(r["truncated"] for r in forecasts.values()),case_probe=artifact["case_probe"])
    write_receipt(folder/("confirmation.json" if audit else "development.json"),
                  dict(experience=data,predictions=forecasts,summary=stats))
    print(json.dumps(stats))
    if audit:
        ci=[row["interval"] for row in stats["effects"]["residual_vs_legacy"]["tables"].values()]
        print("JUNQI_RESEARCH "+("supported" if all(lo>0 for lo,_ in ci) else
                                "refuted" if all(hi<0 for _,hi in ci) else "inconclusive"))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
