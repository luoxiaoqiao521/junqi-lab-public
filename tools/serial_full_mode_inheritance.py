"""Pair frozen resets with true student inheritance on the same saved lesson batches."""
import json
import os
import shutil
import sys
from time import perf_counter
from statistics import mean

from full_mode_checkpoint import execute
from full_mode_runtime import FullModeRun, isolated, sha
from serial_dynamic_reply import ROOT, require
from serial_formation_matrix import contrast, score_difference
from serial_full_student_evolution import games
from serial_worker import read, write_receipt

SOURCE=ROOT/"build/full-mode-resume/resume-isolated-full-mode-training-checkpoint-93fb7a0ce16544096ffc"
EVAL_STARTS=(2860100,2860200,2860300)


def update(folder,course,parameters):
    folder.mkdir(parents=True,exist_ok=True)
    source=folder/"course-input.json"
    write_receipt(source,dict(course,parent=parameters))
    begin=perf_counter()
    execute(folder,"normal")
    elapsed=perf_counter()-begin
    checkpoint=folder/"checkpoint.json"
    result=read(checkpoint)
    require(result["step"]==course["optimizer"]["updates"] and result["model"]!=parameters,
            "student did not consume the full registered update")
    return result,dict(seconds=elapsed,input_sha=sha(source),checkpoint_sha=sha(checkpoint),
                       updates=result["step"],consumed=sum(len(batch) for _,batch in result["batches"]))


def paired_scores(reset,inherited,start):
    def rows(panel,arm):
        return {(r["seed"],r["table"]):r for r in panel["rows"] if r["arm"]==arm}
    first,second=rows(reset,4),rows(inherited,4)
    require(first.keys()==second.keys(),"different independent match grid")
    for arm in (0,1,3):
        require([(k,r["final_hash"]) for k,r in rows(reset,arm).items()]==
                [(k,r["final_hash"]) for k,r in rows(inherited,arm).items()],"frozen control distribution drift")
    contrasts={}
    table_bounds=[]
    for table in range(3):
        diffs=[]
        for seed in range(start,start+16):
            a,b=first[seed,table],second[seed,table]
            diffs.append((0.,0.) if a["final_hash"]==b["final_hash"] else score_difference(a["result"],b["result"]))
        table_bounds.append(diffs)
        contrasts[table]=dict(**contrast(diffs),distinct=sum(a["final_hash"]!=second[k]["final_hash"]
                                                             for k,a in first.items() if k[1]==table))
    pooled=contrast([tuple(map(mean,zip(*sample,strict=True))) for sample in zip(*table_bounds,strict=True)])
    return dict(tables=contrasts,paired_layouts=16,clustered=pooled)


def stage(root,index,previous=None):
    folder=root/f"generation-{index+1}"
    scope=FullModeRun(folder)
    scope.engine.mkdir(parents=True,exist_ok=True)
    original=SOURCE/f"cycle-{index+1}"
    for filename in ("rules.json","deadline_opening.exe"):
        shutil.copyfile(original/"engine"/filename,scope.engine/filename)
    bundle=read(original/"inputs.json")
    parent=bundle["parent"]["models"]["window"]
    require(sha(scope.binary)==bundle["engine_sha"] and sha(scope.rules)==bundle["inputs"]["rules"],
            "staged evaluator differs from the frozen course")
    source=original/"optimizer/course-input.json"
    course=read(source)
    old=read(original/"optimizer/verification.json")
    require(sha(source)==old["source_sha"] and course["parent"]==parent["parameters"] and
            read(original/"course.json")["source"]=="generated-verified-lessons",
            "registered source is stale or only a training smoke test")
    if previous is None:
        prior=parent
    else:
        prior=read(previous/"inherited-student.json")
        require(prior["mean"]==parent["mean"] and prior["scale"]==parent["scale"],
                "inherited student is using a different feature normalization")
    begin=perf_counter()
    with isolated(scope,ROOT/"build"):
        a,meta_a=update(folder/"reset",course,parent["parameters"])
        b,meta_b=update(folder/"inherited",course,prior["parameters"])
        require(a["batches"]==b["batches"],"different sample order or update budget")
        baseline_checkpoint=read(folder/"reset/checkpoint.json")
        require(baseline_checkpoint["model"]==a["model"],"reset model did not reload")
        for name,checkpoint in (("reset",a),("inherited",b)):
            write_receipt(folder/f"{name}-student.json",dict(parent,parameters=checkpoint["model"]))
        if index==0:
            require(a["model"]==b["model"],"identical parent and course did not reproduce")
        else:
            require(a["model"]!=b["model"],"the two parent choices produced the same parameters")
        start=EVAL_STARTS[index]
        reset=games(scope.engine,parent,read(folder/"reset-student.json"),start)
        inherited=games(scope.engine,parent,read(folder/"inherited-student.json"),start)
        comparison=paired_scores(reset,inherited,start)
        write_receipt(folder/"panel.json",dict(reset=reset,inherited=inherited,comparison=comparison))
    train=dict(reset=meta_a,inherited=meta_b)
    if previous is not None:
        train["prior_sha"]=sha(previous/"inherited-student.json")
    timing=perf_counter()-begin
    result=dict(generation=index+1,source_sha=sha(source),start=start,training=train,seconds=timing,
                comparison=comparison,games=len(reset["rows"])+len(inherited["rows"]),
                terminal_truncations=sum(r["result"]<0 for panel in (reset,inherited) for r in panel["rows"]),
                reset_vs_parent=[reset["differences"][f"{t}/learn_continuously"] for t in range(3)],
                inherited_vs_parent=[inherited["differences"][f"{t}/learn_continuously"] for t in range(3)])
    write_receipt(folder/"decision.json",result)
    return result


def main(audit):
    root=ROOT/"build/full-mode-inheritance"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        result=stage(root,0)
        require(result["comparison"]["clustered"]["interval"]==[0,0],"same-parent control diverged")
    else:
        initial=read(root/"generation-1/decision.json")
        second=stage(root,1,root/"generation-1")
        third=stage(root,2,root/"generation-2")
        result=[initial,second,third]
        write_receipt(root/"summary.json",result)
        intervals=[g["comparison"]["clustered"]["interval"] for g in result[1:]]
        verdict="supported" if all(lo>0 for lo,_ in intervals) else "refuted" if all(hi<0 for _,hi in intervals) else "inconclusive"
        print("JUNQI_RESEARCH "+verdict)
    print(json.dumps(result))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
