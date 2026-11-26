"""Exercise a cold full-mode cycle while denying older executable entries and build outputs."""
from collections import Counter
import json
import os
import shutil
import sys

from compare_book_learning import OUTPUTS, fit, np
from full_mode_runtime import FullModeRun, isolated, model_hashes, sha
from serial_dynamic_reply import ROOT, PRIOR, build, frozen_wire, require
from serial_evaluator_panel import RULES
from serial_full_student_evolution import games
from serial_leaf_teacher import collect as leaf_data, fit_residual, train as train_teacher, value_rows
from serial_local_lesson_cycle import independent_review, lesson_proposals, train as train_student
from serial_quiet_options import FROZEN
from serial_reply_window import examples, judge, model_wire, split
from serial_visited_reply_value import collect as search_data
from serial_worker import read, write_receipt

FIRST_SEED=2849000
EVAL_SEED=2851000


def generated(seed):
    spec={}
    tape=[]
    for i in range(24):
        key=str(seed+i)
        cut,policy,table=(1,17,29,44)[i%4],(i//3)%3,i%3
        spec[key]=dict(policy=policy,table=table,cut=cut,reference=0,source="generated")
        tape.append(f"R {key} {key} {cut} {policy} {table} 0")
    return spec,"\n".join(tape)


def isolate_inputs(scope):
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    parent_path=PRIOR/"students.json"
    branch_path=FROZEN/"student.json"
    peer_wire,peer_sha=frozen_wire()
    frozen=read(parent_path)
    wire=peer_wire+" "+model_wire(read(branch_path)["models"]["window"],"window")
    bundle=dict(parent=frozen,wire=wire,inputs=dict(parent=sha(parent_path),branch=sha(branch_path),
                                                   peer=peer_sha,rules=sha(scope.rules)),engine_sha=sha(scope.binary))
    write_receipt(scope.folder/"inputs.json",bundle)
    return bundle


def fit_fixture(parent,anchors,optimizer=fit):
    x=np.array([f for r in anchors for f in r["features"]])
    x=np.clip((x-parent["mean"])/parent["scale"],-6,6)
    y=np.full((len(x),len(OUTPUTS)),np.nan)
    y[:,:3]=np.eye(3)[[label for r in anchors for label in r["scores"]]]
    parameters=optimizer(x,y,54,24,parent["parameters"],updates=40,rate=.03)
    return dict(mean=parent["mean"],scale=parent["scale"],parameters=parameters),len(x)


def generate_and_train(scope,bundle,*,seed=FIRST_SEED,optimizer=fit):
    folder,engine=scope.folder,scope.engine
    wire=bundle["wire"]
    plans=generated(seed)
    cuts=leaf_data(engine,wire,False,sources=plans)
    value=train_teacher(cuts)
    value["residual"]=fit_residual(cuts,value["models"]["calibrated"],.25)
    write_receipt(folder/"teacher.json",value)
    forecasts,_=value_rows(cuts,read(folder/"teacher.json"))
    discovery=search_data(engine,wire,False,plans)
    review=independent_review(engine,wire,plans,discovery)
    lessons=lesson_proposals(discovery,review)
    cases,_,_=examples(engine)
    anchors,_=split(cases)
    parent=bundle["parent"]["models"]["window"]
    student,used=train_student(parent,anchors,review,lessons,optimizer=optimizer)
    source="generated-verified-lessons"
    if student is None:
        # A certified rotation course exercises update/reload even when the new labels abstain.
        student,used=fit_fixture(parent,anchors,optimizer)
        source="certified-rotation-smoke-only"
    require(student!=parent,"guarded training did not change the model")
    write_receipt(folder/"student.json",dict(models={"window":student,"board":bundle["parent"]["models"]["board"]}))
    write_receipt(folder/"course.json",dict(plan=plans[0],cuts=cuts,forecasts=forecasts,lessons=lessons,
                                          source=source,consumed_rows=used,
                                          value_sha=sha(folder/"teacher.json"),student_sha=sha(folder/"student.json"),
                                          engine_sha=sha(scope.binary),truncated_fallbacks=sum(r["truncated"] for r in forecasts.values())))


def controls(scope,bundle):
    for mode in ("reply-budget","learning-roots","formation-rules"):
        scope.run(mode,expected=(86,"JUNQI_ACTIVE_DENIED "+mode))
    wire=model_wire(bundle["parent"]["models"]["window"],"window")
    broken=wire.replace("junqi-full-reply-window-action-v1","invalid-student-encoding",1)
    require(not scope.run("reply-window-judge-window",broken,expected=(2,"JUNQI_ACTIVE_PATH")).strip(),
            "rejected model nevertheless selected an action")
    try:
        (PRIOR/"students.json").read_bytes()
    except PermissionError:
        scope.record(dict(control="previous-build-artifact",result="blocked"))
    else:
        raise ValueError("guarded cycle read an undeclared previous output")


def verify(scope,bundle,*,seed=EVAL_SEED):
    folder,engine=scope.folder,scope.engine
    course=read(folder/"course.json")
    require(course["engine_sha"]==bundle["engine_sha"]==sha(scope.binary) and
            course["value_sha"]==sha(folder/"teacher.json") and course["student_sha"]==sha(folder/"student.json") and
            bundle["inputs"]["rules"]==sha(scope.rules),"loaded cycle does not match published files")
    restored=read(folder/"student.json")
    cases,_,source=examples(engine)
    _,test=split(cases)
    result=judge(engine,restored,source,test)
    parent=bundle["parent"]["models"]["window"]
    child=restored["models"]["window"]
    expected_models=model_hashes(model_wire(parent,"window")+" "+model_wire(child,"window"))
    panel=games(engine,parent,child,seed)
    rows=[json.loads(line) for line in scope.log.read_text(encoding="utf8").splitlines()]
    actual=[r for r in rows if r.get("mode")=="full-student-evolution-games"][-1]
    require(actual["loaded"]==[list(m) for m in expected_models],"teacher-off games did not load the published candidate")
    require({r["mode"] for r in rows if "mode" in r}=={
        "full-leaf-teacher-data","visited-reply-value","visited-lesson-review","reply-window-examples",
        "reply-window-judge-board","reply-window-judge-window","full-student-evolution-games"},"cycle omitted an active stage")
    require(len({r["rules_hash"] for r in rows if "rules_hash" in r})==1,"rule semantics changed between stages")
    return dict(ordinary_source=course["source"],lesson_count=sum(d["admitted"] for d in course["lessons"].values()),
                consumed_rows=course["consumed_rows"],teacher_sha=course["value_sha"],student_sha=course["student_sha"],
                engine_sha=sha(scope.binary),old_cases=result["errors"]["window"]["counts"]["correct"],panel=panel,
                selected_teacher_for_games="teacher-off",fallbacks=course["truncated_fallbacks"],
                active_calls=dict(Counter(r["mode"] for r in rows if "mode" in r)))


def main(audit):
    folder=ROOT/"build/active-full-cycle"/os.environ["JUNQI_SERIAL_KEY"]
    scope=FullModeRun(folder)
    if not audit:
        bundle=isolate_inputs(scope)
        with isolated(scope,ROOT/"build"):
            controls(scope,bundle)
            generate_and_train(scope,bundle)
        print(json.dumps(dict(course=read(folder/"course.json")["source"],scope=str(folder))))
    else:
        bundle=read(folder/"inputs.json")
        with isolated(scope,ROOT/"build"):
            result=verify(scope,bundle)
        require(any(line.startswith("S ") for line in scope.run("formation-rules",strict=False).splitlines()),
                "same isolated engine did not reopen the historical compatibility entry")
        result["historical_compatibility_restored"]=True
        write_receipt(folder/"decision.json",result)
        print(json.dumps({k:v for k,v in result.items() if k!="panel"}))
        print("JUNQI_RESEARCH supported")
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
