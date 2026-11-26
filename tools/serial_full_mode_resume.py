"""Run three guarded cycles and interrupt/restart the real student update in the first."""
import json
import os
import shutil
import subprocess
import sys

from full_mode_checkpoint import ENCODING
from full_mode_runtime import FullModeRun, isolated, sha
from serial_active_full_cycle import generate_and_train, verify
from serial_dynamic_reply import ROOT, require
from serial_worker import read, write_receipt

SOURCE=ROOT/"build/active-full-cycle/verify-isolated-active-runtime-with-windows-build-abbd2229a613c68a9c5c"
ENGINE=SOURCE/"engine"
TRAIN_SEEDS=(2852000,2853000,2854000)
EVAL_SEEDS=(2857000,2858000,2859000)
WORKER=ROOT/"tools/full_mode_checkpoint.py"


def launch(directory,mode,expected=0,marker=""):
    result=subprocess.run([sys.executable,"-X","utf8",WORKER,directory,mode],capture_output=True,
                          encoding="utf8",cwd=directory,timeout=45)
    require(result.returncode==expected and marker in result.stderr,
            f"checkpoint worker {mode} unexpected exit {result.returncode}: {result.stderr.strip()}")


def checkpoint_optimizer(directory,interrupt):
    def optimizer(x,y,seed,width,initial,*,updates,rate):
        directory.mkdir(parents=True,exist_ok=True)
        settings=dict(algorithm="sgd-l2-no-momentum",rate=rate,width=width,seed=seed,updates=updates,
                      interrupt_after=17,batch=12)
        targets=[[float(v) if v==v else None for v in row] for row in y]
        payload=dict(encoding=ENCODING,optimizer=settings,parent=initial,x=x.tolist(),targets=targets)
        path=directory/"course-input.json"
        write_receipt(path,payload)
        active=directory/"checkpoint.json"
        control=directory/"control.json"
        resumed={}
        if interrupt:
            launch(directory,"control")
            launch(directory,"interrupt",75)
            cursor=read(active)
            require(cursor["step"]==17 and cursor["source_sha"]==sha(path),"forced exit missed the committed batch")
            write_receipt(directory/"interrupted.json",cursor)
            damaged=directory/"wrong-input"
            damaged.mkdir()
            write_receipt(damaged/"course-input.json",dict(payload,encoding="stale-course-encoding"))
            shutil.copyfile(active,damaged/"checkpoint.json")
            launch(damaged,"resume",65,"unsupported checkpoint training contract")
            provenance=directory/"wrong-source"
            provenance.mkdir()
            changed=dict(payload)
            changed["x"]=[*payload["x"][1:],payload["x"][0]]
            write_receipt(provenance/"course-input.json",changed)
            shutil.copyfile(active,provenance/"checkpoint.json")
            launch(provenance,"resume",65,"checkpoint model, course or optimizer provenance changed")
            launch(directory,"resume")
            require(active.read_bytes()==control.read_bytes(),"restart changed weights, RNG or consumed data")
            resumed=dict(interrupted_after=cursor["step"],replayed_batches=0,
                         comparison="bitwise-identical-to-uninterrupted",stale_input_rejected=True)
        else:
            launch(directory,"normal")
        result=read(active)
        stamp=sha(active)
        launch(directory,"resume")
        require(result["step"]==updates and stamp==sha(active),"same completed checkpoint trained a second time")
        write_receipt(directory/"verification.json",dict(resumed,iterations=updates,rows=len(x),
                                                        source_sha=sha(path),checkpoint_sha=stamp,
                                                        repeated_resume_unchanged=True))
        return result["model"]
    return optimizer


def stage(directory,baseline,index):
    scope=FullModeRun(directory)
    scope.engine.mkdir(parents=True,exist_ok=True)
    for name in ("rules.json","deadline_opening.exe"):
        shutil.copyfile(ENGINE/name,scope.engine/name)
    bundle=read(SOURCE/"inputs.json")
    require(sha(scope.binary)==bundle["engine_sha"]==baseline["engine_sha"] and
            sha(scope.rules)==bundle["inputs"]["rules"],"cold cycle assets drifted")
    write_receipt(scope.folder/"inputs.json",bundle)
    with isolated(scope,ROOT/"build"):
        generate_and_train(scope,bundle,seed=TRAIN_SEEDS[index],
                           optimizer=checkpoint_optimizer(scope.folder/"optimizer",index==0))
        result=verify(scope,bundle,seed=EVAL_SEEDS[index])
    course=read(scope.folder/"course.json")
    require(len(result["panel"]["rows"])==240 and len(course["plan"])==24,"truncated guarded cycle")
    improvement=[t["interval"][0]>0 for k,t in result["panel"]["differences"].items() if k.endswith("/learn_continuously")]
    route="candidate-for-another-independent-check" if all(improvement) else "retain-frozen-parent"
    proof=read(scope.folder/"optimizer/verification.json")
    write_receipt(scope.folder/"decision.json",dict(result,source_seed=TRAIN_SEEDS[index],
                                                   verification_seed=EVAL_SEEDS[index],optimizer=proof,
                                                   outcome=route))
    return dict(cycle=index+1,root_count=len(course["cuts"]["roots"]),lessons=result["lesson_count"],source=result["ordinary_source"],
                games=len(result["panel"]["rows"]),outcome=route,optimizer=proof)


def main(audit):
    root=ROOT/"build/full-mode-resume"/os.environ["JUNQI_SERIAL_KEY"]
    baseline=read(SOURCE/"decision.json")
    require(baseline["historical_compatibility_restored"],"active scope did not close previously")
    if not audit:
        first=stage(root/"cycle-1",baseline,0)
        write_receipt(root/"first-cycle.json",first)
        print(json.dumps(first))
    else:
        first=read(root/"first-cycle.json")
        require(first["optimizer"]["comparison"]=="bitwise-identical-to-uninterrupted",
                "first cycle lacks recovery proof")
        rows=[first,stage(root/"cycle-2",baseline,1),stage(root/"cycle-3",baseline,2)]
        write_receipt(root/"summary.json",rows)
        print(json.dumps(rows))
        print("JUNQI_RESEARCH supported")
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
