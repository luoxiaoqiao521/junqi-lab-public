"""Resumable shuffled microbatches using the existing student SGD update."""
import os
from pathlib import Path
import sys

from compare_book_learning import fit, np
from full_mode_runtime import sha
from serial_worker import read, write_receipt

SCHEMA="full-mode-student-microbatch-v1"
ENCODING="junqi-full-reply-window-action-v1"


def validate(cursor,config,lineage,rows):
    if cursor.get("schema")!=SCHEMA or cursor.get("source_sha")!=lineage or cursor.get("optimizer")!=config:
        raise ValueError("checkpoint model, course or optimizer provenance changed")
    batches=cursor["batches"]
    if not 0<=cursor["step"]==len(batches)<=config["updates"]:
        raise ValueError("checkpoint step and sample ledger disagree")
    epochs={}
    for epoch,batch in batches:
        before=epochs.setdefault(epoch,[])
        if not 0<len(batch)<=config["batch"] or any(k in before or k<0 or k>=rows for k in batch):
            raise ValueError("duplicate or invalid sample within the saved epoch")
        before.extend(batch)
        if len(set(before))!=len(before):
            raise ValueError("duplicate saved sample IDs")
    if cursor["step"]==0:
        if (cursor["epoch"],cursor["offset"],cursor["order"])!=(-1,0,[]):
            raise ValueError("nonempty cursor without committed training")
        return
    order=cursor["order"]
    epoch=cursor["epoch"]
    offset=cursor["offset"]
    if (list(epochs)!=list(range(epoch+1)) or sorted(order)!=list(range(rows)) or not 0<offset<=rows or
            epochs[epoch]!=order[:offset] or any(len(epochs[e])!=rows for e in range(epoch))):
        raise ValueError("checkpoint permutation or data cursor drifted")


def execute(folder,run_name):
    data=folder/"course-input.json"
    spec=read(data)
    source=sha(data)
    config=spec["optimizer"]
    if spec["encoding"]!=ENCODING or config["algorithm"]!="sgd-l2-no-momentum":
        raise ValueError("unsupported checkpoint training contract")
    x=np.array(spec["x"],dtype=float)
    y=np.array(spec["targets"],dtype=float)
    rng=np.random.default_rng(config["seed"])
    path=folder/("control.json" if run_name=="control" else "checkpoint.json")
    if path.exists():
        cursor=read(path)
        validate(cursor,config,source,len(x))
        rng.bit_generator.state=cursor["rng"]
    else:
        if run_name=="resume":
            raise ValueError("resume requires an explicit checkpoint")
        cursor=dict(schema=SCHEMA,source_sha=source,optimizer=config,rng=rng.bit_generator.state,step=0,
                    epoch=-1,order=[],offset=0,batches=[],model=spec["parent"])
    while cursor["step"]<config["updates"]:
        if cursor["offset"]==len(cursor["order"]):
            cursor["order"]=rng.permutation(len(x)).tolist()
            cursor["offset"]=0
            cursor["epoch"]+=1
        start=cursor["offset"]
        batch=cursor["order"][start:start+config["batch"]]
        cursor["model"]=fit(x[batch],y[batch],config["seed"],config["width"],cursor["model"],
                            updates=1,rate=config["rate"])
        cursor["offset"]+=len(batch)
        cursor["step"]+=1
        cursor["batches"].append([cursor["epoch"],batch])
        cursor["rng"]=rng.bit_generator.state
        write_receipt(path,cursor)
        if run_name=="interrupt" and cursor["step"]==config["interrupt_after"]:
            os._exit(75)
    validate(cursor,config,source,len(x))


if __name__=="__main__":
    try:
        folder,mode=sys.argv[1:]
        if mode not in ("control","interrupt","resume","normal"):
            raise ValueError("unsupported checkpoint operation")
        execute(Path(folder).resolve(),mode)
    except ValueError as error:
        print(error,file=sys.stderr)
        raise SystemExit(65)
