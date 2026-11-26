"""Trace and repair changed historical feature rows without discarding legitimate outcomes."""
from itertools import product
import json
import os
import shutil
import sys

from compare_book_learning import fit, np
from full_mode_runtime import FullModeRun, sha
from serial_dark_asset_inheritance import pair_panel
from serial_dark_course_tape import sample_key
from serial_dark_first_loss import PREVIOUS as SECOND
from serial_dark_ordinary_inheritance import PREVIOUS as FIRST
from serial_dark_ordinary_learning import RATE, RULES, UPDATES, examples_xy
from serial_dark_partner_support import SOURCE
from serial_dark_verified_reuse import TAPES, LEGACY, load_history, recorded_openings, validate_splits
from serial_dynamic_reply import ROOT, build, require
from serial_worker import read, write_receipt


ARMS=("parent-reencoded","batch-reencoded","both-reencoded")
SNAPSHOTS={"g0":FIRST/"parent.json","g1":FIRST/"student.json",
           "g2":SECOND/"inherit.json","g2-restart":SECOND/"restart.json",
           "current":SOURCE/"original.json","peer":SOURCE/"peer.json"}
INPUTS=("grid.json","engine/deadline_opening.exe","engine/rules.json",*[f"{n}.json" for n in SNAPSHOTS],
        *[f"history-{i}/{n}" for i in (1,2) for n in LEGACY])


def prepare(scope):
    folder=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    for name,path in SNAPSHOTS.items():
        shutil.copyfile(path,folder/(name+".json"))
    for i in (1,2):
        dst=folder/f"history-{i}"
        dst.mkdir()
        for n in LEGACY:
            shutil.copyfile(TAPES/f"generation-{i}"/n,dst/n)
    write_receipt(folder/"grid.json",[(24176000+i,(i+i//4)%4,t) for i,t in product(range(64),(0,2,3))])
    write_receipt(folder/"fingerprints.json",{n:sha(folder/n) for n in INPUTS})


def retrain(model,rows,seed):
    x,y=examples_xy(rows,model)
    require(len(x)==len(rows),"unsupported course sample reached correction training")
    weights=fit(x,y,seed,24,model["models"]["collision"],updates=UPDATES,rate=RATE)
    return dict(model,models={"collision":weights})


def delta(a,b):
    x,y=(m["models"]["collision"] for m in (a,b))
    changes=[np.max(np.abs(np.asarray(p)-q)) for n in ("weights","bias")
             for p,q in zip(x[n],y[n],strict=True)]
    return float(max(changes))


def versions(scope,index):
    tape,pool,_=load_history(scope,index)
    raw=read(scope.folder/f"history-{index}"/"original.json")["examples"]
    current=[p["data"] for p in pool]
    require([sample_key(r) for r in raw]==[sample_key(r) for r in current] and
            [r["outcome"] for r in raw]==[r["outcome"] for r in current],
            "correction changed the training episode or its outcome")
    changes=[dict(sample=sample_key(a),feature_fields=[
             i for i,(x,y) in enumerate(zip(a["features"],b["features"],strict=True)) if x!=y])
             for a,b in zip(raw,current,strict=True) if a!=b]
    return raw,current,tape,changes


def learn(scope,grid,models):
    folder=scope.folder
    first_raw,first,first_tape,first_changes=versions(scope,1)
    next_raw,next_course,next_tape,next_changes=versions(scope,2)
    mirror,fresh=recorded_openings(scope,[g[0] for g in grid])
    history={r["game"]["seed"]:r["board"] for r in first_tape+next_tape}
    validate_splits(dict(history=history,confirm=fresh),mirror)
    require((len(first_changes),len(next_changes))==(1,4),"reencoded sample counts changed")
    g0,g1,g2,reset,current=(models[n] for n in ("g0","g1","g2","g2-restart","current"))
    digest=lambda n:sha(folder/(n+".json"))
    for child,parent in (("g1","g0"),("g2","g1"),("g2-restart","g0"),("current","g1")):
        require(models[child]["parent_sha256"]==digest(parent),"archived parent identity changed")
    reproduction=dict(g1=delta(g1,retrain(g0,first_raw,48119)),
                      g2=delta(g2,retrain(g1,next_raw,48120)),
                      restart=delta(reset,retrain(g0,next_raw,48120)))
    require(max(reproduction.values())<1e-10,"old recipe could not be reproduced")
    corrected=retrain(g0,first,48119)
    corrected["parent_sha256"]=digest("g0")
    write_receipt(folder/"g1-reencoded.json",corrected)
    recipes=((corrected,next_raw,"g1-reencoded"),(g1,next_course,"g1"),(corrected,next_course,"g1-reencoded"))
    impact={}
    for name,(parent,rows,tag) in zip(ARMS,recipes,strict=True):
        model=retrain(parent,rows,48120)
        model["parent_sha256"]=digest(tag)
        model["training"]=dict(rows=len(rows),updates=UPDATES,first_course=tag,
                               reencoded_second=rows is next_course)
        write_receipt(folder/(name+".json"),model)
        impact[name]=dict(max_parameter_delta=delta(g2,model),batch_rows=len(rows),updates=UPDATES)
    require(delta(g1,corrected)>0 and all(r["max_parameter_delta"]>0 for r in impact.values()),
            "re-encoding evidence did not propagate to fitted parameters")
    dependencies=dict(g1=dict(direct=first_changes,via_parent=[]),
        g2=dict(direct=next_changes,via_parent=first_changes),
        g2_restart=dict(direct=next_changes,via_parent=[]),
        current=dict(direct_audit="not tested",via_parent=first_changes))
    write_receipt(folder/"course-dependencies.json",dependencies)
    write_receipt(folder/"trained.json",{n:sha(folder/n) for n in
                  ("g1-reencoded.json","course-dependencies.json",*[a+".json" for a in ARMS])})
    return dict(max_recipe_error=reproduction,first_delta=delta(g1,corrected),impacts=impact,
                dependencies={k:{a:len(b) if isinstance(b,list) else b for a,b in v.items()}
                              for k,v in dependencies.items()})


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-course-correction"/os.environ["JUNQI_SERIAL_KEY"])
    folder=scope.folder
    if not audit:
        prepare(scope)
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"correction source changed")
    grid=[tuple(r) for r in read(folder/"grid.json")]
    models={n:read(folder/(n+".json")) for n in SNAPSHOTS}
    if audit:
        require(all(sha(folder/n)==v for n,v in read(folder/"trained.json").items()),"packet or corrected weights changed")
        models.update({n:read(folder/(n+".json")) for n in (*ARMS,"g1-reencoded")})
        result=dict(generation2=pair_panel(scope,grid,models,"g2",ARMS,"generation2.json"),
                    generation1=pair_panel(scope,grid,models,"g1",("g1-reencoded",),"generation1.json"))
        print("JUNQI_RESEARCH supported")
    else:
        result=learn(scope,grid,models)
    write_receipt(folder/("confirmation.json" if audit else "training.json"),result)
    print(json.dumps(result))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected archived feature correction phase")
    main(bool(sys.argv[1:]))
