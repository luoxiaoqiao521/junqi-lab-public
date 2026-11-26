"""Verify actual four-dark ancestry, revalidate old rows and test cumulative learning."""
from itertools import product
import hashlib
import json
import os
from random import Random
import shutil
import subprocess
import sys

from compare_book_learning import fit
from full_mode_runtime import FullModeRun, sha
from serial_dark_first_loss import PREVIOUS as SECOND
from serial_dark_ordinary_inheritance import PREVIOUS as FIRST, by_layout
from serial_dark_ordinary_learning import RATE, RULES, UPDATES, assert_layout_separation, examples_xy, model_wire, run_native, summarize
from serial_dark_partner_support import SOURCE
from serial_dynamic_reply import ROOT, build, require
from serial_worker import read, write_receipt


RECENT=ROOT/"build/dark-ordinary-exploration/compare-four-dark-bounded-ordinary-exploration-91bf7af59274164210aa"
PHASE=ROOT/"build/dark-phase-coverage/compare-four-dark-ordinary-phase-coverage-3438dbf87c2b225545be"
SOURCES={**{f"{n}.json":p for n,p in {
    "g0":FIRST/"parent.json","g1":FIRST/"student.json","g2-inherit":SECOND/"inherit.json",
    "g2-restart":SECOND/"restart.json","g2-parent":SECOND/"generation1.json",
    "phase-parent":PHASE/"parent.json","mixed-parent":SOURCE/"parent.json",
    "current":SOURCE/"original.json","peer":SOURCE/"peer.json","recent":RECENT/"ordinary.json",
    "gen1-course":FIRST/"training.json","gen2-course":SECOND/"training.json",
    "recent-course":RECENT/"experience.json"}.items()}}
FROZEN=("g0","g1","g2-inherit","g2-restart")
NEW_ARMS=("continue","retrain","replay","current-next")
INPUTS=("grid.json","engine/deadline_opening.exe","engine/rules.json",*SOURCES)


def prepare(scope):
    folder=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    for name,source in SOURCES.items():
        shutil.copyfile(source,folder/name)
    grid={stage:[(base+i,(i+i//4)%4,t) for i,t in product(range(64),(0,2,3))]
          for stage,base in (("frozen",24165000),("learned",24166000))}
    assert_layout_separation(grid)
    write_receipt(folder/"grid.json",grid)
    write_receipt(folder/"fingerprints.json",{n:sha(folder/n) for n in INPUTS})


def release_probe(folder):
    target=folder/"release-practice"
    exe=ROOT/"junqi.exe"
    if not exe.is_file():
        exe=ROOT/"build/junqi.exe"
    subprocess.run(["powershell.exe","-NoProfile","-ExecutionPolicy","Bypass","-File",
                    str(ROOT/"start-mvp.ps1"),"-NoOpen","-Seed","24164901","-RunDirectory",str(target)],
                   cwd=ROOT,check=True,capture_output=True,timeout=45)
    log,=target.glob("*/game.jsonl")
    with log.open(encoding="utf8") as f:
        header=json.loads(f.readline())
    require(header["policy"]=="flag-route","release launcher selected an unexpected strategy")
    return dict(entry="start-mvp.ps1",strategy=header["policy"],binary=exe.relative_to(ROOT).as_posix(),
                binary_sha256=sha(exe),rules_hash=header["rules_hash"],observation_hash=header["observation_hash"],
                learned_weight_loaded=False)


def verify_lineage(folder,models):
    for name in ("g2-parent","phase-parent","mixed-parent"):
        require(model_wire(models[name])==model_wire(models["g1"]),f"first-generation identity differs at {name}")
    parents={"g1":"g0","g2-inherit":"g1","g2-restart":"g0","current":"g1","recent":"current"}
    for child,parent in parents.items():
        require(models[child]["parent_sha256"]==sha(folder/(parent+".json")),f"wrong parent fingerprint: {child}")
    return {n:dict(file_sha256=sha(folder/(n+".json")),load_sha256=hashlib.sha256(model_wire(m).encode()).hexdigest(),
                   encoding=m["encoding"],dimensions=len(m["mean"]),parent=parents.get(n))
            for n,m in models.items() if n!="peer"}


def verify_course(scope,name,model):
    original=read(scope.folder/(name+"-course.json"))
    grid=[(r["seed"],r["seat"],r["table"]) for r in original["games"]]
    games,new,_,runtime=run_native(scope,"D",grid,(model,))
    old=original["examples"]
    def key(r):
        return (r["seed"],r["seat"],r["table"],r["arm"],round(r["features"][0]*140))
    old_by={key(r):r for r in old}
    new_by={key(r):r for r in new}
    require(len(old_by)==len(old) and len(new_by)==len(new),"duplicate historical training point")
    exact={tuple(a[k] for k in ("seed","seat","table","arm"))
           for a,b in zip(original["games"],games,strict=True) if a["hash"]==b["hash"] and a["result"]==b["result"]}
    verified=[r for k,r in old_by.items() if k[:4] in exact and r==new_by.get(k)]
    changes=sum(k[:4] in exact and k in new_by and new_by[k]!=r for k,r in old_by.items())
    stats=dict(games=len(games),same_trajectory=len(exact),old_rows=len(old),new_rows=len(new),
               verified_rows=len(verified),changed_features_on_same_trajectory=changes,
               missing_rows_on_same_trajectory=sum(k[:4] in exact and k not in new_by for k in old_by),**runtime)
    write_receipt(scope.folder/(name+"-validated.json"),dict(rows=verified,stats=stats))
    return stats


def pair_panel(scope,grid,models,reference,candidates,file_name):
    series={}
    replay=None
    for name in candidates:
        a,_,first,runtime=run_native(scope,"X",grid,(models[reference],models[name]),peer=models["peer"])
        if replay is None:
            replay=a[::2]
        require(replay==a[::2],"shared frozen control changed across comparisons")
        pairs=list(zip(a[::2],a[1::2],strict=True))
        series[name]=dict(aggregate=by_layout(pairs),tables=summarize(a,first),games=a,first=first,runtime=runtime)
    write_receipt(scope.folder/file_name,series)
    return {name:{k:r[k] for k in ("aggregate","tables")} for name,r in series.items()}


def sample_recent(scope,size):
    logs=read(scope.folder/"recent-course.json")["ordinary"]
    rows=[r for r in logs["examples"] if r["outcome"]>=0]
    require(len(rows)>=size and logs["steps"]==22000,"current terminal corpus is unavailable")
    rng=Random(81291)
    rng.shuffle(rows)
    return rows[:size]


def isolate_budget_training(scope,models):
    folder=scope.folder
    old=read(folder/"gen1-validated.json")["rows"]
    require(len(old)>=96,"too few reverified rows for a historical training comparison")
    historical_name="g1"
    if len(old)!=models["g1"]["training"]["rows"]:
        historical_name="g1-rebuilt"
        original=models["g0"]
        x,y=examples_xy(old,original)
        params=fit(x,y,48119,24,original["models"]["collision"],updates=UPDATES,rate=RATE)
        models[historical_name]=dict(original,models={"collision":params},
              parent_sha256=sha(folder/"g0.json"),training={"rows":len(old),"updates":UPDATES})
        write_receipt(folder/(historical_name+".json"),models[historical_name])
    new=sample_recent(scope,len(old))
    require(not ({r["seed"] for r in old}&{r["seed"] for r in new}),"old and new courses overlap")
    half=len(old)//2
    # The interleaved restart consumes each row 48 times from G0, the same
    # cumulative exposures as G1 plus 48 updates on the new batch.
    recipes={
        "continue":(historical_name,[new]),
        "retrain":("g0",[old[:half]+new[:len(old)-half],old[half:]+new[len(old)-half:]]),
        "replay":(historical_name,[old[:len(old)//4]+new[len(old)//4:]]),
        "current-next":("current",[new])}
    metadata={}
    old_seeds={r["seed"] for r in old}
    for name,(parent,batches) in recipes.items():
        initial=models[parent]
        params=initial["models"]["collision"]
        for batch in batches:
            x,y=examples_xy(batch,initial)
            require(len(x)==len(old),"matched batch width changed")
            params=fit(x,y,80371,24,params,updates=UPDATES,rate=RATE)
        receipt=dict(weights_updated=True,updates=UPDATES*len(batches),cumulative_updates=144 if name=="current-next" else 96,
                     rows_per_update=len(old),parent=parent,
                     historical_in_current_batches=sum(r["seed"] in old_seeds for b in batches for r in b),
                     new_in_current_batches=sum(r["seed"] not in old_seeds for b in batches for r in b))
        model=dict(initial,models={"collision":params},parent_sha256=sha(folder/(parent+".json")),training=receipt)
        require(model_wire(model)!=model_wire(initial),"inheritance did not update weights")
        write_receipt(folder/(name+".json"),model)
        metadata[name]=receipt
    write_receipt(folder/"consumed.json",dict(old=old,new=new,recipes=metadata))
    write_receipt(folder/"learned-fingerprints.json",{n:sha(folder/n) for n in
                  (*[a+".json" for a in NEW_ARMS],"consumed.json")})
    return metadata


def main(stage):
    scope=FullModeRun(ROOT/"build/dark-asset-inheritance"/os.environ["JUNQI_SERIAL_KEY"])
    folder=scope.folder
    if stage=="assets":
        prepare(scope)
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"historical assets changed")
    models={n:read(folder/(n+".json")) for n in (*FROZEN,"current","g2-parent","phase-parent","mixed-parent","recent","peer")}
    grid={n:[tuple(r) for r in rows] for n,rows in read(folder/"grid.json").items()}
    if stage=="assets":
        result=dict(lineage=verify_lineage(folder,models),release=release_probe(folder),
                    courses={n:verify_course(scope,n,models[m]) for n,m in (("gen1","g0"),("gen2","g1"))})
        write_receipt(folder/"assets.json",result)
    elif stage=="frozen":
        result=pair_panel(scope,grid[stage],models,"current",FROZEN,"frozen.json")
    else:
        training=isolate_budget_training(scope,models)
        require(all(sha(folder/n)==v for n,v in read(folder/"learned-fingerprints.json").items()),"learned file changed")
        models.update({n:read(folder/(n+".json")) for n in NEW_ARMS})
        results=pair_panel(scope,grid[stage],models,"retrain",("continue","replay","current-next","current","g1"),"learned.json")
        result=dict(training=training,comparisons=results)
        best=results["continue"]["aggregate"]["interval"]
        print("JUNQI_RESEARCH "+("supported" if best[0]>0 else "refuted" if best[1]<0 else "inconclusive"))
    print(json.dumps(result))


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected a bounded inheritance stage")
    for stage in (("learned",) if sys.argv[1:] else ("assets","frozen")):
        main(stage)
    print("JUNQI_VERDICT accepted")
