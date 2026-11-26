"""Train the current student with strictly validated ordinary data and layout-family separation."""
from collections import Counter
from itertools import product
import hashlib
import json
import os
from random import Random
import shutil
import sys

from compare_book_learning import fit
from full_mode_runtime import FullModeRun, isolated, sha
from serial_dark_asset_inheritance import pair_panel
from serial_dark_course_tape import classify, replay, sample_key
from serial_dark_ordinary_exploration import collect
from serial_dark_ordinary_learning import RATE, RULES, UPDATES, examples_xy, model_wire
from serial_dark_partner_support import SOURCE
from serial_dynamic_reply import ROOT, build, require
from serial_worker import read, write_receipt


TAPES=ROOT/"build/dark-course-tape/replay-four-dark-archived-action-tapes-32ab03bdb0f50cddca94"
ARMS=("fresh","random-history","coverage-history")
LEGACY=("tapes.json","original.json","verified-pool.json")
INPUTS=("parent.json","peer.json","grid.json","engine/deadline_opening.exe","engine/rules.json",
        *[f"history-{i}/{n}" for i in (1,2) for n in LEGACY])
BUDGET=18000


def prepare(scope):
    folder=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    shutil.copyfile(SOURCE/"original.json",folder/"parent.json")
    shutil.copyfile(SOURCE/"peer.json",folder/"peer.json")
    for i in (1,2):
        dst=folder/f"history-{i}"
        dst.mkdir()
        for n in LEGACY:
            shutil.copyfile(TAPES/f"generation-{i}"/n,dst/n)
    grid=dict(train=[(24171000+i,i%4,(i//4)%2) for i in range(160)],
              confirm=[(24172000+i,(i+i//4)%4,t) for i,t in product(range(64),(0,2,3))])
    write_receipt(folder/"grid.json",grid)
    write_receipt(folder/"fingerprints.json",{n:sha(folder/n) for n in INPUTS})


def recorded_openings(scope,seeds):
    schedule=sorted(set(seeds))
    with isolated(scope,ROOT/"build"):
        raw=scope.run("dark-course-openings","\n".join(map(str,schedule))+"\n")
    cells={}
    mirror=None
    for tag,*record in map(str.split,raw.splitlines()):
        data=list(map(int,record))
        if tag=="M":
            require(mirror is None and sorted(data)==list(range(25)) and all(data[data[i]]==i for i in range(25)),
                    "invalid layout mirror")
            mirror=data
        else:
            require(tag=="L" and len(data)==101 and data[0] not in cells,"invalid generated opening")
            cells[data[0]]=data[1:]
    require(list(cells)==schedule and mirror is not None,"opening fingerprint assignments changed")
    return mirror,cells


def family(board,mirror):
    require(len(board)==100,"geometry fingerprint needs all four opening layouts")
    seats=[board[i:i+25] for i in range(0,100,25)]
    forms=[]
    for start in range(4):
        forms.append(bytes(q for u in range(4) for q in seats[(start+u)%4]))
        forms.append(bytes(seats[(start-u)%4][i] for u in range(4) for i in mirror))
    return hashlib.sha256(min(forms)).hexdigest()


def validate_splits(pools,mirror):
    usages={}
    keys={}
    for phase,cells in pools.items():
        keys[phase]={seed:family(board,mirror) for seed,board in cells.items()}
        for signature in keys[phase].values():
            require(usages.setdefault(signature,phase)==phase,"layout family crosses training and confirmation")
    require(len(set(keys["confirm"].values()))==len(keys["confirm"]),"confirmation contains duplicate layout families")
    return keys


def load_history(scope,number):
    folder=scope.folder/f"history-{number}"
    tape=read(folder/"tapes.json")
    frozen=read(folder/"verified-pool.json")
    require(frozen["report"]["provenance"]["tape_sha256"]==sha(folder/"tapes.json") and
            frozen["report"]["provenance"]["original_dataset_sha256"]==sha(folder/"original.json"),
            "the archived inputs do not match their provenance")
    original={sample_key(r):r for r in read(folder/"original.json")["examples"]}
    flags,reencoded=replay(scope,tape)
    admissible,_,reasons,_=classify(original,tape,flags,reencoded)
    require(admissible==frozen["admitted"],"stale feature/label versions in the archived course")
    return tape,admissible,dict(reasons)


def check_negative_splits(mirror,board):
    rotate=board[25:]+board[:25]
    mirror_board=[board[((4-s)%4)*25+i] for s in range(4) for i in mirror]
    expected=family(board,mirror)
    require(family(rotate,mirror)==expected==family(mirror_board,mirror),"geometry family control failed")
    try:
        validate_splits(dict(confirm={9:mirror_board},train={1:rotate}),mirror)
    except ValueError:
        return True
    raise ValueError("a mirrored training layout reached confirmation")


def sample_history(rows,layouts,count,recent):
    shuffled=list(rows)
    Random(75131).shuffle(shuffled)
    plain={}
    for r in shuffled:
        plain.setdefault(layouts[r["data"]["seed"]],r)
    uniform=list(plain.values())[:count]
    require(len(uniform)==count,"too few independent historical families")
    quota=Counter(r["data"]["outcome"] for r in uniform)
    phase=lambda r:round(r["features"][0]*140)//24
    exposure=Counter(phase(r) for r in recent)
    ranked=sorted(shuffled,key=lambda r:(r["status"]!="reencoded",exposure[phase(r["data"])]))
    chosen={}
    for r in ranked:
        row=r["data"]
        key=layouts[row["seed"]]
        if key not in chosen and quota[row["outcome"]]>0:
            chosen[key]=r
            quota[row["outcome"]]-=1
    require(len(chosen)==count and not any(quota.values()),"historical coverage matching failed")
    return uniform,list(chosen.values())


def learn(scope,grid,models):
    folder=scope.folder
    groups=[load_history(scope,i) for i in (1,2)]
    old={r["game"]["seed"]:r["board"] for p,_,_ in groups for r in p}
    mirror,fresh=recorded_openings(scope,[r[0] for g in grid.values() for r in g])
    pools={k:{r[0]:fresh[r[0]] for r in values} for k,values in grid.items()}
    keys=validate_splits(dict(history=old,**pools),mirror)
    negative=check_negative_splits(mirror,next(iter(old.values())))
    experience=collect(scope,grid["train"],models["parent"],False,BUDGET)
    require(experience["steps"]==BUDGET and experience["changed"]==0,"fresh data collection changed its behavior")
    samples=[r for r in experience["examples"] if r["outcome"]>=0]
    Random(47177).shuffle(samples)
    n=min(360,len(samples))
    require(n>=200,"fresh terminal course too small")
    replay_count=n//4
    current=samples[:n-replay_count]
    known=[r for _,pool,_ in groups for r in pool]
    uniform,ranked=sample_history(known,keys["history"],replay_count,current)
    require(Counter(r["data"]["outcome"] for r in uniform)==Counter(r["data"]["outcome"] for r in ranked),
            "coverage and uniform outcome mix differ")
    packets={"fresh":[dict(data=r,status="current") for r in samples[:n]],
             "random-history":[dict(data=r,status="current") for r in current]+uniform,
             "coverage-history":[dict(data=r,status="current") for r in current]+ranked}
    return fit_packet(scope,models["parent"],experience,packets,keys,negative,[g[2] for g in groups])


def fit_packet(scope,parent,experience,packets,families,negative,history):
    folder=scope.folder
    metrics={}
    for name,pool in packets.items():
        source=[r["data"] for r in pool]
        x,y=examples_xy(source,parent)
        require(len(x)==len(source),"unsettled rows reached the trainer")
        weights=fit(x,y,75481,24,parent["models"]["collision"],updates=UPDATES,rate=RATE)
        metrics[name]=dict(data_mix=Counter(r["status"] for r in pool),rows=len(x),updates=UPDATES,
                           returns=Counter(r["outcome"] for r in source),
                           stage_distribution=Counter(round(r["features"][0]*140)//24 for r in source))
        model=dict(parent,models={"collision":weights},parent_sha256=sha(folder/"parent.json"),
                   training=metrics[name])
        require(model_wire(model)!=model_wire(parent),"no trained parameter update")
        write_receipt(folder/(name+".json"),model)
    write_receipt(folder/"consumed.json",packets)
    write_receipt(folder/"families.json",families)
    write_receipt(folder/"fresh-experience.json",experience)
    write_receipt(folder/"trained.json",{n:sha(folder/n) for n in
                  (*[a+".json" for a in packets],"consumed.json","families.json","fresh-experience.json")})
    return dict(engine_steps=experience["steps"],fresh_games=len(experience["games"]),models=metrics,
                mirror_or_rotation_leak_rejected=negative,history=history)


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-verified-reuse"/os.environ["JUNQI_SERIAL_KEY"])
    folder=scope.folder
    if not audit:
        prepare(scope)
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"training evidence changed")
    grid={k:[tuple(r) for r in v] for k,v in read(folder/"grid.json").items()}
    models={n:read(folder/(n+".json")) for n in ("parent","peer")}
    if audit:
        require(all(sha(folder/n)==v for n,v in read(folder/"trained.json").items()),"weights or consumed cohort changed")
        models.update({a:read(folder/(a+".json")) for a in ARMS})
        result=pair_panel(scope,grid["confirm"],models,"fresh",("parent",*ARMS[1:]),"games.json")
        interval=result["coverage-history"]["aggregate"]["interval"]
        print("JUNQI_RESEARCH "+("supported" if interval[0]>0 else "refuted" if interval[1]<0 else "inconclusive"))
    else:
        result=learn(scope,grid,models)
    write_receipt(folder/("confirmation.json" if audit else "training.json"),result)
    print(json.dumps(result))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected verified history training stage")
    main(bool(sys.argv[1:]))
