"""Separate the age of an ordinary course from the effect of its stage coverage."""
from collections import Counter, defaultdict
from itertools import product
import json
import os
from random import Random
import shutil
from statistics import mean
import sys

from full_mode_runtime import FullModeRun, sha
from serial_dark_asset_inheritance import pair_panel
from serial_dark_ordinary_exploration import collect
from serial_dark_ordinary_inheritance import difference, by_layout
from serial_dark_ordinary_learning import RULES
from serial_dark_partner_support import SOURCE
from serial_dark_verified_reuse import (TAPES, INPUTS, LEGACY, fit_packet, load_history,
    recorded_openings, validate_splits, check_negative_splits)
from serial_dynamic_reply import ROOT, build, require
from serial_formation_matrix import contrast
from serial_worker import read, write_receipt


ARMS=("new-natural","new-coverage","old-natural","old-coverage")
EXTRA=90
CORE=270


def prepare(scope):
    path=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    shutil.copyfile(SOURCE/"original.json",path/"parent.json")
    shutil.copyfile(SOURCE/"peer.json",path/"peer.json")
    for i in (1,2):
        dst=path/f"history-{i}"
        dst.mkdir()
        for n in LEGACY:
            shutil.copyfile(TAPES/f"generation-{i}"/n,dst/n)
    grid=dict(core=[(24173000+i,i%4,(i//4)%2) for i in range(170)],
              extra=[(24174000+i,i%4,(i//4)%2) for i in range(300)],
              confirm=[(24175000+i,(i+i//4)%4,t) for i,t in product(range(80),(0,2,3))])
    write_receipt(path/"grid.json",grid)
    write_receipt(path/"fingerprints.json",{n:sha(path/n) for n in INPUTS})


def stage(row):
    x=row["features"]
    ply=round(140*x[0])
    live=tuple(x[i]>0 for i in (26,29,32,35))
    if all(live):
        state="all-active"
    elif live[2]:
        state="ally-active"
    else:
        state="ally-gone"
    return (min(ply//48,2),state)


def group_rows(packets,layouts,seed):
    unseen=defaultdict(list)
    items=list(packets)
    Random(seed).shuffle(items)
    for p in items:
        r=p["data"]
        signature=stage(r)
        unseen[r["outcome"],signature].append((layouts[r["seed"]],p))
    return unseen


def sample_matched(current,old,core,priority,seed):
    rng=Random(seed)
    occupancy=Counter(stage(r["data"]) for r in core)
    needed=Counter(r["data"]["outcome"] for r in core)
    # Equal WDL target for all four arms; nearby states are paired by public stage.
    outcomes=[x for x in needed for _ in range(round(needed[x]*EXTRA/len(core)))]
    if len(outcomes)<EXTRA:
        outcomes.extend([max(needed,key=needed.get)]*(EXTRA-len(outcomes)))
    outcomes=outcomes[:EXTRA]
    rng.shuffle(outcomes)
    selected=[[],[]]
    visited=[set(),set()]
    for label in outcomes:
        support=[]
        for key,rows in current.items():
            if key[0]!=label:
                continue
            a=next(((f,r) for f,r in rows if f not in visited[0]),None)
            b=next(((f,r) for f,r in old[key] if f not in visited[1]),None) if key in old else None
            if a and b:
                support.append((key[1],a,b))
        require(support,"matched historical/current stage support exhausted")
        if priority:
            pick=min(support,key=lambda t:(occupancy[t[0]],rng.random()))
        else:
            weights=[sum(f not in visited[0] for f,r in current[label,t[0]]) for t in support]
            pick=rng.choices(support,weights=weights,k=1)[0]
        signature,a,b=pick
        occupancy[signature]+=1
        for i,(f,r) in enumerate((a,b)):
            visited[i].add(f)
            selected[i].append(r)
    return selected


def packets_for(scope,grid,models):
    histories=[load_history(scope,i) for i in (1,2)]
    old_openings={r["game"]["seed"]:r["board"] for tapes,_,_ in histories for r in tapes}
    mirror,openings=recorded_openings(scope,[r[0] for rows in grid.values() for r in rows])
    layouts=validate_splits(dict(history=old_openings,**{n:{r[0]:openings[r[0]] for r in rows}
                            for n,rows in grid.items()}),mirror)
    control=check_negative_splits(mirror,next(iter(old_openings.values())))
    logs={n:collect(scope,grid[n],models["parent"],False,b) for n,b in (("core",14000),("extra",25000))}
    new={n:[dict(data=r,status="current",origin=n) for r in log["examples"] if r["outcome"]>=0]
         for n,log in logs.items()}
    require(len(new["core"])>=CORE and all(v["left"]==0 and v["changed"]==0 for v in logs.values()),
            "fresh sample supply or native step ledger changed")
    Random(56617).shuffle(new["core"])
    core=new["core"][:CORE]
    a=group_rows(new["extra"],layouts["extra"],91132)
    old=[dict(p,origin=f"history-{i}") for i,(_,pool,_) in enumerate(histories,1) for p in pool]
    b=group_rows(old,layouts["history"],31613)
    new_natural,old_natural=sample_matched(a,b,core,False,9911)
    new_coverage,old_coverage=sample_matched(a,b,core,True,9911)
    tail=(new_natural,new_coverage,old_natural,old_coverage)
    packets={n:core+rows for n,rows in zip(ARMS,tail,strict=True)}
    require(len({tuple(sorted(Counter(r["data"]["outcome"] for r in v).items())) for v in packets.values()})==1,
            "outcome imbalance across the four arms")
    for first,second in ((tail[0],tail[2]),(tail[1],tail[3])):
        require([stage(r["data"]) for r in first]==[stage(r["data"]) for r in second],
                "new/history comparison changed the target stage")
    evidence=dict(steps=sum(log["steps"] for log in logs.values()),games=sum(
                  (log["games"] for log in logs.values()),[]),cohorts=logs)
    stats=fit_packet(scope,models["parent"],evidence,packets,layouts,control,[h[2] for h in histories])
    stats["strategic_stage_extra"]={n:{str(k):v for k,v in Counter(stage(r["data"]) for r in rows).items()}
                                   for n,rows in zip(ARMS,tail,strict=True)}
    return stats


def multiply_effect(outputs):
    reference=outputs["new-coverage"]["games"][::2]
    actual={"new-natural":reference,**{n:r["games"][1::2] for n,r in outputs.items()}}
    delta=lambda baseline,test:by_layout(list(zip(actual[baseline],actual[test],strict=True)))
    groups=defaultdict(list)
    for a,b,c,d in zip(*(actual[n] for n in ARMS),strict=True):
        b_lo,b_hi=difference(a,b)
        d_lo,d_hi=difference(c,d)
        groups[a["seed"]].append((d_lo-b_hi,d_hi-b_lo))
    interaction=contrast([tuple(map(mean,zip(*pairs,strict=True))) for pairs in groups.values()])
    return dict(current_stage=delta("new-natural","new-coverage"),
                historical_stage=delta("old-natural","old-coverage"),
                history_at_coverage=delta("new-coverage","old-coverage"),interaction=interaction)


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-coverage-origin"/os.environ["JUNQI_SERIAL_KEY"])
    folder=scope.folder
    if not audit:
        prepare(scope)
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"coverage experiment inputs changed")
    grid={k:[tuple(r) for r in v] for k,v in read(folder/"grid.json").items()}
    models={n:read(folder/(n+".json")) for n in ("parent","peer")}
    if audit:
        require(all(sha(folder/n)==v for n,v in read(folder/"trained.json").items()),"training artifacts changed")
        models.update({n:read(folder/(n+".json")) for n in ARMS})
        panel=pair_panel(scope,grid["confirm"],models,"new-natural",ARMS[1:],"games.json")
        effects=multiply_effect(read(folder/"games.json"))
        result=dict(panel=panel,effects=effects)
        interval=effects["historical_stage"]["interval"]
        print("JUNQI_RESEARCH "+("supported" if interval[0]>0 else "refuted" if interval[1]<0 else "inconclusive"))
    else:
        result=packets_for(scope,grid,models)
    write_receipt(folder/("confirmation.json" if audit else "training.json"),result)
    print(json.dumps(result))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected a coverage/origin phase")
    main(bool(sys.argv[1:]))
