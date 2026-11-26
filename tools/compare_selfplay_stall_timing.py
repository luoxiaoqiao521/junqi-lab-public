"""Test a single documented intervention before and near the quiet-move deadline."""
from collections import defaultdict
from hashlib import sha256
import json
from random import Random
from statistics import mean
import sys

from audit_public_attack_windows import write_json
from audit_selfplay_stall_decisions import read_output, summary
from calibrate_selfplay_evaluation_signal import (
    GAME, N, NAMES, OUT, RESULT as USED, compile_probe, execute, sources,
)
from validate_layout_source_transfer import stat_bounds

RESULT = OUT / "timing.json"
SEED = 843400
WINDOWS = {"earlier": ("stall-mid",30,39), "near_draw": ("stall",60,69)}


def audit_window(data, games, low, high):
    assert data["games"]==games
    ids = [(r["block"],r["table"]) for r in games]
    assert [(r["block"],r["table"]) for r in data["late"]]==ids
    offer_ids = [(r["block"],r["table"]) for r in data["offers"]]
    assert len(offer_ids)==len(set(offer_ids))
    assert all(low<=r["clock"]<=high and r["kind"] in (1,2) for r in data["offers"])
    assert [(r["block"],r["table"],r["rep"]) for r in data["pairs"]]==[
        (*key,i) for key in offer_ids for i in range(12)]
    assert all(r["actual"] in (-1,0,1,2) and r["alternative"] in (-1,0,1,2)
               for r in data["pairs"])
    assert data["summary"]==summary(data["late"],data["offers"],data["pairs"])


def mean_bounds(rows, table):
    outcomes=defaultdict(list)
    for row in rows["pairs"]:
        if row["table"]==table:
            outcomes[row["block"]].append(stat_bounds(row["alternative"],row["actual"]))
    return {b:tuple(map(mean,zip(*pairs))) for b,pairs in outcomes.items()}


def layout_effects(earlier, late):
    rng=Random(843430)
    samples=[rng.choices(range(N),k=N)+rng.choices(range(N,2*N),k=N) for _ in range(1200)]
    result={}
    for table,name in enumerate(NAMES):
        lanes=[]
        for phase in (earlier,late):
            observed=mean_bounds(phase,table)
            lanes.append([observed.get(block,(0,0)) for block in range(2*N)])
        phases=dict(zip(WINDOWS,lanes,strict=True))
        phases["earlier_minus_near"]=[(a[0]-b[1],a[1]-b[0]) for a,b in zip(*lanes,strict=True)]
        result[name]={label:interval(bounds,samples) for label,bounds in phases.items()}
    return result


def interval(bounds,samples):
    low,high=zip(*bounds)
    a=sorted(mean(low[i] for i in indices) for indices in samples)
    b=sorted(mean(high[i] for i in indices) for indices in samples)
    return dict(bounds=[mean(low),mean(high)], block_interval=[a[29],b[1169]])


def frozen_sources():
    raw=USED.read_bytes()
    old=json.loads(raw)
    selected,pool=sources(sample_seed=843391, exclude=old["archive"])
    return old,sha256(raw).hexdigest(),selected,pool


def run():
    old, digest, selected, pool=frozen_sources()
    results={}
    for label,(mode,_,_) in WINDOWS.items():
        process,elapsed=execute(selected,mode,seed=SEED)
        rule,games,late,offers,pairs=read_output(process.stdout)
        results[label]=dict(rule_hash=rule,elapsed_seconds=elapsed,games=games,late=late,
                            offers=offers,pairs=pairs,summary=summary(late,offers,pairs))
    write_json(RESULT,dict(calibration_sha256=digest, pool=pool,
                           archive=[k for _,k,_ in selected if k], windows=results,
                           effects=layout_effects(results["earlier"],results["near_draw"])))
    audit(old,digest,selected,pool)


def audit(old=None,digest=None,selected=None,pool=None):
    if old is None:
        old,digest,selected,pool=frozen_sources()
    r=json.loads(RESULT.read_text(encoding="utf8"))
    assert r["archive"]==[k for _,k,_ in selected if k] and len(set(r["archive"]))==N
    assert not set(r["archive"]).intersection(old["archive"])
    assert r["calibration_sha256"]==digest and r["pool"]==pool
    windows=r["windows"]
    games=windows["earlier"]["games"]
    expected=[(b,s,b%4,b//4%4,SEED+b,t) for b,(s,_,_) in enumerate(selected) for t in range(3)]
    assert [tuple(row[k] for k in GAME[:6]) for row in games]==expected
    for label,(_,low,high) in WINDOWS.items():
        assert windows[label]["rule_hash"]==old["rule_hash"]
        audit_window(windows[label],games,low,high)
    assert r["effects"]==layout_effects(windows["earlier"],windows["near_draw"])
    print("PASS independent layout groups",len(selected),"identical natural games",len(games))
    print(json.dumps(r["effects"],sort_keys=True))


if __name__=="__main__":
    if sys.argv[1:]==["audit"]:
        audit()
    else:
        compile_probe()
        run()
