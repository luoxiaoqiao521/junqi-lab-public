"""Detect public quiet cycles and test matched route-changing alternatives."""
from collections import Counter
import json
from random import Random
from statistics import mean, median
import sys

from audit_public_attack_windows import write_json
from calibrate_selfplay_evaluation_signal import (
    GAME, N, NAMES, OUT, RESULT as CALIBRATION, compile_probe, execute, sources,
)
from compare_selfplay_stall_timing import RESULT as TIMING
from partner_crossplay import model_roster
from validate_layout_source_transfer import stat_bounds

RESULT = OUT / "cycle-entry.json"
SEED = 857400
LATE = ("block","table","first","returned","cycle_start","confirmed","period","alert","alert_clock")
OFFER = ("block","table","ply","clock","new_nodes","same_mover","actual_from","actual_to","route_from",
         "route_to","control_from","control_to")
STATUS = ("score","enemy","ally","persists","used")
ARMS = ("actual","route","control")
PAIR = ("block","table","rep",*(f"{a}_{x}" for a in ARMS for x in STATUS))


def selection():
    first=json.loads(CALIBRATION.read_text(encoding="utf8"))
    second=json.loads(TIMING.read_text(encoding="utf8"))
    selected,pool=sources(sample_seed=857391,exclude=first["archive"]+second["archive"])
    return selected,pool


def parse(output):
    game,late,offers,pairs = [],[],[],[]
    types=dict(G=(game,GAME),L=(late,LATE),O=(offers,OFFER),P=(pairs,PAIR))
    lines=output.splitlines()
    tag, rule=lines.pop(0).split()
    if tag!="R":
        raise ValueError("missing rule identity")
    for row in lines:
        tag,*values=row.split()
        out,keys=types[tag]
        out.append(dict(zip(keys,map(int,values),strict=True)))
    return rule,game,late,offers,pairs


def timeline(rows):
    def dist(key):
        values=[r[key] for r in rows if r[key]]
        return dict(count=len(values), median=median(values) if values else None,
                    earliest=min(values) if values else None)
    return {key:dist(key) for key in LATE[2:-1]}


def pairs_summary(offers,pairs,draws):
    grouped={(r["block"],r["table"]):[] for r in offers}
    for pair in pairs:
        grouped[(pair["block"],pair["table"])].append(pair)
    overall={}
    for alt,ref in (("route","actual"),("route","control"),("control","actual")):
        key=f"{alt}_minus_{ref}"
        delta=lambda r:stat_bounds(r[alt+"_score"],r[ref+"_score"])
        effects=[tuple(map(mean,zip(*(delta(row) for row in block),strict=True))) if block else (0,0)
                 for block in (grouped.get((b,draws),[]) for b in range(2*N))]
        rng=Random(857312)
        samples=[rng.choices(range(N),k=N)+rng.choices(range(N,2*N),k=N) for _ in range(1200)]
        lo,hi=zip(*effects,strict=True)
        overall[key]=dict(bounds=[mean(lo),mean(hi)],
                          interval=[sorted(mean(lo[i] for i in s) for s in samples)[29],
                                    sorted(mean(hi[i] for i in s) for s in samples)[1169]],
                          improved=sum(0<=r[ref+"_score"]<r[alt+"_score"] for r in pairs),
                          harmed=sum(0<=r[alt+"_score"]<r[ref+"_score"] for r in pairs),
                          unknown=sum(min(r[alt+"_score"],r[ref+"_score"])<0 for r in pairs))
    return overall


def summarized(games,late,offers,pairs):
    result={}
    for table,name in enumerate(NAMES):
        a=[r for r in late if r["table"]==table]
        b=[r for r in offers if r["table"]==table]
        c=[r for r in pairs if r["table"]==table]
        status={}
        for arm in ARMS:
            replied=[r for r in c if r[arm+"_enemy"]>0]
            status[arm]=dict(after_enemy=len(replied), reachable=sum(r[arm+"_persists"] for r in replied),
                             used=sum(r[arm+"_used"] for r in replied))
        result[name]=dict(natural=dict(Counter(str(r["score"]) for r in games if r["table"]==table)),
                          timeline=timeline(a),offers=len(b),same_mover=sum(r["same_mover"] for r in b),
                          pairs=len(c),routes=status, score=pairs_summary(b,c,table))
    return result


def audit(selected=None,pool=None):
    if selected is None:
        selected,pool=selection()
    r=json.loads(RESULT.read_text(encoding="utf8"))
    assert r["archive"]==[key for _,key,_ in selected if key] and r["pool"]==pool
    assert r["weights"]==model_roster()["candidate"]["artifact_sha256"]
    assert r["rule_hash"]==json.loads(CALIBRATION.read_text(encoding="utf8"))["rule_hash"]
    games,late,offers,pairs=(r[k] for k in ("games","late","offers","pairs"))
    expected=[(b,s,b%4,b//4%4,SEED+b,t) for b,(s,_,_) in enumerate(selected) for t in range(3)]
    assert [tuple(row[k] for k in GAME[:6]) for row in games]==expected
    ids=[(row["block"],row["table"]) for row in games]
    assert [(row["block"],row["table"]) for row in late]==ids
    for game, row in zip(games,late,strict=True):
        assert game["score"] in (-1,0,1,2) and 0<game["plies"]<=1600
        assert (row["first"]==row["returned"]==0) or 0<row["first"]<row["returned"]<=game["plies"]
        assert (row["confirmed"]==row["cycle_start"]==row["period"]==0) or (
            0<row["period"] and row["cycle_start"]+2*row["period"]==row["confirmed"]<=game["plies"])
        assert not row["alert"] or 0<row["returned"]<=row["alert"]<=game["plies"] and row["alert_clock"]<70
    roots={(row["block"],row["table"]):row for row in late if row["alert"]}
    offers_keys=[(row["block"],row["table"]) for row in offers]
    assert len(set(offers_keys))==len(offers_keys)
    for offer,key in zip(offers,offers_keys,strict=True):
        assert (offer["ply"],offer["clock"])==(roots[key]["alert"],roots[key]["alert_clock"])
        assert offer["new_nodes"]>0 and offer["same_mover"] in (0,1)
        assert len({(offer[arm+"_from"],offer[arm+"_to"]) for arm in ARMS})==3
    assert [(r["block"],r["table"],r["rep"]) for r in pairs]==[(*key,i) for key in offers_keys for i in range(12)]
    for pair in pairs:
        for arm in ARMS:
            assert pair[arm+"_score"] in (-1,0,1,2) and 0<=pair[arm+"_used"]<=pair[arm+"_persists"]<=1
    assert r["summary"]==summarized(games,late,offers,pairs)
    print("PASS cycle chronology, first eligible alerts and same-prefix branches",len(games),len(offers),len(pairs))
    print(json.dumps({name:dict(timeline=d["timeline"],offers=d["offers"],pairs=d["pairs"])
                      for name,d in r["summary"].items()},sort_keys=True))


def run():
    selected,pool=selection()
    result,elapsed=execute(selected,"cycle",seed=SEED)
    rule,games,late,offers,pairs=parse(result.stdout)
    write_json(RESULT,dict(rule_hash=rule,weights=model_roster()["candidate"]["artifact_sha256"],
                           pool=pool,archive=[k for _,k,_ in selected if k],games=games,
                           late=late,offers=offers,pairs=pairs,elapsed_seconds=elapsed,
                           summary=summarized(games,late,offers,pairs)))
    audit(selected,pool)


if __name__=="__main__":
    if sys.argv[1:]==["audit"]:
        audit()
    else:
        compile_probe()
        run()
