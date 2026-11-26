"""Locate sustained seat-local backtracking and test quiet escape options."""
from collections import Counter
import json
from statistics import median
import sys

from audit_public_attack_windows import write_json
from audit_selfplay_cycle_entry import RESULT as GLOBAL, ARMS, parse, summarized
from calibrate_selfplay_evaluation_signal import (
    GAME,N,NAMES,OUT,RESULT as CALIBRATION,compile_probe,execute,sources,
)
from compare_selfplay_stall_timing import RESULT as TIMING
from partner_crossplay import model_roster

RESULT = OUT / "local-cycle-entry.json"
SEED = 861400
LOCAL = ("block","table","first","established","quiet_first","quiet_at","quiet_clock",
         "quiet_seat","node_a","node_b","warning","check_at","prediction")


def selection():
    exclude = []
    for path in (CALIBRATION,TIMING,GLOBAL):
        exclude.extend(json.loads(path.read_text(encoding="utf8"))["archive"])
    return sources(sample_seed=861391,exclude=exclude)


def timings(games,rows):
    result={}
    for table,name in enumerate(NAMES):
        a=[r for r in rows if r["table"]==table]
        def measure(key):
            x=[r[key] for r in a if r[key]>0]
            return dict(count=len(x),median=median(x) if x else None, earliest=min(x) if x else None)
        b=[g for g in games if g["table"]==table]
        result[name]={key:measure(key) for key in ("first","established","quiet_first","quiet_at",
                                                  "quiet_clock","warning","check_at")}
        result[name]["prediction"]=dict(Counter(str(r["prediction"]) for r in a if r["warning"]))
        result[name]["quiet_cycles_by_final_result"]=dict(Counter(str(g["score"])
                   for g,r in zip(b,a,strict=True) if r["quiet_at"]))
        gaps=[r["quiet_at"]-r["quiet_first"] for r in a if r["quiet_at"]]
        result[name]["quiet_confirmation_delay"]=median(gaps) if gaps else None
    return result


def read_output(data):
    other, local = [], []
    for line in data.splitlines():
        if line.startswith("Q "):
            local.append(dict(zip(LOCAL,map(int,line.split()[1:]),strict=True)))
        else:
            other.append(line)
    return (*parse("\n".join(other)),local)


def audit(selected=None,pool=None):
    if selected is None:
        selected,pool=selection()
    r=json.loads(RESULT.read_text(encoding="utf8"))
    assert r["archive"]==[key for _,key,_ in selected if key] and len(set(r["archive"]))==N and r["pool"]==pool
    assert r["weights"]==model_roster()["candidate"]["artifact_sha256"]
    assert r["rule_hash"]==json.loads(CALIBRATION.read_text(encoding="utf8"))["rule_hash"]
    games,late,offers,pairs,local=(r[k] for k in ("games","late","offers","pairs","local"))
    expected=[(b,s,b%4,b//4%4,SEED+b,t) for b,(s,_,_) in enumerate(selected) for t in range(3)]
    assert [tuple(g[k] for k in GAME[:6]) for g in games]==expected
    ids=[(g["block"],g["table"]) for g in games]
    assert all([(row["block"],row["table"]) for row in seq]==ids for seq in (late,local))
    for g,ln,q in zip(games,late,local,strict=True):
        assert (q["first"]==q["established"]==0) or 0<q["first"]<q["established"]<=g["plies"]
        assert q["prediction"] in (-1,0,1)
        assert (q["prediction"]==-1)==(q["check_at"]==0)
        assert not q["check_at"] or 0<q["warning"]<q["check_at"]<=g["plies"]
        if q["quiet_at"]:
            assert q["established"]<=q["quiet_at"]<=g["plies"]
            assert 0<q["quiet_at"]-q["quiet_first"]<=q["quiet_clock"]<70
            assert 0<q["warning"]<q["quiet_at"] and 0<=q["quiet_seat"]<4 and q["node_a"]!=q["node_b"]
        assert not ln["alert"] or 0<q["quiet_at"]<=ln["alert"]<=g["plies"] and ln["alert_clock"]<70
    anchors={(row["block"],row["table"]):row for row in late}
    offer_ids=[(row["block"],row["table"]) for row in offers]
    assert len(set(offer_ids))==len(offers)
    for row,key in zip(offers,offer_ids,strict=True):
        anchor=anchors[key]
        assert row["ply"]==anchor["alert"] and row["clock"]==anchor["alert_clock"] and row["new_nodes"]>0
        assert len({(row[a+"_from"],row[a+"_to"]) for a in ARMS})==3
    assert [(r["block"],r["table"],r["rep"]) for r in pairs]==[(*key,i) for key in offer_ids for i in range(12)]
    assert all(row[a+"_score"] in (-1,0,1,2) and 0<=row[a+"_used"]<=row[a+"_persists"]<=1
               for row in pairs for a in ARMS)
    assert r["summary"]==summarized(games,late,offers,pairs) and r["timings"]==timings(games,local)
    print("PASS local warning order, actual returns and matched root actions",len(games),len(offers),len(pairs))
    print(json.dumps(r["timings"],sort_keys=True))


def run():
    selected,pool=selection()
    output,elapsed=execute(selected,"cycle-local",seed=SEED)
    rule,games,late,offers,pairs,local=read_output(output.stdout)
    write_json(RESULT,dict(rule_hash=rule,weights=model_roster()["candidate"]["artifact_sha256"],
                           pool=pool,archive=[key for _,key,_ in selected if key],
                           games=games,late=late,offers=offers,pairs=pairs,local=local,
                           elapsed_seconds=elapsed,timings=timings(games,local),
                           summary=summarized(games,late,offers,pairs)))
    audit(selected,pool)


if __name__=="__main__":
    if sys.argv[1:]==["audit"]:
        audit()
    else:
        compile_probe()
        run()
