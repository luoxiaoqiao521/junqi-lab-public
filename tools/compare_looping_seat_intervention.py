"""Test an exit from a repeated edge on the seat actually playing it."""
from collections import Counter
from hashlib import sha256
import json
from random import Random
from statistics import mean
import sys

from audit_public_attack_windows import write_json
from audit_local_selfplay_cycle_entry import RESULT as USED, SEED, read_output, selection
from calibrate_selfplay_evaluation_signal import NAMES, OUT, compile_probe, execute
from validate_layout_source_transfer import stat_bounds

RESULT = OUT / "cycle-actor.json"
KEY = ("block","table","rep")
ROOTS = ("block","table","actor","kind","neutral_from","neutral_to")
RETURN = (*KEY,"actual_back","route_back","control_back")
NEUTRAL = (*KEY,"actual_score","actual_enemy","actual_back","control_score","control_enemy","control_back")


def extract(data):
    a,e,u,left=[],[],[],[]
    tags=dict(A=(a,ROOTS),E=(e,RETURN),U=(u,NEUTRAL))
    for line in data.splitlines():
        tag,*parts=line.split()
        if tag in tags:
            rows,columns=tags[tag]
            rows.append(dict(zip(columns,map(int,parts),strict=True)))
        else:
            left.append(line)
    return (*read_output("\n".join(left)),a,e,u)


def effects(rows, chosen, reference):
    def delta(row):
        return stat_bounds(row[chosen+"_score"],row[reference+"_score"])
    per_layout=[tuple(map(mean,zip(*(delta(row) for row in rows if row["block"]==b))))
                for b in sorted({r["block"] for r in rows})]
    rng=Random(861611)
    samples=[rng.choices(per_layout,k=len(per_layout)) for _ in range(1200)] if per_layout else []
    bounds=list(map(mean,zip(*per_layout))) if per_layout else None
    lo=sorted(mean(x[0] for x in s) for s in samples)
    hi=sorted(mean(x[1] for x in s) for s in samples)
    return dict(bounds=bounds, interval=[lo[29],hi[1169]] if samples else None,
                improved=sum(0<=r[reference+"_score"]<r[chosen+"_score"] for r in rows),
                harmed=sum(0<=r[chosen+"_score"]<r[reference+"_score"] for r in rows),
                unknown=sum(min(r[chosen+"_score"],r[reference+"_score"])<0 for r in rows))


def summary(roots, pairs, returned, neutral, games):
    result={}
    focus={(r["block"],r["table"]):r["seat"] for r in games}
    for table,name in enumerate(NAMES):
        a=[r for r in roots if r["table"]==table]
        p=[r for r in pairs if r["table"]==table]
        e=[r for r in returned if r["table"]==table]
        u=[r for r in neutral if r["table"]==table]
        result[name]=dict(roots=len(a), kinds=dict(Counter(str(r["kind"]) for r in a)),
                          relative_actors=dict(Counter(str((r["actor"]-focus[r["block"],table])%4) for r in a)),
                          triples=len(p), controls_only=len(u),
                          returned={key:sum(r[key] for r in e) for key in RETURN[3:]},
                          retained={arm:sum(r[arm+"_enemy"]>0 and r[arm+"_persists"] for r in p)
                                    for arm in ("actual","route","control")},
                          used={arm:sum(r[arm+"_enemy"]>0 and r[arm+"_used"] for r in p)
                                for arm in ("actual","route","control")},
                          neutral_returned={arm:sum(r[arm+"_back"] for r in u) for arm in ("actual","control")},
                          route_vs_actual=effects(p,"route","actual"),
                          route_vs_control=effects(p,"route","control"),
                          control_vs_actual=effects(p,"control","actual"),
                          neutral_vs_actual=effects(u,"control","actual"))
    return result


def keys(rows, columns=KEY):
    return [tuple(r[x] for x in columns) for r in rows]


def reference():
    raw=USED.read_bytes()
    return json.loads(raw),sha256(raw).hexdigest()


def audit():
    base,digest=reference()
    r=json.loads(RESULT.read_text(encoding="utf8"))
    assert r["local_sha256"]==digest and r["rule_hash"]==base["rule_hash"]
    assert r["games"]==base["games"] and r["local"]==base["local"]
    roots,pairs,returned,neutral=(r[k] for k in ("roots","pairs","returned","neutral"))
    expected=[q for q in base["local"] if q["quiet_at"]]
    assert keys(roots,KEY[:2])==keys(expected,KEY[:2])
    late={(l["block"],l["table"]):l for l in r["late"]}
    for a,q in zip(roots,expected,strict=True):
        row=late[(a["block"],a["table"])]
        assert a["actor"]==q["quiet_seat"] and a["kind"] in (0,1,2)
        assert row["alert"]==q["quiet_at"] and row["alert_clock"]==q["quiet_clock"]
    roster=[(a["block"],a["table"],rep) for a in roots if a["kind"]==1 for rep in range(12)]
    assert keys(pairs)==keys(returned)==roster
    assert keys(r["offers"],KEY[:2])==keys([a for a in roots if a["kind"]==1],KEY[:2])
    assert keys(neutral)==[(a["block"],a["table"],i) for a in roots if a["kind"]==2 for i in range(12)]
    assert all(row[k] in (0,1) for row in returned for k in RETURN[3:])
    for rows, arms in ((pairs,("actual","route","control")),(neutral,("actual","control"))):
        assert all(row[a+"_score"] in (-1,0,1,2) for row in rows for a in arms)
    assert r["summary"]==summary(roots,pairs,returned,neutral,r["games"])
    print("PASS first looping actor, fixed policy assignment and matched branches",len(roots),len(pairs),len(neutral))


def run():
    base,digest=reference()
    selected,_=selection()
    assert base["archive"]==[k for _,k,_ in selected if k]
    output,elapsed=execute(selected,"cycle-actor",seed=SEED)
    rule,games,late,offers,pairs,local,roots,returned,neutral=extract(output.stdout)
    write_json(RESULT,dict(rule_hash=rule,local_sha256=digest,elapsed_seconds=elapsed,
                           games=games,late=late,offers=offers,pairs=pairs,local=local,roots=roots,
                           returned=returned,neutral=neutral,
                           summary=summary(roots,pairs,returned,neutral,games)))
    audit()


if __name__=="__main__":
    if sys.argv[1:]==["audit"]:
        audit()
    else:
        compile_probe()
        run()
