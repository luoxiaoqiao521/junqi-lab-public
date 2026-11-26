"""Compare a defender using paths with a defender also using public collisions."""
from collections import Counter, defaultdict
import json
import subprocess
import sys
import tempfile

from audit_public_attack_windows import OUT, write_json
from evidence_model import sha
from partner_crossplay import model_roster
from screen_attack_candidates import EXE, PLIES, RESULT as BASE, compile_probe, jqs_sources, qq_sources, send

RESULT=OUT/"public-defender.json"
PROFILES=("frozen","path","contact")


def collect(output):
    windows, evidence, branches = {}, {}, {}
    fixture=0
    for line in output.splitlines():
        kind,*data=line.split()
        if kind=="F":
            fixture+=int(data[0])
            continue
        group,ply,*raw=data
        key=f"{group}/{ply}"
        values=list(map(int,raw))
        if kind=="X":
            windows[key]=None
        elif kind=="D":
            names=("legal","routes","played_route","routes_rank","paired")
            windows[key]=dict(zip(names,values,strict=True))
        elif kind=="E":
            evidence[key]=values
        elif kind=="R":
            seed,root,*values=values
            branch=f"{key}/{seed}/{root}"
            if branch in branches or len(values)!=9:
                raise ValueError("invalid matched defense row")
            branches[branch]=[values[i:i+3] for i in (0,3,6)]
        else:
            raise ValueError("unexpected defender evidence")
    return dict(windows=windows,evidence=evidence,branches=branches,fixture=fixture)


def difference(a,b):
    return ((a if a>=0 else 0)-(b if b>=0 else 2))/2, ((a if a>=0 else 2)-(b if b>=0 else 0))/2


def summarize(data):
    evidence,branches=data["evidence"],data["branches"]
    result={}
    for family in ("jqh","qq"):
        marks=[v for key,v in evidence.items() if key.startswith(family)]
        rows={key:value for key,value in branches.items() if key.startswith(family)}
        result[family]=dict(windows=len(marks),narrowed=sum(v[0]>0 for v in marks),
                            singleton_pieces=sum(v[1] for v in marks),
                            threat_accuracy_changed=sum(v[2:4]!=v[4:6] for v in marks),
                            guard_changes=sum(value[1]!=value[2] for value in rows.values()),
                            branches=len(rows))
        profiles={}
        for i,name in enumerate(PROFILES):
            score=Counter(v[i][0] for v in rows.values())
            warns=sum(v[i][2] for v in rows.values())
            pairs=[(value[i][0],rows[key[:-1]+"1"][i][0]) for key,value in rows.items() if key.endswith("/0")]
            bounds=[difference(a,b) for a,b in pairs]
            profiles[name]=dict(outcomes=dict(score),changed=sum(v[i][1] for v in rows.values()),warnings=warns,
                                natural_pairs=sum(a>=0 and b>=0 for a,b in pairs),
                                bounds=[sum(row[j] for row in bounds)/len(bounds) for j in range(2)])
        result[family]["profiles"]=profiles
    return result


def run():
    selected=list(jqs_sources())+list(qq_sources())
    model=model_roster()["candidate"]
    with tempfile.TemporaryFile(mode="w+t",encoding="ascii",newline="\n",dir=OUT) as stream:
        print(len(selected)*len(PLIES),file=stream)
        print(*model["weights"],file=stream)
        for group,board,events in selected:
            for ply in PLIES:
                send(stream,group,board,events,ply)
        stream.seek(0)
        process=subprocess.run([EXE,"defend"],stdin=stream,capture_output=True,text=True,timeout=140)
    if process.returncode:
        raise RuntimeError(process.stderr.strip() or f"probe exited {process.returncode}")
    data=collect(process.stdout)
    data.update(baseline_sha256=sha(BASE), model=model["artifact_sha256"], summary=summarize(data))
    write_json(RESULT,data)
    audit()


def audit():
    data=json.loads(RESULT.read_text(encoding="utf-8"))
    base=json.loads(BASE.read_text(encoding="utf-8"))
    if (data["baseline_sha256"]!=sha(BASE) or data["windows"]!=base["sampled"] or
            data["model"]!=base["model"] or data["fixture"]!=1):
        raise ValueError("frozen route windows or the mine collision fixture changed")
    valid={key for key,row in data["windows"].items() if row is not None}
    if set(data["evidence"])!=valid:
        raise ValueError("public evidence did not cover the full valid sample")
    planned={key for key,row in data["windows"].items() if row and row["paired"]}
    expected={f"{k}/{s}/{arm}" for k in planned for s in range(8) for arm in (0,1)}
    if set(data["branches"])!=expected:
        raise ValueError("one route lacked the matching defender options")
    for profiles in data["branches"].values():
        if any(b[1]==0 and b!=profiles[0] for b in profiles[1:]):
            raise ValueError("outcome drift outside deliberate interventions")
    if data["summary"]!=json.loads(json.dumps(summarize(data))):
        raise ValueError("summary changed")
    print(json.dumps(data["summary"]))


if __name__=="__main__":
    if sys.argv[1:]==["audit"]:
        audit()
    else:
        compile_probe()
        run()
