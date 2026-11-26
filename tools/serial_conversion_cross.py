"""Replay the locked opening grid and cross two early and two conversion policies."""
from collections import Counter
import json
import os
from statistics import mean, median
import subprocess
import sys
from time import perf_counter

from serial_first_exit_approach import COUNT, END, KEY, START
from measure_deadline_opening import LABEL, ROOT, RULES, SCORE, build, file_sha, interval, model_roster, require
from serial_worker import write_receipt

REFERENCE = ROOT/"build/first-exit-approach/confirm-first-exit-approach-under-replies-b7bd4dce758ca8dc018d/paired.json"
SNAPSHOT = ("early_at", "early_move", "handoff", "checkpoint", "friends", "engineers", "bombs", "clock",
            "revealed", "first_enemy", "enemy_reason", "first_friendly", "friendly_reason")
ACTION = ("base_move", "played", "method", "samples")
ARMS = ("E0C0", "E0C1", "E1C0", "E1C1")


def parse(raw, reference):
    lines = raw.splitlines()
    require(len(lines)==2*len(reference)==2*COUNT*16,"incomplete four-arm grid")
    pairs = []
    for i, old in enumerate(reference):
        row = {k:old[k] for k in KEY} | {"at":old["at"], "handoffs":{}, "arms":{}}
        for early in (0,1):
            values = list(map(int,lines[2*i+early].split()))
            require(values[:5]==[*(old[k] for k in KEY),early],"misordered opening arm")
            snap = dict(zip(SNAPSHOT,values[5:5+len(SNAPSHOT)],strict=True))
            tail = values[5+len(SNAPSHOT):]
            require(len(tail)==2*(len(ACTION)+len(END)),"missing conversion arm")
            row["handoffs"][f"E{early}"] = snap
            expected_move = old["ordinary_move" if early==0 else "approach_move"] if old["at"] else -1
            require((snap["early_at"],snap["early_move"])==(old["at"],expected_move),"changed locked early action")
            for late in (0,1):
                size=len(ACTION)+len(END)
                end = dict(zip((*ACTION,*END),tail[late*size:(late+1)*size],strict=True))
                row["arms"][f"E{early}C{late}"]=end
                require(end["outcome"] in LABEL and end["ending"]>0 and 0<end["plies"]<=3000,"invalid ending")
                if not late:
                    require({k:end[k] for k in END}=={k:old["base" if early==0 else "trial"][k] for k in END},
                            "frozen conversion did not match previous full-game trace")
                require((bool(snap["handoff"]))==(end["played"]>=0),"invalid handoff or choice")
                require(not snap["handoff"] or (snap["first_enemy"]<snap["handoff"]<=end["plies"] and snap["friends"] in (1,2)),
                        "invalid handoff phase")
            a,b=(row["arms"][f"E{early}C{c}"] for c in (0,1))
            require(a["base_move"]==b["base_move"],"conversion arms consumed different prefix random streams")
            if a["played"]==b["played"]:
                require(all(a[k]==b[k] for k in END),"no-op conversion changed the path")
        if not old["at"]:
            require(row["handoffs"]["E0"]==row["handoffs"]["E1"] and all(
                row["arms"][f"E0C{c}"]==row["arms"][f"E1C{c}"] for c in (0,1)),"untriggered early arms diverged")
        pairs.append(row)
    return pairs


def effects(rows, coefficients):
    groups = {}
    for r in rows:
        combined = Counter()
        for arm,weight in coefficients.items():
            play=r["arms"][arm]
            combined[(play["outcome"],play["trace"])]+=weight
        lo=hi=0.
        for (outcome,_),weight in combined.items():
            value=SCORE[outcome]
            lo+=min(0,weight) if value is None else value*weight
            hi+=max(0,weight) if value is None else value*weight
        groups.setdefault(r["seed"],[]).append((lo,hi))
    if not groups:
        return None
    low,high=zip(*(tuple(mean(items) for items in zip(*values)) for values in groups.values()))
    return dict(layouts=len(groups),games=len(rows),bounds=[mean(low),mean(high)],grouped_95=[interval(low)[0],interval(high)[1]])


def paired_change(rows, early):
    base,trial=(f"E{early}C{c}" for c in (0,1))
    both_won=[r for r in rows if r["arms"][base]["outcome"]==r["arms"][trial]["outcome"]==2]
    return dict(score=effects(rows,{base:-1,trial:1}),
                changes=sum(r["arms"][trial]["played"]!=r["arms"][trial]["base_move"] for r in rows),
                methods=dict(Counter(str(r["arms"][trial]["method"]) for r in rows)),
                transitions=dict(Counter(f'{LABEL[r["arms"][base]["outcome"]]}->{LABEL[r["arms"][trial]["outcome"]]}' for r in rows)),
                common_wins=len(both_won),
                common_win_remaining_delta=median(r["arms"][trial]["plies"]-r["arms"][base]["plies"] for r in both_won)
                if both_won else None)


def summarize(rows):
    arms = {}
    for name in ARMS:
        values=[r["arms"][name] for r in rows]
        arms[name]=dict(terminal=dict(Counter(LABEL[a["outcome"]] for a in values)), observed_mean_plies=mean(a["plies"] for a in values),
                        horizons={str(h):dict(Counter(LABEL[a["outcome"]] if a["plies"]<=h else "ongoing" for a in values))
                                  for h in (100,140)})
    same, tables = {}, {}
    for early in (0,1):
        same[f"E{early}"]={str(friends):paired_change([r for r in rows if r["handoffs"][f"E{early}"]["handoff"] and
                                                      r["handoffs"][f"E{early}"]["friends"]==friends],early)
                           for friends in (1,2)}
    for p in (1,3):
        for t in (0,1):
            table=[r for r in rows if (r["partner"],r["table"])==(p,t)]
            tables[f"{p}/{t}"]=[effects(table,{f"E{e}C0":-1,f"E{e}C1":1}) for e in (0,1)]
    return dict(arms=arms, untriggered=sum(not r["at"] for r in rows), same_snapshot=same, tables=tables,
                conversion=[paired_change(rows,e) for e in (0,1)],
                early=[effects(rows,{f"E0C{c}":-1,f"E1C{c}":1}) for c in (0,1)],
                interaction=effects(rows,dict(zip(ARMS,(1,-1,-1,1),strict=True))),
                adverse_probe=[r for r in rows if tuple(r[k] for k in KEY)==(1203411,3,3,1)])


def main(audit):
    path=ROOT/"build/conversion-cross"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        path.mkdir(parents=True,exist_ok=False)
        build(path)
    exe=path/"deadline_opening.exe"
    reference=json.loads(REFERENCE.read_text(encoding="utf8"))
    roster=model_roster()
    models=[roster[name] for name in ("candidate","historical")]
    require(models==reference["models"] and file_sha(RULES)==reference["rules"],"locked rules or models changed")
    payload=" ".join(map(str,(START,COUNT,*(w for model in models for w in model["weights"]))))
    start=perf_counter()
    run=subprocess.run([exe,RULES,"cross-conversion"],cwd=ROOT,input=payload,encoding="utf8",capture_output=True,timeout=120)
    print(f"cross conversion seconds={perf_counter()-start:.2f}")
    require(run.returncode==0,f"conversion engine failed: {run.stderr.strip()}")
    rows=parse(run.stdout,reference["rows"])
    result=dict(engine=file_sha(exe),reference=file_sha(REFERENCE),summary=summarize(rows),rows=rows)
    if audit:
        require(run.stdout==(path/"cross.txt").read_text(encoding="utf8") and
                result==json.loads((path/"cross.json").read_text(encoding="utf8")),"crossed continuations did not replay")
    else:
        (path/"cross.txt").write_text(run.stdout,encoding="utf8")
        write_receipt(path/"cross.json",result)
    print(json.dumps(result["summary"],indent=2))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
