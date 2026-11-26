"""Check the actual proof-first selector separately from defensive estimation."""
from collections import Counter
import json
import os
import sys

from measure_human_winner_takeover import (CUTS, MODEL, ROOT, analyze_native, build, comparison, encode,
                                           model_arguments, native, persist, require, source_frames, speed)
from measure_deadline_opening import model_roster


HISTORICAL = ROOT/"build/human-winner-takeover/measure-human-winner-takeover-cb07059a3f34683e3c18/summary.json"
ARMS = (0,4,5)


def params(seed, count, model, name="replay_control"):
    roster=model_roster()
    weights=[w for key in ("candidate","historical") for w in roster[key]["weights"]]
    return " ".join(map(str,(seed,count,*weights)))+" "+model_arguments(model,name)


def select(ledger):
    previous=json.loads(HISTORICAL.read_bytes())["manifest"]
    excluded={r["family"] for r in previous.values()}
    selected={k:r for k,r in ledger.items() if r["pilot"]}
    for group,cap in (("winning",16),("comeback",8)):
        added=0
        for key,row in sorted(ledger.items(),reverse=True):
            if row["group"]!=group or row["family"] in excluded:
                continue
            selected[key]=row
            excluded.add(row["family"])
            added+=1
            if added==cap:
                break
        require(added==cap,"underfilled independent archive roster")
    return selected


def validate_proof(cases):
    seen=Counter()
    for row in cases.values():
        for rep in range(len(row["runs"])//len(ARMS)):
            old,proof,guard=(row["runs"][f"{rep}/{arm}"] for arm in ARMS)
            require(not proof["guards"] and not guard["corrections"],"proof and defense interventions mixed")
            if proof["known_win"]:
                require(proof["known_win"]==1 and proof["outcome"]==2 and
                        proof["earliest_visible_win"]==proof["plies"],"proof did not end its own game")
                seen["immediate_wins"]+=1
                seen["different_trace"]+=proof["trace"]!=old["trace"]
            else:
                require(all(proof[k]==old[k] for k in ("trace","outcome","plies")),"selector changed without proof")
            if not guard["guards"]:
                require(guard["trace"]==old["trace"],"defensive arm changed without a reported guard")
    return dict(seen)


def main(audit):
    directory=ROOT/"build/finish-and-conversion"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        build(directory)
    ledger,_=source_frames()
    selected=select(ledger)
    payload=[params(1,1,json.loads(MODEL.read_bytes()))]
    for key,row in sorted(selected.items()):
        payload.append(f'{key} {int(row["group"]=="comeback")} {2 if row["pilot"] else 4} {row["tape"]}')
    cases,rejected=analyze_native(native(directory,"winner-finish-confirm","\n".join(payload)),selected,ARMS)
    receipt={}
    for panel in ("pilot",*CUTS):
        group={(k,c):r for (k,c),r in cases.items() if
               panel==("pilot" if selected[k]["pilot"] else CUTS[c])}
        rows=[(k,r) for (k,_),r in group.items()]
        receipt[panel]=dict(roots=len(rows),proof=validate_proof(group),
                            profiles={a:speed(rows,a) for a in ARMS},
                            pair={a:comparison(rows,a) for a in (4,5)})
    require(sum(x["proof"].get("immediate_wins",0) for x in receipt.values())>0,"no direct proof exercised")
    summary=dict(panels=receipt,rejected=rejected,
                 sources={k:{"family":r["family"],"group":r["group"]} for k,r in selected.items()},
                 rows=[dict(key=k,cut=c,**r) for (k,c),r in cases.items()])
    persist(directory/"finish.json",encode(summary),audit)
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
