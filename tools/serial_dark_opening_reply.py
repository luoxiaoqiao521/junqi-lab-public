"""Run the reply/value alternatives once at the first natural qualifying four-dark decision."""
from collections import Counter
from itertools import product
import json
import os
import shutil
import sys
from time import perf_counter

from full_mode_runtime import FullModeRun, isolated, sha
from serial_dark_reply_leaf import ARMS
from serial_dynamic_reply import ROOT, build, require
from serial_formation_matrix import contrast, score_difference
from serial_worker import read, write_receipt


RULES=ROOT/"config/ruleset-selfplay.json"
POLICIES=("frozen",*ARMS)
FIELDS=("seed learner table triggered ply learner_turns nodes contradicted unresolved replay_failed "
        "other bomb short_steps final_steps changed_replies uncertain_values").split()


def prepare(scope):
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    folds=[[(24048000+i, i%4) for i in range(24)],[(24049000+i, i%4) for i in range(32)]]
    folder=scope.folder
    write_receipt(folder/"manifest.json",folds)
    write_receipt(folder/"fingerprints.json",{n:sha(folder/n) for n in
                  ("manifest.json","engine/deadline_opening.exe","engine/rules.json")})


def parse(raw,expected):
    loaded=set()
    rows=[]
    for tag,*numbers in map(str.split,raw.splitlines()):
        require(tag=="G","unexpected opening output")
        values=list(map(int,numbers))
        r=dict(zip(FIELDS,values[:len(FIELDS)],strict=True))
        key=(r["seed"],r["learner"],r["table"])
        require(key in expected and key not in loaded,"unexpected or duplicated opening")
        loaded.add(key)
        tail=values[len(FIELDS):]
        games={name:dict(zip(("action","score","plies","trace"),batch,strict=True))
               for name,batch in zip(POLICIES,(tail[i:i+4] for i in range(0,len(tail),4)),strict=True)}
        ref=games["frozen"]
        recovered=r["other"] and r["bomb"]
        require(bool(r["short_steps"])==bool(recovered) and (r["triggered"] or not (r["other"] or r["bomb"])),
                "unsupported opening intervention")
        for g in games.values():
            require(-1<=g["score"]<=2 and r["ply"]<=g["plies"]<=400 and (r["triggered"]==(g["action"]>=0)),
                    "broken result score, trace or opening bound")
            if not recovered or g["action"]==ref["action"]:
                require(g==ref,"no-op opening control diverged")
        rows.append(dict(**r,policy=games))
    require(loaded==expected,"opening study lost assignments")
    return rows


def compare(rows,base,name):
    pairs=[(r["policy"][base],r["policy"][name]) for r in rows]
    differences=[(0,0) if a["action"]==b["action"] else score_difference(a["score"],b["score"]) for a,b in pairs]
    return dict(**contrast(differences),changed=sum(a["action"]!=b["action"] for a,b in pairs),
                worse=sum(hi<0 for _,hi in differences),better=sum(lo>0 for lo,_ in differences))


def summarize(rows):
    table={}
    for t in range(4):
        group=[r for r in rows if r["table"]==t]
        table[t]=dict(roots=len(group),triggered=sum(r["triggered"] for r in group),
                      compared=sum(r["other"] and r["bomb"] for r in group),
                      versus_frozen={name:compare(group,"frozen",name) for name in ARMS},
                      versus_leaf={name:compare(group,"natural_leaf",name) for name in ARMS[1:]},
                      outcomes={name:dict(Counter(r["policy"][name]["score"] for r in group)) for name in POLICIES})
    triggers=sorted(r["ply"] for r in rows if r["triggered"])
    counts={k:sum(r[k] for r in rows) for k in ("learner_turns","nodes","contradicted","unresolved","replay_failed",
                                                "short_steps","final_steps","changed_replies","uncertain_values")}
    return dict(counts,opened=len(rows),triggered=len(triggers),two_sided=sum(r["other"] and r["bomb"] for r in rows),
                trigger_plies=triggers,tables=table)


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-opening-reply"/os.environ["JUNQI_SERIAL_KEY"])
    folder=scope.folder
    if not audit:
        prepare(scope)
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"opening study inputs changed")
    blocks=read(folder/"manifest.json")[int(audit)]
    assignments=[(seed,seat,table) for (seed,seat),table in product(blocks,range(4))]
    wire="\n".join(" ".join(map(str,r)) for r in assignments)+"\n"
    started=perf_counter()
    with isolated(scope,ROOT/"build"):
        raw=scope.run("dark-opening-reply",wire)
    rows=parse(raw,set(assignments))
    summary=summarize(rows)
    summary["seconds"]=perf_counter()-started
    write_receipt(folder/("confirmation.json" if audit else "development.json"),dict(rows=rows,summary=summary))
    if audit:
        supported=any(all(t["versus_frozen"][a]["interval"][0]>0 and t["versus_leaf"][a]["interval"][0]>0
                          for t in summary["tables"].values()) for a in ARMS[1:])
        print("JUNQI_RESEARCH "+("supported" if supported else "inconclusive"))
    print(json.dumps(summary))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no argument or audit")
    main(bool(sys.argv[1:]))
