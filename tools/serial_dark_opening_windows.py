"""Compare early and middle-game entry along identical frozen opening trajectories."""
from itertools import product
import json
import os
import shutil
from statistics import median
import sys
from time import perf_counter

from full_mode_runtime import FullModeRun, isolated, sha
from serial_dark_opening_reply import ARMS, RULES, parse, summarize
from serial_dynamic_reply import ROOT, build, require
from serial_formation_matrix import contrast, score_difference
from serial_worker import read, write_receipt


WINDOWS=(0,72)


def prepare(scope):
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    folder=scope.folder
    folds=[[(24051000+i,i%4) for i in range(16)],[(24052000+i,i%4) for i in range(24)]]
    write_receipt(folder/"manifest.json",folds)
    write_receipt(folder/"fingerprints.json",{n:sha(folder/n) for n in
                  ("manifest.json","engine/deadline_opening.exe","engine/rules.json")})


def read_windows(raw,assignments):
    lines={w:[] for w in WINDOWS}
    for line in raw.splitlines():
        tag,window,played=line.split(" ",2)
        require(tag=="W","missing intervention window")
        lines[int(window)].append(played)
    early,middle=[sorted(parse("\n".join(lines[w]),set(assignments)),key=lambda r:(r["seed"],r["table"]))
                  for w in WINDOWS]
    for a,b in zip(early,middle,strict=True):
        assert_frozen(a,b)
    return early,middle


def assert_frozen(early,middle):
    a,b=(r["policy"]["frozen"] for r in (early,middle))
    require(early["seed"]==middle["seed"] and early["table"]==middle["table"] and
            (a["score"],a["trace"],a["plies"])==(b["score"],b["trace"],b["plies"]),
            "moving the entry window changed the frozen whole game")
    require(early["ply"]<=middle["ply"] and (not middle["triggered"] or middle["ply"]>=72),"unexpected entry time")


def summarize_windows(early,middle):
    paired={t:{} for t in range(4)}
    for t in paired:
        pairs=[(a,b) for a,b in zip(early,middle,strict=True) if a["table"]==t]
        for arm in ARMS:
            diffs=[]
            for a,b in pairs:
                g,h=(r["policy"][arm] for r in (a,b))
                diffs.append((0,0) if g["trace"]==h["trace"] else score_difference(g["score"],h["score"]))
            paired[t][arm]=dict(**contrast(diffs),better=sum(lo>0 for lo,_ in diffs),worse=sum(hi<0 for _,hi in diffs))
    stats={w:summarize(r) for w,r in zip(WINDOWS,(early,middle),strict=True)}
    for s in stats.values():
        points=s.pop("trigger_plies")
        s["median_trigger"]=median(points) if points else None
    return dict(windows=stats,paired=paired,same_frozen=len(early))


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-opening-windows"/os.environ["JUNQI_SERIAL_KEY"])
    folder=scope.folder
    if not audit:
        prepare(scope)
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"opening window inputs changed")
    assignments=[(seed,seat,t) for (seed,seat),t in product(read(folder/"manifest.json")[int(audit)],range(4))]
    wire="\n".join(" ".join(map(str,r)) for r in assignments)+"\n"
    started=perf_counter()
    with isolated(scope,ROOT/"build"):
        output=scope.run("dark-opening-reply-window",wire)
    early,middle=read_windows(output,assignments)
    stats=summarize_windows(early,middle)
    stats["seconds"]=perf_counter()-started
    write_receipt(folder/("confirmation.json" if audit else "development.json"),dict(early=early,middle=middle,summary=stats))
    if audit:
        supported=any(all(t[a]["interval"][0]>0 for t in stats["paired"].values()) for a in ARMS)
        print("JUNQI_RESEARCH "+("supported" if supported else "inconclusive"))
    print(json.dumps(stats))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no argument or audit")
    main(bool(sys.argv[1:]))
