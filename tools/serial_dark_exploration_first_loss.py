"""Reproduce the exploration regressions and separate first actions from later policy."""
from collections import Counter
import json
import os
import shutil
import sys

from full_mode_runtime import FullModeRun, isolated, sha
from serial_dark_asset_inheritance import RECENT
from serial_dark_first_loss import REPLICAS, parse, summarize
from serial_dark_ordinary_learning import RULES, model_wire, assert_layout_separation
from serial_dynamic_reply import ROOT, build, require
from serial_worker import read, write_receipt


INPUTS=("ordinary.json","explore.json","grid.json","engine/rules.json","engine/deadline_opening.exe")


def known_cases():
    report=read(RECENT/"games.json")
    games=report["ordinary_vs_exploration"]
    groups=[[],[]]
    for a,b,first in zip(games[::2],games[1::2],report["exploration_first"],strict=True):
        if a["table"]!=0 or (a["result"],b["result"]) not in ((2,0),(0,2)):
            continue
        key=int(a["result"]==0)
        groups[key].append(dict(first,control=[a,b]))
    require([len(g) for g in groups]==[13,2],"recorded ordinary/exploration reversals changed")
    return [*groups[0][:2],*groups[1]]


def prepare(scope):
    folder=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    for n in ("ordinary","explore"):
        shutil.copyfile(RECENT/(n+".json"),folder/(n+".json"))
    grids=dict(development=known_cases(),confirmation=[
        dict(seed=24170000+i,seat=i%4,table=0,ply=-1,parent=-1,child=-1) for i in range(48)])
    assert_layout_separation({k:[(r["seed"],) for r in rows] for k,rows in grids.items()})
    write_receipt(folder/"grid.json",grids)
    write_receipt(folder/"fingerprints.json",{n:sha(folder/n) for n in INPUTS})


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-exploration-first-loss"/os.environ["JUNQI_SERIAL_KEY"])
    folder=scope.folder
    if not audit:
        prepare(scope)
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"exploration fork inputs changed")
    phase="confirmation" if audit else "development"
    cases=read(folder/"grid.json")[phase]
    inputs=[model_wire(read(folder/(n+".json"))) for n in ("ordinary","explore")]
    fields=("seed","seat","table","ply","parent","child")
    inputs.extend(" ".join(map(str,(*(r[k] for k in fields),REPLICAS))) for r in cases)
    with isolated(scope,ROOT/"build"):
        raw=scope.run("dark-first-loss","\n".join(inputs)+"\n")
    roots=parse(raw,cases)
    summary=summarize(roots)
    if audit:
        effects=summary["aggregate"]
        require(bool(effects),"no real policy disagreements on new ordinary layouts")
        later=[effects[x]["interval"] for x in ("policy_old_move","policy_new_move")]
        verdict="supported" if all(x[1]<0 for x in later) else "inconclusive"
        print("JUNQI_RESEARCH "+verdict)
    write_receipt(folder/(phase+".json"),dict(summary=summary,roots=roots))
    print(json.dumps(summary))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected exploration first-loss phase")
    main(bool(sys.argv[1:]))
