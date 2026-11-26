"""Check how much public collision evidence resolves the team value of a recapture."""
from collections import Counter, defaultdict
from itertools import product
import json
import os
import shutil
from statistics import mean
import sys

from full_mode_runtime import FullModeRun, isolated, sha
from serial_dark_ordinary_learning import RULES, assert_layout_separation, model_wire, run_native
from serial_dark_partner_support import SOURCE, assess, parse
from serial_dynamic_reply import ROOT, build, require
from serial_formation_matrix import contrast
from serial_worker import read, write_receipt
from train_early_deadline_student import model_arguments


EVIDENCE={-1:"unresolved",0:"loses_only",1:"may_remove",2:"removes_in_all_supported_cases"}
FILES=("original.json","mixed.json","peer.json","grid.json","engine/deadline_opening.exe","engine/rules.json")


def prepare(scope):
    folder=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    for name in ("original.json","mixed.json","peer.json"):
        shutil.copyfile(SOURCE/name,folder/name)
    grid={phase:[(start+i,i%4,t) for i,t in product(range(count),(0,2))]
          for phase,start,count in (("development",24135000,24),("confirmation",24136000,64))}
    assert_layout_separation(grid)
    write_receipt(folder/"grid.json",grid)
    write_receipt(folder/"fingerprints.json",{n:sha(folder/n) for n in FILES})


def screen(forks):
    by_class=defaultdict(lambda:defaultdict(list))
    policies=defaultdict(lambda:defaultdict(list))
    cases=Counter()
    for (seed,_,_),fork in forks.items():
        group=EVIDENCE[fork["evidence"]]
        both=[fork["effect"][t] for t in (0,2)]
        pooled=tuple(map(mean,zip(*both,strict=True)))
        by_class[group][seed].append(pooled)
        policies["all"][seed].append(pooled)
        policies["remove_proven"][seed].append(pooled if fork["evidence"]==2 else (0,0))
        policies["remove_possible"][seed].append(pooled if fork["evidence"]>=1 else (0,0))
        if all(lo>0 for lo,_ in both):
            cases[group+"/positive_in_both"]+=1
        if all(hi<0 for _,hi in both):
            cases[group+"/negative_in_both"]+=1
        if (both[0][0]>0 and both[1][1]<0) or (both[1][0]>0 and both[0][1]<0):
            cases[group+"/reversal"]+=1
    def panel(bands):
        return {g:dict(roots=sum(map(len,by_seed.values())),layouts=len(by_seed),
                       **contrast([tuple(map(mean,zip(*rows,strict=True))) for rows in by_seed.values()]))
                for g,by_seed in bands.items()}
    return dict(classes=panel(by_class),policies=panel(policies),cases=cases)


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-partner-conditions"/os.environ["JUNQI_SERIAL_KEY"])
    if not audit:
        prepare(scope)
    folder=scope.folder
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"public support inputs changed")
    stage="confirmation" if audit else "development"
    grid=[tuple(r) for r in read(folder/"grid.json")[stage]]
    original,mixed,peer=(read(folder/(n+".json")) for n in ("original","mixed","peer"))
    baseline,_,_,_=run_native(scope,"X",grid,(original,mixed),peer=peer)
    replicas=4
    wire=[model_wire(original),model_wire(mixed),model_arguments(peer,"small_nn")]
    wire+=[" ".join(map(str,(*r,replicas))) for r in grid]
    with isolated(scope,ROOT/"build"):
        raw=scope.run("dark-partner-support","\n".join(wire)+"\n")
    rows,forks=parse(raw,grid,replicas)
    require(forks,"condition contrast sampled no real recapture opportunities")
    summary=assess(rows,forks,baseline)
    tested=screen(forks)
    write_receipt(folder/(stage+".json"),dict(summary=summary,conditions=tested,
                  forks=[(*r,v) for r,v in forks.items()]))
    if audit:
        low,high=tested["policies"]["remove_proven"]["interval"]
        verdict=("supported" if low>0 else "refuted" if high<0 else "inconclusive")
        print("JUNQI_RESEARCH "+verdict)
    print(json.dumps(tested))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected public support condition stage")
    main(bool(sys.argv[1:]))
