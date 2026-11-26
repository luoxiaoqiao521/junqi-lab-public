"""Trace public recapture opportunities and vary first action and partner on the same root."""
from collections import Counter, defaultdict
from itertools import product
import json
import os
import shutil
from statistics import mean
import sys

from full_mode_runtime import FullModeRun, isolated, sha
from serial_dark_ordinary_learning import RULES, assert_layout_separation, model_wire, run_native
from serial_dynamic_reply import ROOT, build, require
from serial_formation_matrix import contrast, score_difference
from serial_worker import read, write_receipt
from train_early_deadline_student import model_arguments


SOURCE=ROOT/"build/dark-mixed-partner/compare-four-dark-mixed-partner-experience-10d5f3045448cd6a5746"
FILES=("original.json","mixed.json","peer.json","grid.json","engine/deadline_opening.exe","engine/rules.json")
TEAM=("result","plies","hash")
COUNTS=("possible","nominated","selected","resolved")


def key(row):
    return row["seed"],row["seat"],row["table"]


def prepare(scope):
    folder=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    for name in ("original.json","mixed.json","peer.json","confirmation.json"):
        shutil.copyfile(SOURCE/name,folder/("historical.json" if name=="confirmation.json" else name))
    development=[key(r) for r in read(folder/"historical.json")["games"][::2] if r["table"]==2]
    confirmation=[(24134000+i,i%4,t) for i,t in product(range(24),(0,2))]
    grid=dict(development=development,confirmation=confirmation)
    assert_layout_separation(grid)
    write_receipt(folder/"grid.json",grid)
    write_receipt(folder/"fingerprints.json",{n:sha(folder/n) for n in (*FILES,"historical.json")})


def parse(raw,grid,replicas):
    rows={}
    forks={}
    for kind,seed,seat,table,*tail in map(str.split,raw.splitlines()):
        root=tuple(map(int,(seed,seat,table)))
        if kind=="G":
            arm,*data=map(int,tail)
            require((root,arm) not in rows,"duplicate game audit")
            row=dict(zip((*TEAM,*COUNTS),data,strict=True))
            chain=[row[n] for n in COUNTS]
            require(chain==sorted(chain,reverse=True) and chain[-1]>=0,"invalid support counts")
            rows[root,arm]=row
        elif kind=="F":
            require(root not in forks,"duplicate counterfactual root")
            cut,a,b,evidence=map(int,tail)
            require(a!=b and evidence in (-1,0,1,2),"recapture fork has invalid public evidence or moves")
            forks[root]=dict(cut=cut,actions=[a,b],evidence=evidence,cells={})
        else:
            require(kind=="R","unexpected support wire")
            replicate,table_after,action,result,plies,trace=map(int,tail)
            require(forks[root]["cut"]<plies<=720 and result in (-1,0,1,2),"invalid support continuation")
            cells=forks[root]["cells"]
            cell=(replicate,table_after,action)
            require(cell not in cells,"duplicate support continuation")
            cells[cell]=(result,plies,trace)
    require(set(rows)==set(product(grid,(0,1))) and forks.keys()<=set(grid),"support replay coverage changed")
    require(all(f["cells"].keys()==set(product(range(replicas),(0,2),(0,1))) for f in forks.values()),
            "support continuation is missing a paired cell")
    return rows,forks


def assess(rows,forks,baseline):
    panels=defaultdict(Counter)
    for n,expected in enumerate(baseline):
        arm=n%2
        root=key(expected)
        r=rows[root,arm]
        require(all(r[k]==expected[k] for k in TEAM),"diagnostic changed actual game")
        changed=baseline[n-arm]["result"]==2 and baseline[n-arm+1]["result"]==0
        groups=(f"{root[2]}/all",f"{root[2]}/win-to-loss") if changed else (f"{root[2]}/all",)
        for g in groups:
            panels[f"{g}/{arm}"].update(dict(games=1,**{k:r[k] for k in COUNTS}))
    deltas=defaultdict(list)
    for root,fork in forks.items():
        cells=fork["cells"]
        per_table={t:[] for t in (0,2)}
        for i in range(len(cells)//4):
            for t in per_table:
                before,after=(cells[i,t,a] for a in (0,1))
                per_table[t].append((0,0) if before[2]==after[2] else score_difference(before[0],after[0]))
        per_table["interaction"]=[(b[0]-a[1],b[1]-a[0]) for a,b in zip(*per_table.values(),strict=True)]
        fork["effect"]={m:tuple(map(mean,zip(*v,strict=True))) for m,v in per_table.items()}
        for m,v in fork["effect"].items():
            deltas[root[2],m].append(v)
        fork["cells"]=[(*k,*v) for k,v in cells.items()]
    return dict(panels=panels,branch_contrasts={f"{t}/{m}":dict(roots=len(v),**contrast(v))
                                               for (t,m),v in deltas.items()})


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-partner-support"/os.environ["JUNQI_SERIAL_KEY"])
    if not audit:
        prepare(scope)
    folder=scope.folder
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"support study inputs changed")
    stage="confirmation" if audit else "development"
    grid=[tuple(r) for r in read(folder/"grid.json")[stage]]
    original,mixed,peer=(read(folder/(n+".json")) for n in ("original","mixed","peer"))
    if audit:
        baseline,_,_,_=run_native(scope,"X",grid,(original,mixed),peer=peer)
    else:
        baseline=[r for r in read(folder/"historical.json")["games"] if r["table"]==2]
    replicas=4
    wire=[model_wire(original),model_wire(mixed),model_arguments(peer,"small_nn")]
    wire+=[" ".join(map(str,(*r,replicas))) for r in grid]
    with isolated(scope,ROOT/"build"):
        raw=scope.run("dark-partner-support","\n".join(wire)+"\n")
    rows,forks=parse(raw,grid,replicas)
    summary=assess(rows,forks,baseline)
    write_receipt(folder/(stage+".json"),dict(summary=summary,games=[(*r,a,v) for (r,a),v in rows.items()],
                                             forks=[(*r,v) for r,v in forks.items()]))
    if audit:
        limits=[r["interval"] for name,r in summary["branch_contrasts"].items() if not name.endswith("interaction")]
        verdict=("supported" if len(limits)==4 and all(lo>0 for lo,hi in limits) else
                 "refuted" if any(hi<0 for lo,hi in limits) else "inconclusive")
        print("JUNQI_RESEARCH "+verdict)
    print(json.dumps(summary))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected support study stage")
    main(bool(sys.argv[1:]))
