"""Replay strict four-dark archive roots and isolate what each candidate consumer can know."""
from collections import Counter
from hashlib import sha256
from itertools import product
import json
import os
import shutil
from statistics import mean
import sys

from full_mode_runtime import FullModeRun,isolated,sha
from serial_capture_history import SOURCE
from serial_capture_world_coverage import MODEL,EXCHANGE
from serial_dynamic_reply import ROOT,build,require
from serial_formation_matrix import contrast,score_difference
from serial_worker import read,write_receipt
from train_early_deadline_student import RULES, model_arguments

ARMS=("path","snapshot","history","joint","ally_oracle","full_oracle")
D_KEYS="seat cut legal shortlist events changed_features worlds proposals nodes replay_failures unresolved sides".split()
SEEDS=(20100,20200)


def sources():
    rows=read(SOURCE/"probe.json")["rows"]
    pool=[r for r in rows if r["format"]==".jgs" and r["result"] in (1,2,3) and r["mid"] and
          not any(r[k] for k in ("prior_layout","prior_trajectory","stopped","notice"))]
    used=set()
    result=[]
    for index in range(2):
        selection=[]
        for phase in ("mid","post"):
            chosen=sorted((r for r in pool if r[phase] and r["family"] not in used),
                          key=lambda r:sha256((f"actual-dark-input-{index}-{phase}:"+r["family"]).encode()).digest())
            for r in chosen:
                if r["family"] in used:
                    continue
                used.add(r["family"])
                selection.append(dict(key=r["key"],family=r["family"],viewer=r["learner"],
                                      phase=phase,cut=r[phase]-1,direction=r["direction"],
                                      seed=SEEDS[index]+len(selection)))
                if len(selection)==(6 if phase=="mid" else 12):
                    break
        require(len(selection)==12,"not enough disjoint reconstructed archive layouts")
        result.append(selection)
    return result


def prepare(scope):
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    shutil.copyfile(MODEL,scope.folder/"student.json")
    cohorts=sources()
    seen={r["key"] for group in cohorts for r in group}
    tapes={}
    for line in (SOURCE/"games.txt").read_text(encoding="utf8").splitlines():
        kind,key,tape=line.split(maxsplit=2)
        if kind=="G" and key in seen:
            tapes[key]=tape
    require(tapes.keys()==seen,"missing frozen game prefix")
    for i,group in enumerate(cohorts):
        (scope.folder/f"replay-{i}.txt").write_text("\n".join(
            "{key} {viewer} {cut} {direction} {seed} ".format(**r)+tapes[r["key"]] for r in group)+"\n",encoding="ascii")
    write_receipt(scope.folder/"manifest.json",cohorts)
    names=("engine/deadline_opening.exe","engine/rules.json","student.json","replay-0.txt","replay-1.txt")
    write_receipt(scope.folder/"fingerprints.json",{n:sha(scope.folder/n) for n in names})


def read_roots(output,manifest):
    roots={r["key"]:dict(r,hypotheses={},future=[],outcomes=[]) for r in manifest}
    controls={}
    for tag,key,*parts in map(str.split,output.splitlines()):
        if tag=="C":
            controls[int(key)]=list(map(int,parts))
            continue
        r=roots[key]
        vals=list(map(int,parts))
        if tag=="D":
            scalars=dict(zip(D_KEYS,vals[:len(D_KEYS)],strict=True))
            require(scalars["seat"]==r["viewer"] and scalars["cut"]==r["cut"],"archive seat or cut changed")
            r.update(scalars,picks=dict(zip(ARMS,vals[len(D_KEYS):],strict=True)))
        elif tag=="H":
            a,n,k,lo,hi,work=vals
            require(a not in r["hypotheses"] and 0<=k<=n and 0<=lo<=hi<=2*n,"invalid hypothesis bound")
            r["hypotheses"][a]=dict(rolls=n,natural=k,bounds_times_two=[lo,hi],moves=work)
        elif tag=="I":
            r["future"].append(vals)
        else:
            require(tag=="Y","unexpected dark replay result")
            r["outcomes"].append(vals)
    for r in roots.values():
        combos=set(product(set(r["picks"].values()),range(4),range(4)))
        require(combos=={tuple(row[:3]) for row in r["outcomes"]} and len(combos)==len(r["outcomes"]) and
                len(r["future"])==r["worlds"] and len(r["hypotheses"])==r["shortlist"] and r["replay_failures"]<=4 and
                all(row["rolls"]==2*r["worlds"] for row in r["hypotheses"].values()),
                "missing or invalid candidate rollout or actor check")
    require(set(controls)==set(range(4)) and all(
        len(row)==9 and row[1+owner]==0 and row[5+owner]>0 and
        all(row[1+s]>0 and row[5+s]==0 for s in range(4) if s!=owner) for owner,row in controls.items()),
        "four-actor isolation control failed to separate hidden and own identity")
    return roots,controls


def summarize(roots):
    panels={}
    for t in range(4):
        panels[t]={}
        for a,b in (("path","snapshot"),("snapshot","history"),("history","joint"),
                    ("snapshot","ally_oracle"),("snapshot","full_oracle")):
            diffs=[]
            for r in roots.values():
                observed={(act,j):s for act,table,j,s,_ in r["outcomes"] if table==t}
                x,y=r["picks"][a],r["picks"][b]
                diffs.append(tuple(map(mean,zip(*[(0,0) if x==y else score_difference(observed[x,i],observed[y,i])
                                                 for i in range(4)],strict=True))))
            panels[t][f"{b}_vs_{a}"]=contrast(diffs)
    return dict(sources=len(roots),worlds=sum(x["worlds"] for x in roots.values()),
                roots_without_worlds=sum(not x["worlds"] for x in roots.values()),
                both_collision_sides=sum(x["sides"]==3 for x in roots.values()),
                read_events=sum(x["events"] for x in roots.values()),changed_features=sum(x["changed_features"] for x in roots.values()),
                moves_changed={arm:sum(x["picks"][arm]!=x["picks"]["snapshot"] for x in roots.values()) for arm in ARMS},
                actors_equal=[sum(y[i+1] for r in roots.values() for y in r["future"]) for i in range(4)],
                actors_private=[sum(y[i+5] for r in roots.values() for y in r["future"]) for i in range(4)],
                review_plies=sum(y[-1] for r in roots.values() for y in r["outcomes"]),
                truncations=sum(y[-2]<0 for r in roots.values() for y in r["outcomes"]),panels=panels)


def main(audit):
    folder=ROOT/"build/dark-information-audit"/os.environ["JUNQI_SERIAL_KEY"]
    scope=FullModeRun(folder)
    if not audit:
        prepare(scope)
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"frozen dark inputs changed")
    i=int(audit)
    manifest=read(folder/"manifest.json")[i]
    wire=model_arguments(read(folder/"student.json"),"collision",EXCHANGE)
    wire+="\n"+(folder/f"replay-{i}.txt").read_text(encoding="ascii")
    with isolated(scope,ROOT/"build"):
        output=scope.run("dark-info-audit",wire)
    roots,controls=read_roots(output,manifest)
    result=summarize(roots)
    result["synthetic_actor_checks"]=[sum(row[1+i] for row in controls.values()) for i in range(4)]
    require(any(result["actors_equal"]) and len(roots)==12,"no sequential private-state control")
    write_receipt(folder/("confirmation.json" if audit else "development.json"),dict(roots=roots,summary=result))
    if audit:
        old=read(folder/"development.json")
        require(set(r["family"] for r in old["roots"].values()).isdisjoint(r["family"] for r in roots.values()),
                "layout family reused between cohorts")
        require(old["summary"]["synthetic_actor_checks"]==result["synthetic_actor_checks"],
                "same controlled actor fixtures changed across archive cohorts")
        print("JUNQI_RESEARCH inconclusive")
    print(json.dumps(result))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no args or audit")
    main(bool(sys.argv[1:]))
