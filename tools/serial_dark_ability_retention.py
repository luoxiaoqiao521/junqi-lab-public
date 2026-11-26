"""Check specific abilities before starting any anti-forgetting replay."""
from collections import Counter
import json
import os
import shutil
import sys

from full_mode_runtime import FullModeRun, sha
from serial_dark_coverage_origin import ARMS
from serial_dark_ordinary_learning import RULES
from serial_dark_partner_evidence import pairs, run
from serial_dark_verified_reuse import recorded_openings, validate_splits
from serial_dynamic_reply import ROOT, build, require
from serial_formation_matrix import contrast
from serial_worker import read, write_receipt


PANEL=ROOT/"build/dark-coverage-origin/test-four-dark-history-coverage-vs-return-shift-e8631093b1bf9ec8dea0"
TACTIC=ROOT/"build/dark-heritage-nominations/compare-four-dark-omitted-capture-outcomes-463cda51625500510dff"
PROBE=ROOT/"build/dark-partner-evidence/audit-four-dark-partner-resource-consumption-verified-72c1c7a0a84803011017/student.json"
MODELS=(*ARMS,"contact-control")
INPUTS=("parent.json","grid.json","known.json","engine/rules.json","engine/deadline_opening.exe",
        *[f"{n}.json" for n in MODELS])


def prepare(scope):
    folder=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    for name in ("parent",*ARMS):
        shutil.copyfile(PANEL/(name+".json"),folder/(name+".json"))
    shutil.copyfile(PROBE,folder/"contact-control.json")
    prior=read(TACTIC/"grid.json")
    known=read(TACTIC/"known.json")["summary"]["examples"]
    extra=read(TACTIC/"fresh.json")["summary"]["examples"]
    keys={r["seed"] for r in known+extra}
    known_grid=[r for stage in ("known","fresh") for r in prior[stage] if r[0] in keys]
    require(len(known_grid)==len(keys)==4,"strategic cases lost their source assignment")
    fresh=[(24177000+i,i%4,(i//4)%2,(16,32,48,64,88,112)[(i//4+i%4)%6]) for i in range(144)]
    grid=dict(development=known_grid,confirmation=fresh)
    used=read(PANEL/"grid.json")
    all_grids=dict(old_train=used["core"]+used["extra"],old_confirm=used["confirm"],
                   development=known_grid,confirm=fresh)
    mirror,boards=recorded_openings(scope,[r[0] for rows in all_grids.values() for r in rows])
    families=validate_splits({k:{r[0]:boards[r[0]] for r in batch} for k,batch in all_grids.items()},mirror)
    old_families={v for group in read(PANEL/"families.json").values() for v in group.values()}
    require(not (set(families["confirm"].values())&old_families),"ability confirmation overlaps consumed layouts")
    write_receipt(folder/"grid.json",grid)
    write_receipt(folder/"known.json",known+extra)
    write_receipt(folder/"fingerprints.json",{n:sha(folder/n) for n in INPUTS})


def signature(rows):
    return [(r["seed"],r["seat"],r["table"],r["ply"],r["valid"],r["stock"],r.get("before"),
             [(a["move"],a["label"],a["listed"],a["before"],a["x"]) for a in r["actions"]]) for r in rows]


def collision_report(rows):
    counts=Counter()
    effects=[]
    for r,a,b in pairs(rows):
        tags={n:a[n]>b[n] for n in ("before","after")}
        counts["paired_outcome_roots"]+=1
        counts["both_in_original_shortlist"]+=bool(a["listed"] and b["listed"])
        counts["parent_correct"]+=tags["before"]
        counts["candidate_correct"]+=tags["after"]
        counts["lost_previously_correct"]+=tags["before"] and not tags["after"]
        counts["newly_correct"]+=tags["after"] and not tags["before"]
        change=int(tags["after"])-int(tags["before"])
        effects.append((change,change))
    for r in rows:
        if not r["valid"]:
            continue
        counts["restored"]+=1
        counts["isolation_controls"]+=r["isolation"]
        selected={a["move"]:a["label"] for a in r["actions"]}
        counts["parent_picked_known_loss"]+=selected[r["before"]]==0
        counts["candidate_picked_known_loss"]+=selected[r["after"]]==0
    return dict(counts=counts,paired_interval=contrast(effects) if effects else None)


def strategic_report(rows,expectations):
    lookup={r["seed"]:r for r in rows}
    results=[]
    for expected in expectations:
        r=lookup[expected["seed"]]
        actions={a["move"]:a for a in r["actions"]}
        original,target=(actions[expected[k]] for k in ("old","proposed"))
        require(r["before"]==original["move"],"known strategic root changed before scoring")
        results.append(dict(seed=r["seed"],listed=bool(target["listed"]),
            parent_gap=target["before"]-original["before"],candidate_gap=target["after"]-original["after"],
            chosen=r["after"]==target["move"],reference=dict(same=expected["same"],changed_enemies=expected["different"])))
    return results


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-ability-retention"/os.environ["JUNQI_SERIAL_KEY"])
    folder=scope.folder
    if not audit:
        prepare(scope)
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"skill test fixed inputs changed")
    phase="confirmation" if audit else "development"
    grid=read(folder/"grid.json")[phase]
    parent=read(folder/"parent.json")
    control=None
    result={}
    raw={}
    for name in MODELS:
        rows,seconds=run(scope,grid,parent,read(folder/(name+".json")))
        marks=signature(rows)
        if control is None:
            control=marks
        require(marks==control,"ranking comparison did not use the same visible decision and candidate list")
        info=collision_report(rows)
        if not audit:
            info["strategic_cases"]=strategic_report(rows,read(folder/"known.json"))
        result[name]=dict(info,seconds=seconds)
        raw[name]=rows
    if audit:
        require(all(x["counts"]["isolation_controls"]>0 for x in result.values()),
                "fixed-root capability comparison did not run an isolation control")
        credible_loss=[n for n in ARMS if result[n]["paired_interval"] is not None and
                       result[n]["paired_interval"]["interval"][1]<0]
        print("JUNQI_RESEARCH "+("supported" if credible_loss else "inconclusive"))
    write_receipt(folder/(phase+".json"),dict(summary=result,rows=raw))
    print(json.dumps(result))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected ability retention phase")
    main(bool(sys.argv[1:]))
