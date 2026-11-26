"""Blind-fold audit of two formation banks and same-budget dark search."""
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from statistics import mean
import sys
from time import perf_counter

from archive_action_screen import select
from audit_annotated_replays import COLLECTIONS, DOWNLOADS, layout_audit, layout_files, mirror
from audit_qq_archive import INPUT, decode as replay_decode, public_records
from full_mode_runtime import FullModeRun, isolated, sha
from replay_deadline_transfer import encode_layout
from serial_dark_search_budget import CALIBRATION, decode, finish_roots
from serial_dynamic_reply import ROOT, build, require
from serial_formation_matrix import contrast, score_bounds, score_difference
from serial_replay_phase import INTAKE
from serial_worker import read, write_receipt
from train_early_deadline_student import RULES, model_arguments
from validate_layout_source_transfer import SWAPS, canonical


SEARCH=ROOT/"build/dark-search-budget"
ARMS=(("fixed","slots"),("focused","slots"),("fixed","recorded"),("focused","recorded"))
SOURCES=("recorded","collection")
KINDS=("exact","one_swap")
MATCH_FIELDS=("scanned","attempts","nodes","matched","bombs","unresolved","rejected","replay_failed")


def known_families():
    manifests=sorted(SEARCH.glob("compare-dark-search-budget-on-action-sensitive-roots-*/manifest.json"))
    require(bool(manifests),"previous search cohorts missing")
    families={r["family"] for r in read(CALIBRATION/"manifest.json")}
    for path in manifests:
        families.update(r["family"] for cohort in read(path) for r in cohort)
    return families


def select_roots(extra=(),count=32):
    blocked=known_families()|set(extra)
    rows=read(INTAKE/"probe.json")["rows"]
    pool=sorted((r for r in rows if r["format"]==".jgs" and r["result"] in (1,2,3) and r["mid"]>1 and
                 not any(r[k] for k in ("prior_layout","prior_trajectory","stopped","notice"))),
                key=lambda r:sha256(("formation-library-v1:"+r["family"]).encode()).digest())
    picked=[]
    for row in pool:
        if row["family"] in blocked:
            continue
        blocked.add(row["family"])
        picked.append({k:row[k] for k in ("key","family","learner","mid","direction")})
        if len(picked)==count:
            break
    require(len(picked)==count,"insufficient disjoint archive layout families")
    return [picked[:count//2],picked[count//2:]]


def freeze_replays(folder,cohorts):
    keys={r["key"] for group in cohorts for r in group}
    tapes={}
    for line in (INTAKE/"games.txt").read_text(encoding="utf8").splitlines():
        kind,key,tape=line.split(maxsplit=2)
        if kind=="G" and key in keys:
            tapes[key]=tape
    require(tapes.keys()==keys,"formation audit replay missing")
    for i,cohort in enumerate(cohorts):
        (folder/f"replay-{i}.txt").write_text("\n".join(
            f"{r['key']} {r['learner']} {r['mid']-1} {r['direction']} {25700+100*i+j} {tapes[r['key']]}"
            for j,r in enumerate(cohort))+"\n",encoding="ascii")
    return {layout for tape in tapes.values() for layout in replay_decode(bytes.fromhex(tape))[1]}


def nearby_layouts(holdout):
    forbidden={canonical(layout) for layout in holdout}
    for layout in holdout:
        trial=bytearray(layout)
        for i,j in SWAPS:
            if trial[i]==trial[j]:
                continue
            trial[i],trial[j]=trial[j],trial[i]
            forbidden.add(canonical(bytes(trial)))
            trial[i],trial[j]=trial[j],trial[i]
    return forbidden


def freeze_banks(folder,holdout):
    archive=select(public_records(Path(read(INPUT)["corpus"])))
    recorded={canonical(layout) for r in archive.values() for layout in r[1]}
    collection_counts,collection=layout_audit(raw for name in COLLECTIONS for raw in layout_files(DOWNLOADS/name))
    require(bool(recorded and collection),"played or collected formations unavailable")
    forbidden=nearby_layouts(holdout)
    metadata={"seats_held":len(holdout),"collection_scan":collection_counts,"played_game_families":len(archive),
              "shared_before_filter":len(recorded & collection)}
    with (folder/"library.txt").open("w",encoding="ascii") as sink:
        for name,layouts in zip(SOURCES,(recorded,collection),strict=True):
            retained=layouts-forbidden
            oriented=sorted(retained | {mirror(layout) for layout in retained})
            require(bool(oriented),"holdout filter removed an entire formation bank")
            metadata[name]=dict(families=len(layouts),excluded=len(layouts)-len(retained),oriented=len(oriented))
            print(len(oriented),file=sink)
            for layout in oriented:
                print(*encode_layout(layout),file=sink)
    return metadata


def prepare(scope,cohorts=None):
    folder=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    shutil.copyfile(CALIBRATION/"student.json",folder/"student.json")
    if cohorts is None:
        cohorts=select_roots()
    holdout=freeze_replays(folder,cohorts)
    write_receipt(folder/"manifest.json",dict(cohorts=cohorts,banks=freeze_banks(folder,holdout)))
    names=("engine/deadline_opening.exe","engine/rules.json","student.json","manifest.json",
           "replay-0.txt","replay-1.txt","library.txt")
    write_receipt(folder/"fingerprints.json",{name:sha(folder/name) for name in names})


def unpack(raw,manifest,fold):
    search=[]
    profiles={}
    loaded=[]
    for line in raw.splitlines():
        tag,key,*tokens=line.split()
        if tag=="L":
            loaded.append([int(key),int(tokens[0])])
        elif tag=="X":
            action,moved,truth=map(int,tokens)
            profiles[key]=dict(action=action,moved=bool(moved),truth=truth,matches={name:{} for name in SOURCES})
        elif tag=="F":
            name,kind,*counts=map(int,tokens)
            r=dict(zip(MATCH_FIELDS,counts,strict=True))
            require(r["attempts"]<=16 and r["bombs"]<=r["matched"] and r["attempts"]==
                    sum(r[k] for k in ("matched","unresolved","rejected","replay_failed")),"incomplete witness accounting")
            profiles[key]["matches"][SOURCES[name]][KINDS[kind]]=r
        else:
            search.append(line)
    require(len(loaded)==len(SOURCES),"both formation banks must be checked by the rules")
    banks={}
    for name,(asked,kept) in zip(SOURCES,loaded,strict=True):
        require(asked==manifest["banks"][name]["oriented"] and 0<kept<=asked,"wrong formation bank")
        banks[name]=dict(loaded=kept,illegal=asked-kept)
    roots=decode("\n".join(search),manifest["cohorts"][fold])
    active=finish_roots(roots)
    for row in active:
        profile=profiles[row["key"]]
        require(row["focus"]==profile["action"],"structural matcher and search evaluated different pieces")
    return roots,active,profiles,banks


def probabilities(profile,base):
    """A frozen shrinkage diagnostic. The unique-template counts are not posterior samples."""
    probs={"slots":base}
    for name,matches in profile["matches"].items():
        n=matches["exact"]["matched"]
        b=matches["exact"]["bombs"]
        if n:
            probs[name+"_exact"]=(b+8*base)/(n+8)
        n+=matches["one_swap"]["matched"]
        b+=matches["one_swap"]["bombs"]
        probs[name]=(b+8*base)/(n+8)
    return probs


def choose(row,search,weight):
    cells=row["arms"][search]["cells"]
    p=row["probabilities"][weight]
    return max(row["candidates"],key=lambda m:(
        (1-p)*cells[m][0][2]/cells[m][0][0]+p*cells[m][1][2]/cells[m][1][0],m==row["history"]))


def interaction(values,choices):
    coeffs=Counter([choices[0],choices[3]])
    coeffs.subtract([choices[1],choices[2]])
    lower=upper=0
    for move,coef in coeffs.items():
        low,high=score_bounds(values[move])
        lower+=coef*(low if coef>0 else high)
        upper+=coef*(high if coef>0 else low)
    return lower,upper


def paired(row,table):
    choices=list(row["choices"].values())
    values={(m,j):v for m,k,j,v,_ in row["outcomes"] if k==table}
    result={name:[] for name in ("search","prior","both","interaction")}
    for j in range(4):
        draws={m:values[m,j] for m in choices}
        for name,move in zip(("search","prior","both"),choices[1:],strict=True):
            result[name].append((0,0) if move==choices[0] else score_difference(draws[choices[0]],draws[move]))
        result["interaction"].append(interaction(draws,choices))
    return {name:tuple(map(mean,zip(*seq,strict=True))) for name,seq in result.items()}


def summarize(roots,active,profiles,banks):
    coverage={}
    for name in SOURCES:
        exact=sum(p["matches"][name]["exact"]["matched"]>0 for p in profiles.values())
        either=sum(any(v["matched"] for v in p["matches"][name].values()) for p in profiles.values())
        totals={kind:{f:sum(p["matches"][name][kind][f] for p in profiles.values()) for f in MATCH_FIELDS}
                for kind in KINDS}
        coverage[name]=dict(exact=exact,recovered=either-exact,uncovered=len(profiles)-either,rows=totals)
    calibration={}
    for name in (*SOURCES,*(s+"_exact" for s in SOURCES)):
        rows=[r for r in active if name in r["probabilities"]]
        if rows:
            calibration[name]=dict(roots=len(rows),bombs=sum(r["truth"] for r in rows),
                                   brier=mean((r["probabilities"][name]-r["truth"])**2 for r in rows),
                                   slots=mean((r["probabilities"]["slots"]-r["truth"])**2 for r in rows),
                                   mean_p=mean(r["probabilities"][name] for r in rows))
    tables={}
    for table in range(4):
        rows=[paired(r,table) for r in active]
        if rows:
            tables[table]={k:dict(**contrast([r[k] for r in rows]),positive=sum(r[k][0]>0 for r in rows),
                                  negative=sum(r[k][1]<0 for r in rows)) for k in rows[0]}
    return dict(registered=len(roots),probed=len(profiles),two_sided=len(active),banks=banks,coverage=coverage,
                calibration=calibration,tables=tables,changed={a:sum(r["choices"][a]!=r["choices"]["fixed_slots"] for r in active)
                    for a in ("focused_slots","fixed_recorded","focused_recorded")},
                validation_truncated=sum(y[-2]<0 for r in active for y in r["outcomes"]))


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-formation-library"/os.environ["JUNQI_SERIAL_KEY"])
    folder=scope.folder
    begin=perf_counter()
    if not audit:
        prepare(scope)
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"formation inputs changed")
    fold=int(audit)
    model=read(folder/"student.json")
    tape="\n".join((model_arguments(model,"collision",model["encoding"]),
                    (folder/"library.txt").read_text(encoding="ascii"),
                    (folder/f"replay-{fold}.txt").read_text(encoding="ascii")))
    with isolated(scope,ROOT/"build"):
        raw=scope.run("dark-formation-library",tape)
    roots,active,profiles,banks=unpack(raw,read(folder/"manifest.json"),fold)
    for row in active:
        profile=profiles[row["key"]]
        row["truth"]=profile["truth"]
        row["probabilities"]=probabilities(profile,2/row["slots"])
        row["choices"]={f"{a}_{b}":choose(row,a,b) for a,b in ARMS}
        require(row["choices"]["fixed_slots"]==row["arms"]["fixed"]["choice"] and
                row["choices"]["focused_slots"]==row["arms"]["focused"]["choice"],"unaltered search control drifted")
    stats=summarize(roots,active,profiles,banks)
    stats["seconds"]=perf_counter()-begin
    write_receipt(folder/("confirmation.json" if audit else "development.json"),
                  dict(roots=roots,profiles=profiles,summary=stats))
    if audit:
        supported=bool(stats["tables"]) and all(r["both"]["interval"][0]>0 for r in stats["tables"].values())
        print("JUNQI_RESEARCH "+("supported" if supported else "inconclusive"))
    print(json.dumps(stats))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no argument or audit")
    main(bool(sys.argv[1:]))
