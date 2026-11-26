"""Audit and mutate the versioned collision semantics without altering live source."""
from collections import Counter
from itertools import product
import json
import os
from pathlib import Path
from shutil import copytree
import subprocess
import sys
from tempfile import TemporaryDirectory

from serial_formation_matrix import contrast, score_difference
from serial_route_choice import ROOT, RULES, build, native, require, write_receipt

PREVIOUS = ROOT/"build/full-mode-baseline/audit-full-mode-baseline-c74dbef116bb99782125/relocated copy with spaces/junqi.exe"
N = 16


def semantics(directory):
    raw = native(directory,"collision-semantics","")
    require(raw[-1:]==[["T","bomb-flag-terminal-before-local-score"]],"missing guaranteed team finish control")
    rows = [list(map(int,fields)) for tag,*fields in raw[:-1] if tag=="M"]
    require([r[:2] for r in rows]==[[i,120] for i in range(8)],"matrix did not cover each combat variant")
    return rows


def mutation_controls(directory):
    variants = {
        "rank_order": ("score.objective = known_collision_objective(rank,target,*rules);",
                       "score.objective = target < rank ? 60 : -60;"),
        "frozen_rules": ("const auto outcome=resolve(attacker,defender,rules);",
                         "(void)rules;\n  const auto outcome=resolve(attacker,defender,Rules{});"),
    }
    killed = []
    for variant,(before,after) in variants.items():
        with TemporaryDirectory(dir=directory) as scratch:
            work = Path(scratch)
            for folder in ("core","sim","strategy","observe","tests"):
                copytree(ROOT/folder,work/folder)
            source = work/"strategy/heuristic.h"
            content = source.read_text(encoding="utf8")
            require(content.count(before)==1,"mutation destination is ambiguous")
            source.write_text(content.replace(before,after),encoding="utf8")
            build(work,work)
            result = subprocess.run([work/"deadline_opening.exe",RULES,"collision-semantics"],input="",encoding="utf8",
                                    capture_output=True,timeout=20)
            require(result.returncode==1 and "known collision objective differs" in result.stderr,
                    f"valid mutant was not detected by the expected assertion: {variant}")
            killed.append(variant)
    return killed


def formation(directory, start):
    rows = [list(map(int,numbers)) for tag,*numbers in native(directory,"formation-collision-impact",f"{start} {N}\n") if tag=="G"]
    require([r[:5] for r in rows]==[[start+i,i%4,s,p,2] for i in range(N) for s,p in product((0,2),(0,3))] and
            all(r[5] in (-1,0,1,2) for r in rows),"missing registered paired formation")
    cells = {}
    for row in rows:
        cells.setdefault((row[2],row[3]),[]).append(row)
    totals = {f"{shape}/{policy}":dict(Counter(r[5] for r in games)) for (shape,policy),games in cells.items()}
    gaps = {}
    for policy in (0,3):
        gaps[policy] = [score_difference(closed[5],inside[5]) for closed,inside in zip(cells[0,policy],cells[2,policy],strict=True)]
    interaction = [(newlo-oldhi,newhi-oldlo) for (newlo,newhi),(oldlo,oldhi) in zip(gaps[3],gaps[0],strict=True)]
    return dict(rows=rows,totals=totals,gaps={policy:contrast(ds) for policy,ds in gaps.items()},interaction=contrast(interaction))


def historical_full_log(directory):
    with TemporaryDirectory(dir=directory) as scratch:
        old = Path(scratch)/"legacy.jsonl"
        corrected = Path(scratch)/"corrected.jsonl"
        exe = ROOT/"build/junqi.exe"
        conf = [RULES,ROOT/"config/observation.json"]
        for engine,policy,path in ((PREVIOUS,"heuristic-all",old),(exe,"heuristic-2-all",corrected)):
            subprocess.run([engine,"simulate-full","42",path,*conf,policy],check=True,capture_output=True,timeout=20)
            subprocess.run([exe,"replay",path],check=True,capture_output=True,timeout=20)
        legacy = Path(scratch)/"baseline-repeat.jsonl"
        subprocess.run([exe,"simulate-full","42",legacy,*conf,"heuristic-all"],check=True,capture_output=True,timeout=20)
        require(legacy.read_bytes()==old.read_bytes(),"frozen old heuristic failed bitwise compatibility")
        a,b = ([json.loads(line) for line in p.read_bytes().splitlines()] for p in (old,corrected))
        require(a[0]["layouts"]==b[0]["layouts"] and a[1]==b[1],"starting state drift")
        first = next((i for i,(x,y) in enumerate(zip(a[2:],b[2:]),1) if x["event"]!=y["event"]),None)
        return dict(legacy_byte_identical=True,first_different_move=first,legacy_plies=len(a)-2,corrected_plies=len(b)-2)


def main(audit):
    out=ROOT/"build/collision-semantics"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        out.mkdir(parents=True,exist_ok=True)
        build(out)
        report = dict(matrix=semantics(out),mutants=mutation_controls(out))
        write_receipt(out/"controls.json",report)
    else:
        initial = json.loads((out/"controls.json").read_bytes())
        require(semantics(out)==initial["matrix"],"matrix did not reproduce")
        dev = formation(out,2293200)
        confirm = formation(out,2293600)
        require(formation(out,2293200)==dev,"formation development did not reproduce")
        report = dict(development=dev,confirmation=confirm,legacy=historical_full_log(out))
        write_receipt(out/"confirmation.json",report)
        print("JUNQI_RESEARCH supported")
    print(json.dumps({k:({m:v for m,v in x.items() if m!="rows"} if isinstance(x,dict) else x) for k,x in report.items()}))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected audit or no argument")
    main(bool(sys.argv[1:]))
