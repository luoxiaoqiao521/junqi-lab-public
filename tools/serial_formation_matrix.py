"""Measure controlled mine swaps and policy interactions in full information."""
from collections import Counter
from itertools import product
import json
import os
from pathlib import Path
from random import Random
from statistics import mean
import subprocess
import sys
from tempfile import TemporaryDirectory

from measure_deadline_opening import ROOT, RULES, build, require
from serial_worker import write_receipt

COUNT = 32
SEEDS = (2113700, 2114100)
SHAPES = ("closed", "top", "inside", "outside")
PROFILES = ("legacy", "collision-aware", "flag-route")
CELLS = tuple(product(range(4), range(3), (0,2)))


def native(directory, mode, payload=""):
    result = subprocess.run([directory/"deadline_opening.exe",RULES,mode],input=payload,encoding="utf8",
                            capture_output=True,timeout=95)
    require(result.returncode==0,f"{mode} failed: {result.stderr.strip()}")
    return [line.split() for line in result.stdout.splitlines()]


def rules_controls(directory):
    *cases, generated = native(directory,"formation-rules")
    require([int(row[1]) for row in cases]==list(range(4)) and all(
        row[0]=="R" and int(row[2])>0 and int(row[3])<0 for row in cases),"missing rotated mine and flag controls")
    tag,n,tables,layouts,*coverage = generated
    require(tag=="S" and int(n)==512 and int(tables)>1 and int(layouts)>4 and sum(map(int,coverage))==512*4,
            "layout stream appears fixed or incomplete")
    return dict(controls=[list(map(int,row[1:])) for row in cases],games=int(n),unique_tables=int(tables),
                unique_layouts=int(layouts),mine_coverage=list(map(int,coverage)))


def cli_pair(directory):
    with TemporaryDirectory(dir=directory) as scratch:
        openings = []
        for name,mode,seed in (("dark","simulate",50000),("repeat","simulate",50000),
                               ("full","simulate-full",50000),("next","simulate",50001)):
            path = Path(scratch)/(name+".jsonl")
            subprocess.run([ROOT/"build/junqi.exe",mode,str(seed),path,RULES,ROOT/"config/observation.json","random"],
                           check=True,capture_output=True,timeout=20)
            with path.open(encoding="utf8") as replay:
                header,frame = json.loads(next(replay)),json.loads(next(replay))
            openings.append((header["layouts"],frame["hash"],sum(v>=0 for v in frame["views"][0])))
        require(openings[0]==openings[1] and openings[0][:2]==openings[2][:2] and openings[0][0]!=openings[3][0] and
                [o[2] for o in openings]==[25,25,100,25],"CLI seed or observation contract changed")
        return dict(same_seed_repeats=True,different_seed_changes_table=True,visible_counts=[o[2] for o in openings])


def collect(directory, seed):
    initial, games = [], []
    for tag,*fields in native(directory,"formation-matrix",f"{seed} {COUNT}\n"):
        record = tuple(map(int,fields))
        if tag=="N":
            initial.append(record)
        else:
            require(tag=="G" and len(record)==11,"unexpected formation outcome")
            games.append(record)
    require(len(initial)==COUNT*len(SHAPES),"missing formation starts")
    expected = []
    for i in range(COUNT):
        seat = i%4
        first = (seat+i//4)%4
        column = initial[i*4][3]
        expected.extend((seed+i,seat,first,column,shape) for shape in range(4))
    require([r[:5] for r in initial]==expected and all(r[3] in (1,3) for r in initial) and
            len({r[5] for r in initial})==len(initial),"duplicate or unbalanced initial conditions")
    require([r[:5] for r in games]==[(seed+i,i%4,*cell) for i in range(COUNT) for cell in CELLS],"missing paired game")
    require(all(r[5] in (-1,0,1,2) and r[6]>0 and all(0<=p<=r[6] for p in r[7:]) for r in games),"bad settlement data")
    return dict(initial=initial,games=games)


def show_cell(cell):
    shape, own, foe = cell
    return "/".join((SHAPES[shape],PROFILES[own],PROFILES[foe]))


def score_bounds(score):
    return (0,1) if score<0 else (score/2,score/2)


def score_difference(before, after):
    old_low,old_high = score_bounds(before)
    new_low,new_high = score_bounds(after)
    return new_low-old_high,new_high-old_low


def contrast(differences):
    low,high = zip(*differences)
    rng = Random(311)
    samples = [rng.choices(range(len(low)),k=len(low)) for _ in range(2000)]
    bottom = sorted(mean(low[i] for i in sample) for sample in samples)
    top = sorted(mean(high[i] for i in sample) for sample in samples)
    return dict(bounds=[mean(low),mean(high)],interval=[bottom[49],top[1949]])


def summarize(data):
    groups = {cell:[] for cell in CELLS}
    for r in data["games"]:
        groups[tuple(r[2:5])].append(r)
    totals, differences, interactions = {}, {}, {}
    for (shape,own,foe),rows in groups.items():
        key = show_cell((shape,own,foe))
        totals[key] = dict(outcomes=dict(Counter(str(r[5]) for r in rows)),flag_taken=sum(r[7]>0 for r in rows),
                           out_other=sum(r[8]>0 and r[7]==0 for r in rows),mine_cleared=sum(r[9]>0 for r in rows),
                           win100=sum(r[5]==2 and r[6]<=100 for r in rows),win140=sum(r[5]==2 and r[6]<=140 for r in rows),
                           loss140=sum(r[5]==0 and r[6]<=140 for r in rows),mean_recorded_plies=mean(r[6] for r in rows))
        if shape:
            closed = groups[(0,own,foe)]
            diff = [score_difference(x[5],y[5]) for x,y in zip(closed,rows,strict=True)]
            differences[key] = contrast(diff)
            if own==2:
                legacy = groups[(shape,0,foe)]
                legacy_closed = groups[(0,0,foe)]
                legacy_diff = [score_difference(x[5],y[5]) for x,y in zip(legacy_closed,legacy,strict=True)]
                interactions[key] = contrast([(lo-old_hi,hi-old_lo) for (lo,hi),(old_lo,old_hi) in zip(diff,legacy_diff,strict=True)])
    return dict(totals=totals,versus_closed=differences,route_vs_legacy_interaction=interactions)


def main(audit):
    directory = ROOT/"build/full-formation"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        build(directory)
        controls = rules_controls(directory)
        write_receipt(directory/"controls.json",controls)
        print(json.dumps(controls))
    else:
        controls = json.loads((directory/"controls.json").read_bytes())
        require(rules_controls(directory)==controls,"rule and layout results did not reproduce")
        cli = cli_pair(directory)
        development = collect(directory,SEEDS[0])
        dev = summarize(development)
        # Lock one signed comparison before evaluating held-out seeds.
        ranked = [(key,sign) for key in dev["versus_closed"] for sign in (1,-1)]
        def lower(section,item):
            key,sign = item
            lo,hi = section["versus_closed"][key]["interval"]
            return lo if sign==1 else -hi
        selected = max(ranked,key=lambda item:lower(dev,item))
        write_receipt(directory/"development.json",dict(data=development,summary=dev,selected=selected))
        confirmation = collect(directory,SEEDS[1])
        confirm = summarize(confirmation)
        require(collect(directory,SEEDS[0])==development,"development arms did not reproduce")
        supported = lower(dev,selected)>0 and lower(confirm,selected)>0
        write_receipt(directory/"confirmation.json",dict(data=confirmation,summary=confirm,selected=selected,
                                                         supported=supported,cli=cli))
        print(json.dumps(dict(selected=selected,development=dev["versus_closed"][selected[0]],
                              confirmation=confirm["versus_closed"][selected[0]],cli=cli)))
        print("JUNQI_RESEARCH "+("supported" if supported else "inconclusive"))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected audit or no argument")
    main(bool(sys.argv[1:]))
