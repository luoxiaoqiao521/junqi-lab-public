"""Verify full-information route switches on mine diagrams and known replay cuts."""
import hashlib
from itertools import combinations, groupby, product
import json
import os
from pathlib import Path
import subprocess
import sys

from PIL import Image

from measure_deadline_opening import ROOT, RULES, build, require
from serial_archive_pressure import archives
from serial_formation_matrix import contrast, score_difference
from serial_worker import write_receipt

IMAGES = Path(os.environ.get("JUNQI_LOCAL_COLLECTION_ROOT", ROOT / "data/local-collection")) / "\u519b\u68cb\u96f7\u578b"
PRESSURE = ROOT/"build/archive-pressure/audit-full-mode-route-replies-68b01ca4aae2333e2297"
SLOTS = {25:1,20:2,21:3,22:4,23:5,24:6,29:7,27:8,28:9}


def piece_icons(path):
    # The supplied diagrams share this raster grid and the same colored piece sprites.
    with Image.open(path) as image:
        raster = image.convert("RGB")
    pixels = raster.load()
    hits = {}
    for y in range(raster.height//2,raster.height):
        orange = [x for x in range(raster.width) if orange_pixel(*pixels[x,y])]
        for _,group in groupby(enumerate(orange),lambda item:item[1]-item[0]):
            span = [x for _,x in group]
            if 33<=len(span)<=38:
                hits.setdefault(span[0],[]).append(y)
    columns = []
    for x in sorted(hits,key=lambda x:(-len(hits[x]),x)):
        if len(hits[x])>=2 and all(abs(x-other)>3 for other in columns):
            columns.append(x)
    bottom = max(y for x in columns for y in hits[x])
    icons = []
    for x in columns:
        for row,end in ((4,bottom-40),(5,bottom)):
            if sum(end-23<=y<=end for y in hits[x])>=3:
                ink = tuple(min(pixels[a,b])>140 and min(pixels[a,b][:2])>190
                            for b in range(end-20,end-3) for a in range(x+4,x+32))
                icons.append((row,x,ink))
    require(len(icons)==4,f"ambiguous piece locations in {path.name}")
    return icons


def orange_pixel(red,green,blue):
    return red>150 and 35<green<190 and blue<85 and red-green>35


def panels():
    stamp = next(ink for row,x,ink in piece_icons(IMAGES/"1-123.png") if (row,x)==(5,50))
    rows = []
    for index,path in enumerate(sorted(IMAGES.glob("*.png"),key=lambda p:int(p.stem.split("-")[0])),1):
        boxes = piece_icons(path)
        ranks = sorted((sum(a!=b for a,b in zip(ink,stamp,strict=True)),row,x) for row,x,ink in boxes)
        distance,row,flag = ranks[0]
        require(distance<ranks[1][0] and row==5 and all((x-flag)%40==0 for _,x,_ in boxes),
                f"uncertain flag recognition in {path.name}")
        mines = sorted(r*5+1+(x-flag)//40 for r,x,_ in boxes if (r,x)!=(row,flag))
        label = "".join(str(k) for k in sorted(SLOTS[n] for n in mines) if k<9)
        rows.append(dict(id=index,name=path.name,mines=mines,label_matches=label==path.stem.split("-")[1],
                         sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    require({tuple(r["mines"]) for r in rows}==set(combinations(sorted(SLOTS),3)) and len(rows)==84,
            "mine diagrams do not cover exactly the 84 distinct combinations")
    return rows


def native(directory, mode, wire):
    run = subprocess.run([directory/"deadline_opening.exe",RULES,mode],input=wire,encoding="utf8",capture_output=True,timeout=70)
    require(run.returncode==0,f"{mode} failed: {run.stderr.strip()}")
    return [line.split() for line in run.stdout.splitlines()]


def compare_panels(directory):
    catalog = panels()
    wire = "".join(" ".join(map(str,[r["id"],*r["mines"]]))+"\n" for r in catalog)
    raw = native(directory,"route-panels",wire)
    controls = [list(map(int,row[1:])) for row in raw if row[0]=="C"]
    proofs = [list(map(int,row[1:])) for row in raw if row[0]=="M"]
    require(len(controls)==4 and len(proofs)==168 and
            [p[:2] for p in proofs]==[[r["id"],g] for r in catalog for g in (0,1)],"incomplete mine route results")
    return dict(catalog=catalog,controls=controls,proofs=proofs)


def replay_cuts(audit):
    receipt = json.loads((PRESSURE/("confirmation.json" if audit else "development.json")).read_bytes())
    cuts = [r for r in receipt["result"]["roots"] if r["alternative"]>=0 and (audit or r["chosen_team_upper"]==0)]
    require(len(cuts)==(9 if audit else 3) and len({r["key"] for r in cuts})==len(cuts),"mismatched registered roots")
    return cuts


def compare_replays(directory, audit):
    assigned,wires = archives()
    cuts = replay_cuts(audit)
    payload = []
    for cut in cuts:
        key = cut["key"]
        tag,_,ccw,cap,tape = wires[key].split(maxsplit=4)
        seed = int(key[:12],16) + (846100 if audit else 846000)
        require(assigned[key]["audit"]==audit,"layout families crossed between development and confirmation")
        payload.append(f'{tag} {key} {ccw} {cap} {cut["ply"]} {cut["chosen"]} {cut["alternative"]} {seed} {tape}')
    rows = []
    for tag,key,*fields in native(directory,"route-replays","".join(payload)):
        require(tag=="G","unexpected route replay output")
        rows.append([key,*map(int,fields)])
    expected = []
    for cut in cuts:
        actions = (cut["chosen"],cut["alternative"],cut["human"])
        expected.extend([cut["key"],cut["ply"],arm,table,stream,actions[arm]]
                        for arm,table,stream in product(range(3),range(2),range(8)))
    require([r[:6] for r in rows]==expected and all(r[6] in (-1,0,1,2) and r[7]>=r[1] for r in rows),
            "missing paired route replay or inconsistent archived action")
    return dict(cuts=cuts,rows=rows)


def summarize(data):
    report = {}
    for table in (0,1):
        matches = {row[0]:{} for row in data["rows"]}
        totals = {arm:dict(win=0,draw=0,loss=0,unknown=0,win140=0,loss140=0,flag=0) for arm in range(3)}
        for key,cut,arm,profile,stream,move,score,ply,flag,_ in data["rows"]:
            if profile!=table:
                continue
            matches[key][arm,stream] = score
            totals[arm][{2:"win",1:"draw",0:"loss",-1:"unknown"}[score]] += 1
            totals[arm]["win140"] += score==2 and ply<=140
            totals[arm]["loss140"] += score==0 and ply<=140
            totals[arm]["flag"] += flag>0
        effects = {}
        for arm in (1,2):
            per_root = []
            for scores in matches.values():
                bound = [score_difference(scores[0,s],scores[arm,s]) for s in range(8)]
                per_root.append(tuple(sum(values)/8 for values in zip(*bound)))
            effects[arm] = contrast(per_root)
        report[table] = dict(totals=totals,effects=effects)
    return report


def main(audit):
    directory = ROOT/"build/full-route-choice"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        build(directory)
        panel = compare_panels(directory)
        data = compare_replays(directory,False)
        write_receipt(directory/"development.json",dict(panels=panel,data=data,summary=summarize(data)))
        print(json.dumps(dict(panels=len(panel["catalog"]),roots=len(data["cuts"]),responses=len(data["rows"]))))
    else:
        dev = json.loads((directory/"development.json").read_bytes())
        require(compare_panels(directory)==dev["panels"] and compare_replays(directory,False)==dev["data"],
                "development evidence did not reproduce")
        data = compare_replays(directory,True)
        report = summarize(data)
        supported = all(dev["summary"][str(t)]["effects"]["1"]["interval"][0]>0 and
                        report[t]["effects"][1]["interval"][0]>0 for t in (0,1))
        write_receipt(directory/"confirmation.json",dict(data=data,summary=report,supported=supported))
        print(json.dumps(report))
        print("JUNQI_RESEARCH "+("supported" if supported else "inconclusive"))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected audit or no argument")
    main(bool(sys.argv[1:]))
