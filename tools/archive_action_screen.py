"""Fit a small local ranker to recorded opening choices and test team transfer."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

os.environ["OPENBLAS_NUM_THREADS"]="1"
import numpy as np

from audit_qq_archive import INPUT, ROOT, decode, initial_board, local_records, public_records
from compare_qq_rules import VC, tape
from partner_crossplay import model_roster

OUT=ROOT/"build/qq-evidence"
RESULT=OUT/"archive-action-screen.json"
EXE=OUT/"replay_action_probe.exe"
WIDTH=9


def layout_group(layouts):
    mirrored=[b"".join(layout[i:i+5][::-1] for i in range(0,30,5))
              for layout in reversed(layouts)]
    forms=[b"".join(v[i:]+v[:i]) for v in (layouts,mirrored) for i in range(4)]
    return hashlib.sha256(min(forms)).digest()


def select(inputs):
    groups={}
    for raw in inputs:
        try:
            key,layouts,seats,events=decode(raw)
        except ValueError as error:
            if str(error)=="four_complete_layouts":
                continue
            raise
        group=layout_group(layouts)
        if group not in groups or key<groups[group][0]:
            groups[group]=(key,layouts,seats,events)
    return groups


def prefixes(inputs):
    holdout=select(local_records(inputs["local"]))
    public=select(public_records(inputs["corpus"]))
    pool=sorted(public.keys()-holdout.keys())
    picked=random.Random(31415).sample(pool,192)
    groupings=[(0,[public[k] for k in picked[:128]]),
               (1,[public[k] for k in picked[128:]]),(2,list(holdout.values()))]
    return [(split,record) for split,records in groupings for record in records], [len(records) for _,records in groupings]


def assemble():
    OUT.mkdir(parents=True,exist_ok=True)
    srcs=" ".join(f'"{ROOT.as_posix()}/{p}"' for p in (
        "core/board.cpp","core/rules.cpp","core/state.cpp","core/layout.cpp","tests/replay_action_probe.cpp"))
    task=f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /O2 /MT /utf-8 /I"{ROOT.as_posix()}/core" {srcs} /Fe:replay_action_probe.exe'
    subprocess.run(f'cmd.exe /d /s /c "{task}"',cwd=OUT,check=True,timeout=55)


def extract(positions):
    with tempfile.TemporaryFile(mode="w+t",encoding="ascii",newline="\n",dir=OUT) as feed:
        feed.write(f"{len(positions)}\n")
        for split,(_,layouts,seats,events) in positions:
            rows,_=tape(layouts,seats,events)
            prefix=[]
            steps=0
            for row in rows:
                if row[0]>=0: steps+=1
                if steps>40: break
                prefix.append(row)
            feed.write(f"{split} {len(prefix)}\n")
            for (x,y),(_,rank) in initial_board(layouts).items():
                feed.write(f"{x} {y} {rank}\n")
            for row in prefix:
                feed.write(" ".join(map(str,row))+"\n")
        feed.seek(0)
        data=subprocess.run([EXE],stdin=feed,capture_output=True,text=True,check=True,timeout=60).stdout
    groups=[[] for _ in range(3)]
    for line in data.splitlines():
        split,count,index,payload=line.split(" ",3)
        choices=np.fromstring(payload,sep=" ",dtype=np.float64).reshape(-1,WIDTH)
        index=int(index)
        if int(count)!=len(choices) or not 0<=index<len(choices):
            raise ValueError("action list contract")
        groups[int(split)].append((choices,index))
    return groups


class Dataset:
    def __init__(self,rows):
        sizes=np.array([len(r[0]) for r in rows],dtype=np.int64)
        self.offsets=np.cumsum(sizes)-sizes
        self.chosen=self.offsets+np.array([r[1] for r in rows])
        self.owner=np.repeat(np.arange(len(rows)),sizes)
        self.x=np.concatenate([row[0] for row in rows],axis=0)

    def loss(self,w,gradient=False):
        z=self.x@w
        top=np.maximum.reduceat(z,self.offsets)
        scaled=z-top[self.owner]
        mass=np.exp(scaled)
        normal=np.add.reduceat(mass,self.offsets)
        loss=float((np.log(normal)-scaled[self.chosen]).mean())
        if not gradient:
            ties=np.add.reduceat(scaled==0,self.offsets)
            top_hit=float(((scaled[self.chosen]==0)/ties).mean())
            return {"cross_entropy":loss,"top_probability":top_hit,
                    "decisions":int(len(self.chosen))}
        mass/=normal[self.owner]
        mass[self.chosen]-=1
        return self.x.T@mass/len(self.chosen)


def learn(train,width):
    w=np.zeros(WIDTH,dtype=np.float64)
    for _ in range(128):
        grad=train.loss(w,gradient=True)
        w[:width]-=.25*(grad[:width]+.001*w[:width])
    return w


def compare(simulation, blocks=256):
    output={}
    for context in (0,1):
        rows=[r for r in simulation if r[1]==context]
        if len(rows)!=blocks or len({r[0] for r in rows})!=blocks:
            raise ValueError("wrong paired evaluation coverage")
        output[str(context)]={}
        for index,name in ((3,"fit_existing"),(4,"fit_position")):
            wdl=dict(Counter(r[index] for r in rows))
            low=[((r[index] if r[index]>=0 else 0)-(r[2] if r[2]>=0 else 2))/2 for r in rows]
            high=[((r[index] if r[index]>=0 else 2)-(r[2] if r[2]>=0 else 0))/2 for r in rows]
            indexes=np.random.default_rng(313033).integers(0,len(rows),(2048,len(rows)))
            output[str(context)][name]=dict(
                delta_bounds=[float(np.mean(low)),float(np.mean(high))],
                interval=[float(np.quantile(np.asarray(low)[indexes].mean(axis=1),.025)),
                          float(np.quantile(np.asarray(high)[indexes].mean(axis=1),.975))],
                candidate=wdl,frozen=dict(Counter(r[2] for r in rows)))
    return output


def run():
    manifest={k:Path(v) for k,v in json.loads(INPUT.read_text(encoding="utf-8-sig")).items()}
    assemble()
    positions, counts=prefixes(manifest)
    datasets=list(map(Dataset,extract(positions)))
    frozen=np.array(model_roster()["candidate"]["weights"]+[0.,0.,0.,0.])
    fitted=[learn(datasets[0],width) for width in (5,9)]
    names=("frozen","fit_existing","fit_position")
    weights=dict(zip(names,[frozen,*fitted],strict=True))
    metrics={name:[d.loss(w) for d in datasets] for name,w in weights.items()}
    payload="\n".join(" ".join(map(str,weights[n])) for n in names)+"\n"
    raw=subprocess.run([EXE,"evaluate"],input=payload,capture_output=True,text=True,
                       check=True,timeout=90).stdout
    simulation=[list(map(int,line.split())) for line in raw.splitlines()]
    report=dict(group_counts=counts, weights={n:w.tolist() for n,w in weights.items()},
                choice=metrics, team=compare(simulation), simulation=simulation)
    temporary=RESULT.with_suffix(".tmp")
    temporary.write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    temporary.replace(RESULT)
    print(json.dumps({k:v for k,v in report.items() if k!="simulation"},sort_keys=True))


def audit():
    report=json.loads(RESULT.read_text(encoding="utf8"))
    if report["group_counts"][:2]!=[128,64] or report["group_counts"][2]<25:
        raise ValueError("expected layout group holdouts")
    if json.loads(json.dumps(compare(report["simulation"])))!=report["team"]:
        raise ValueError("unexpected team evaluation")
    if all(x==0 for x in report["weights"]["fit_position"]):
        raise ValueError("gradient never changed weights")
    print("PASS grouped replay action screen and frozen full-game comparisons")


if __name__=="__main__":
    audit() if sys.argv[1:]==["audit"] else run()
