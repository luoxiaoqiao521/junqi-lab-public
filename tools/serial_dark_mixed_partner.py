"""Train on matched ordinary games with one change in the partner distribution."""
from collections import Counter, defaultdict
from itertools import product
import json
import os
import shutil
import sys

from compare_book_learning import fit
from full_mode_runtime import FullModeRun, sha
from serial_dark_historical_peers import PEER, STUDENTS
from serial_dark_ordinary_learning import RATE, RULES, UPDATES, assert_layout_separation, examples_xy, model_wire, run_native, summarize
from serial_dynamic_reply import ROOT, build, require
from serial_formation_matrix import contrast, score_difference
from serial_worker import read, write_receipt


NAMES=("original","mixed")
INPUTS=("parent.json","peer.json","grid.json","engine/deadline_opening.exe","engine/rules.json")
MIN_PRACTICAL=.03


def prepare(scope):
    folder=scope.folder
    scope.engine.mkdir(parents=True,exist_ok=True)
    build(scope.engine)
    shutil.copyfile(RULES,scope.rules)
    shutil.copyfile(STUDENTS/"parent.json",folder/"parent.json")
    shutil.copyfile(PEER,folder/"peer.json")
    grid=dict(train=[(24130000+i,i%4,(i//4)%2) for i in range(96)])
    for stage,base,count in (("development",24131000,16),("confirmation",24132000,48)):
        grid[stage]=[(base+i,i%4,t) for i,t in product(range(count),range(6))]
    assert_layout_separation(grid)
    write_receipt(folder/"grid.json",grid)
    write_receipt(folder/"fingerprints.json",{p:sha(folder/p) for p in INPUTS})


def training_pairs(games,examples):
    groups=[defaultdict(list),defaultdict(list)]
    for output,rows in zip(groups,examples,strict=True):
        for row in rows:
            output[row["seed"]].append(row)
    course=[[],[]]
    unlabelled=0
    for a,b in zip(*games,strict=True):
        key=a["seed"]
        require(key==b["seed"] and a["arm"]==b["arm"],"training used different baseline decisions")
        if b["table"]==0:
            strip={k:v for k,v in b.items() if not k.startswith("peer_")}
            require(a==strip and groups[0][key]==groups[1][key],"unmodified training game changed")
        if min(a["result"],b["result"])<0:
            unlabelled+=1
            continue
        count=min(len(group[key]) for group in groups)
        for data,group in zip(course,groups,strict=True):
            data.extend(group[key][:count])
    return course,unlabelled


def learn(scope,grid,models):
    folder=scope.folder
    parent,peer=(models[n] for n in ("parent","peer"))
    standard=[(s,p,0) for s,p,t in grid]
    a,x,_,base=run_native(scope,"D",standard,(parent,))
    b,y,_,mixed=run_native(scope,"M",grid,(parent,),peer=peer)
    compare=[r for r in b if r["table"]]
    require(len(compare)==48 and sum(r["peer_changes"] for r in compare)>0,"mixed training did not use the partner")
    course,unlabelled=training_pairs((a,b),(x,y))
    require(len(course[0])==len(course[1])>0,"training sample budget differs")
    for name,data,trace in zip(NAMES,course,(base,mixed),strict=True):
        rows,labels=examples_xy(data,parent)
        learned=fit(rows,labels,48120,24,parent["models"]["collision"],updates=UPDATES,rate=RATE)
        model=dict(parent,models={"collision":learned},parent_sha256=sha(folder/"parent.json"),
                   training=dict(rows=len(rows),by_result=dict(Counter(r["outcome"] for r in data)),
                                 updates=UPDATES,rate=RATE,unlabelled_layouts=unlabelled,**trace))
        require(model_wire(model)!=model_wire(parent),"no weight update in mixed partner comparison")
        write_receipt(folder/(name+".json"),model)
    write_receipt(folder/"experience.json",dict(games=[a,b],examples=course,changed_partner=len(compare),
                  partner_turns=sum(r["peer_turns"] for r in compare),partner_overrides=sum(r["peer_changes"] for r in compare)))
    write_receipt(folder/"trained.json",{n:sha(folder/n) for n in ("experience.json",*(a+".json" for a in NAMES))})


def evaluate(scope,grid,models):
    folder=scope.folder
    original,mixed=(read(folder/(name+".json")) for name in NAMES)
    peer=models["peer"]
    a,_,_,base_trace=run_native(scope,"X",grid,(models["parent"],original),peer=peer)
    b,_,first,mix_trace=run_native(scope,"X",grid,(original,mixed),peer=peer)
    require(a[1::2]==[dict(r,arm=3) for r in b[::2]],"original student failed to reproduce as the A/B control")
    expected={t:sum(r["peer_changes"] for r in b if r["table"]==t) for t in range(6)}
    require(all((expected[t]>0)==(t in (1,3,4)) for t in expected),"evaluation assigned the historical model incorrectly")
    panel=summarize(b,first)
    by_seed=defaultdict(dict)
    for old,new in zip(b[::2],b[1::2],strict=True):
        by_seed[old["seed"]][old["table"]]=(0,0) if old["hash"]==new["hash"] else score_difference(old["result"],new["result"])
    # Compare the mixed-training effect before and after changing the peer position.
    def interaction(v):
        lo=sum(v[t][0] for t in (0,4))-sum(v[t][1] for t in (2,3))
        hi=sum(v[t][1] for t in (0,4))-sum(v[t][0] for t in (2,3))
        return lo,hi
    return dict(original_vs_parent=summarize(a),mixed_vs_original=panel,
                environment_interaction=contrast([interaction(v) for v in by_seed.values()]),
                games=b,first=first,parent_games=a,original_runtime=base_trace,mixed_runtime=mix_trace)


def main(audit):
    scope=FullModeRun(ROOT/"build/dark-mixed-partner"/os.environ["JUNQI_SERIAL_KEY"])
    folder=scope.folder
    if not audit:
        prepare(scope)
    require(all(sha(folder/n)==v for n,v in read(folder/"fingerprints.json").items()),"mixed partner inputs changed")
    models={n:read(folder/(n+".json")) for n in ("parent","peer")}
    grid={k:[tuple(r) for r in v] for k,v in read(folder/"grid.json").items()}
    if not audit:
        learn(scope,grid["train"],models)
    require(all(sha(folder/n)==v for n,v in read(folder/"trained.json").items()),"mixed partner outputs changed")
    stage="confirmation" if audit else "development"
    report=evaluate(scope,grid[stage],models)
    write_receipt(folder/(stage+".json"),report)
    if audit:
        primary=report["mixed_vs_original"][2]
        verdict=("supported" if primary["interval"][0]>0 and primary["bounds"][0]>=MIN_PRACTICAL else
                 "refuted" if primary["interval"][1]<MIN_PRACTICAL else "inconclusive")
        print("JUNQI_RESEARCH "+verdict)
    print(json.dumps({k:report[k] for k in ("mixed_vs_original","environment_interaction")}))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected mixed partner stage")
    main(bool(sys.argv[1:]))
