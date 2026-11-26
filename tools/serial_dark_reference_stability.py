"""Test whether higher-budget reference directions reproduce across independent cells."""
from collections import Counter, defaultdict
from itertools import combinations, product
import json
import os
from statistics import mean
import sys

from full_mode_runtime import sha
from serial_dynamic_reply import ROOT, require
from serial_formation_matrix import contrast, score_difference
from serial_worker import read, write_receipt


SOURCE=ROOT/"build/dark-response-distribution/audit-four-dark-rollout-response-distribution-0de90b2b04085d5138ab"
SPLITS={
    "worlds":({4,5},{6,7},{60,61},{60,61}),
    "reply_streams":({4,5,6,7},{4,5,6,7},{60},{61}),
}


def build_panel(root):
    moves={int(k):v for k,v in root["moves"].items()}
    if root["worlds"]!=8:
        require(not root["R"],"incomplete root emitted reference outcomes")
        return {},None
    rows={(a,w,t,i):score for a,w,t,i,score,_ in root["R"]}
    require(rows.keys()==set(product(moves,range(4,8),(0,2,5),(60,61))) and
            len(rows)==len(root["R"]) and all(s in (-1,0,1,2) for s in rows.values()),"invalid fixed reference matrix")
    base=next(k for k,v in moves.items() if v==root["base"])
    return rows,base


def average_delta(panel,a,b,w,reps):
    cells=[score_difference(panel[a,z,t,i],panel[b,z,t,i]) for z,t,i in product(w,(0,2,5),reps)]
    return tuple(map(mean,zip(*cells,strict=True)))


def top(panel,moves,worlds,replies):
    # With an unfinished entry only report comparisons whose interval excludes zero.
    if any(v<0 for (a,w,_,i),v in panel.items() if w in worlds and i in replies):
        return set()
    values={a:sum(panel[a,w,t,i] for w,t,i in product(worlds,(0,2,5),replies)) for a in moves}
    best=max(values.values())
    return {a for a,v in values.items() if v==best}


def direction(bounds):
    low,high=bounds
    return 1 if low>1e-9 else -1 if high<-1e-9 else 0


def audit_root(panel,base,moves,split):
    w0,w1,r0,r1=split
    first=top(panel,moves,w0,r0)
    second=top(panel,moves,w1,r1)
    counts=Counter(top_resolved=int(bool(first and second)),top_overlap=int(bool(first&second)),
                   unique_both=int(len(first)==len(second)==1),
                   unique_same=int(len(first)==len(second)==1 and first==second))
    pairs=[]
    for a,b in combinations(sorted(moves),2):
        x,y=average_delta(panel,a,b,w0,r0),average_delta(panel,a,b,w1,r1)
        d,e=direction(x),direction(y)
        counts["pairs"]+=1
        counts["both_directional"]+=bool(d and e)
        counts["sign_flip"]+=d*e==-1
        if a==base or b==base:
            counts["base_both_directional"]+=bool(d and e)
            counts["base_flip"]+=d*e==-1
        pairs.append(dict(indices=[a,b],first=x,second=y))
    for row in pairs:
        x,y=(row[k] for k in ("first","second"))
        for old,new in ((x,y),(y,x)):
            d=direction(old)
            if not d or min(map(abs,old))<.125-1e-9:
                continue
            counts["sampled_margin"]+=1
            sign=d*direction(new)
            counts[{1:"same",0:"unclear",-1:"opposite"}[sign]]+=1
    return counts,pairs


def audit(receipt):
    counts=defaultdict(Counter)
    rates=defaultdict(list)
    rows=[]
    for seed,seat,table,root in receipt["roots"]:
        panel,base=build_panel(root)
        if not panel:
            continue
        moves={a for a,*_ in panel}
        split_detail={}
        for name,split in SPLITS.items():
            observed,pairs=audit_root(panel,base,moves,split)
            counts[name].update(observed)
            if observed["both_directional"]:
                rate=observed["sign_flip"]/observed["both_directional"]
                rates[name].append((rate,rate))
            split_detail[name]=dict(counts=observed,pairs=pairs)
        rows.append(dict(seed=seed,seat=seat,table=table,splits=split_detail))
    require(rows and rates,"reference repeatability was never exercised")
    rate={k:contrast(v) for k,v in rates.items()}
    return dict(source_roots=len(receipt["roots"]),evaluated=len(rows),counts=counts,
                layout_averaged_conditional_flip=rate),rows


def main(confirmation):
    folder=ROOT/"build/dark-reference-stability"/os.environ["JUNQI_SERIAL_KEY"]
    folder.mkdir(parents=True,exist_ok=True)
    files=("development.json","confirmation.json")
    if not confirmation:
        write_receipt(folder/"sources.json",{n:sha(SOURCE/n) for n in files})
    require(all(sha(SOURCE/n)==v for n,v in read(folder/"sources.json").items()),"reference changed across stages")
    phase="confirmation" if confirmation else "development"
    counts,rows=audit(read(SOURCE/(phase+".json")))
    write_receipt(folder/(phase+".json"),dict(summary=counts,roots=rows))
    if confirmation:
        unstable=counts["layout_averaged_conditional_flip"]["worlds"]["interval"][0]>.1
        print("JUNQI_RESEARCH "+("supported" if unstable else "inconclusive"))
    print(json.dumps(counts))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected repeatability stage")
    main(bool(sys.argv[1:]))
