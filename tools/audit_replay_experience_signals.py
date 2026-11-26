"""Report archive rates for opening-camp and non-movement clues."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random
import subprocess
import sys

from audit_qq_archive import ROOT
from compare_qq_rules import VC

OLD = ROOT/"build/replay-intake/intake-new-replay-families-729f8aa0fa7e1069e0e6"
OUT = ROOT/"build/experience-evidence"
RULES = ROOT/"config/ruleset-selfplay.json"
RUN = ROOT/"build/replay-intake/experience-census-20261122"


def read_catalog(folder):
    return json.loads((folder/"probe.json").read_bytes())["rows"]


def select(previous, fresh):
    old = {r["family"] for r in previous}
    admissible = [r for r in fresh if r["format"]==".jgs" and not r["stopped"] and not r["prior_layout"]
                  and r["result"] in (1,2,3)]
    groups = {}
    for r in sorted(admissible,key=lambda r:(r["family"],r["key"])):
        groups.setdefault(r["family"],r)
    return {r["key"]: r|{"cohort":"recent" if family not in old else "retained"}
            for family,r in groups.items()}


def phrases(data):
    cases = defaultdict(dict)
    for line in data.splitlines():
        tag,key,seat,a,b,c,*tail = line.split()
        numbers = list(map(int,(seat,a,b,c,*tail)))
        identity = tuple(numbers[:2]) if tag=="N" else (numbers[0],)
        if tag not in ("N","C") or len(numbers)!=(4 if tag=="N" else 7) or identity in cases[key,tag]:
            raise ValueError("invalid or duplicated clue row")
        cases[key,tag][identity] = numbers[1:]
    return cases


def corner_group(opportunities):
    return "no_exit" if opportunities==0 else "one_two" if opportunities<=2 else "three_plus"


def features(camps,corners):
    tally = Counter()
    for row,wing,rank,rail,_,_ in camps.values():
        group = "upper" if row==1 else "lower"
        tally[group,"total"]+=1
        tally[group,"bomb"]+=rank==2
        tally[group,"rail_origin"]+=rail
        if wing!=2:
            tally[group,"wing_total"]+=1
            tally[group,"marshal_same"]+=wing==0
    for side,opportunities,rank in corners.values():
        group = corner_group(opportunities)
        tally[group,"total"]+=1
        tally[group,"mine"]+=rank==1
        tally[group,"big"]+=rank in (9,10,11)
        tally[group,"bomb"]+=rank==2
    return tally


def describe(blocks):
    total = sum(blocks,Counter())
    output = {}
    for group,fields in (("upper",("bomb","marshal_same","rail_origin")),("lower",("bomb","marshal_same","rail_origin")),
                         ("no_exit",("mine","big","bomb")),("one_two",("mine","big","bomb")),
                         ("three_plus",("mine","big","bomb"))):
        for field in fields:
            n = "wing_total" if field=="marshal_same" else "total"
            output[f"{group}_{field}"] = [total[group,field],total[group,n]]
    rng = random.Random(28512)
    gaps = [("lower_bomb_minus_upper_bomb","lower","upper","bomb"),
            ("three_plus_mine_minus_one_two_mine","three_plus","one_two","mine")]
    paired = {}
    for title,a,b,k in gaps:
        def difference(x):
            return x[0]/x[1]-x[2]/x[3] if x[1] and x[3] else None
        packed = [(c[a,k],c[a,"total"],c[b,k],c[b,"total"]) for c in blocks]
        actual = difference(tuple(map(sum,zip(*packed,strict=True)))) if packed else None
        sims = []
        for _ in range(1000 if packed else 0):
            sample = rng.choices(packed,k=len(packed))
            effect = difference(tuple(map(sum,zip(*sample,strict=True))))
            if effect is not None:
                sims.append(effect)
        sims.sort()
        paired[title] = dict(diff=actual,interval_95=[sims[len(sims)*25//1000],sims[len(sims)*975//1000]] if sims else None)
    return dict(families=len(blocks),counts=output,contrasts=paired)



def calibration(cases, roster):
    pools = defaultdict(lambda: ([],[]))
    for key,r in roster.items():
        if r["cohort"]!="recent":
            continue
        # No neighboring positions from the same family enter training and validation.
        fold = int(sha256(("experience-v1:"+r["family"]).encode()).hexdigest()[:8],16)%5
        part = int(fold in (3,4))
        for landing,_,rank,_,row,col in cases[key,"C"].values():
            pools["camp_bomb"][part].append((key,(row,abs(col-2)),landing,int(rank==2)))
        for side,opportunities,rank in cases[key,"N"].values():
            pools["corner_mine"][part].append((key,(side,),corner_group(opportunities),int(rank==1)))
    results = {}
    for name,(train,test) in pools.items():
        base,extra = Counter(),Counter()
        for _,position,clue,label in train:
            base[position,label]+=1
            extra[position,clue,label]+=1
        prior = (sum(label for *_,label in train)+1)/(len(train)+2)
        losses = defaultdict(lambda: [0.,0.,0])
        for key,position,clue,label in test:
            plain = (base[position,1]+12*prior)/(base[position,0]+base[position,1]+12)
            update = (extra[position,clue,1]+12*plain)/(extra[position,clue,0]+extra[position,clue,1]+12)
            row=losses[key]
            for arm,pred in enumerate((plain,update)):
                row[arm]+=(pred-label)**2
            row[2]+=1
        if losses:
            blocks = list(losses.values())
            a,b,n = map(sum,zip(*blocks))
            rng = random.Random(38196)
            sampled = []
            for _ in range(1000):
                x,y,m = map(sum,zip(*rng.choices(blocks,k=len(blocks))))
                sampled.append((y-x)/m)
            sampled.sort()
            results[name]=dict(train=len(train),test=n,test_families=len(blocks),brier=[a/n,b/n],
                               delta_95=[sampled[25],sampled[975]])
    return results

def run(compile=True):
    roster=select(read_catalog(OLD),read_catalog(RUN))
    tapes={}
    for line in (RUN/"games.txt").read_text(encoding="utf8").splitlines():
        tag,key,tape=line.split(maxsplit=2)
        if tag=="G" and key in roster:
            tapes[key]=tape
    if set(tapes)!=set(roster):
        raise ValueError("missing selected archive tape")
    wire="".join(f'{k} {r["direction"]} {tapes[k]}\n' for k,r in sorted(roster.items()))
    if compile:
        src=" ".join(f'"{ROOT.as_posix()}/{f}"' for f in ("core/board.cpp","core/layout.cpp","core/rules.cpp",
                           "core/state.cpp","tests/replay_experience_signals.cpp"))
        command=f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /W4 /WX /O2 /MT /utf-8 /I"{ROOT.as_posix()}/core" {src} /Fe:replay_experience_signals.exe'
        subprocess.run(f'cmd.exe /d /s /c "{command}"',cwd=OUT,check=True,timeout=90)
    native=subprocess.run([OUT/"replay_experience_signals.exe",RULES],input=wire,text=True,encoding="utf8",
                          capture_output=True,check=True,timeout=95).stdout
    parsed=phrases(native)
    if any(k not in roster for k,_ in parsed):
        raise ValueError("unexpected source row")
    by=defaultdict(list)
    for key,row in roster.items():
        raw=features(parsed[key,"C"],parsed[key,"N"])
        by["all"].append(raw)
        by[row["cohort"]].append(raw)
    result={"archive_wire_sha":sha256(wire.encode("utf8")).hexdigest(),
            "raw_digest":sha256(native.encode("utf8")).hexdigest(),
            "groups":{name:describe(items) for name,items in by.items()},
            "heldout_calibration":calibration(parsed,roster)}
    report=OUT/"replay-signals.json"
    if compile:
        report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    elif json.loads(report.read_bytes())!=result:
        raise ValueError("archive signal audit changed")
    print(json.dumps(result))


if __name__=="__main__":
    if sys.argv[1:] not in ([],["audit"]):
        raise ValueError("usage: audit_replay_experience_signals.py [audit]")
    run(not sys.argv[1:])
