"""Separate the effects of clearing a gate and having the ally take it."""
from collections import Counter, defaultdict
import json
import shutil
import sys
import time

from early_route import EXE, ROOT, RULES, collect, model_roster, score_delta, sha, source_keys, scores_summary

TICKET = "test-partner-route-followthrough"
RUN = ROOT/"build"/(TICKET + "-v1")
REPORT = ROOT/"docs"/"REPORTS"/(TICKET + ".json")
GRID = source_keys(304000,48)
ARMS = ("original", "opens", "original_accepts", "opens_accepts")
DIFFERENCES = {"single_opening":(1,0), "accept_after_opening":(3,1),
               "opening_with_acceptance":(3,2), "joint":(3,0)}


def validate_forks(forks, same_first):
    if len(forks)!=len(ARMS):
        raise ValueError("missing coordination arm")
    for score,turn,available,taken,enemy in forks:
        if not (score in (-1,0,1,2) and 0<=taken<=available<=turn<=1 and enemy in (0,1)):
            raise ValueError("wrong route outcome")
    for plain,accepts in zip(forks[:2],forks[2:],strict=True):
        if plain[1:3]!=accepts[1:3] or plain[4]!=accepts[4] or accepts[2]!=accepts[3]:
            raise ValueError("partner treatment changed a preceding action or failed to take a legal gate")
        if (not plain[2] or plain[3]) and plain!=accepts:
            raise ValueError("paired path changed even without a new partner action")
    if same_first and (forks[0]!=forks[1] or forks[2]!=forks[3]):
        raise ValueError("identical first moves did not reproduce")


def update(forks, seed, counts, pairs):
    counts["matched"] += 1
    for name, (score,turn,available,taken,enemy) in zip(ARMS,forks,strict=True):
        counts.update({f"{name}/possible":available,f"{name}/taken":taken,
                       f"{name}/score/{score}":1})
    for name,(a,b) in DIFFERENCES.items():
        bounds = (0,0) if a==3 and b==1 and forks[a]==forks[b] and (
            not forks[b][2] or forks[b][3]) else score_delta(forks[a][0],forks[b][0])
        pairs[name][seed].append(bounds)
        counts[f"{name}/gains"] += int(bounds[0]>0)
        counts[f"{name}/losses"] += int(bounds[1]<0)
    ax,bx,cx,dx = (f[0] for f in forks)
    lo1,hi1 = score_delta(dx,cx)
    lo0,hi0 = score_delta(bx,ax)
    pairs["interaction"][seed].append((lo1-hi0,hi1-lo0))


def summarize(rows):
    if [(r["seed"],r["seat"],r["variant"]) for r in rows]!=GRID:
        raise ValueError("wrong independent source grid")
    coverage = {str(v):Counter() for v in (0,2)}
    counts = defaultdict(Counter)
    team = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        base = coverage[str(row["variant"])]
        base["games"] += 1
        w = row["window"]
        if not w:
            continue
        base["windows"] += 1
        base["compatible"] += int(w["worlds"]==2)
        if (w["decision"] not in (4,8,12,16) or w["moves"][1][0]!=w["gate"]
                or len(w["moves"])!=2 or w["worlds"] not in (1,2) or len(w["cells"])!=2*w["worlds"]
                or any(len(cell)!=10 for cell in w["cells"])):
            raise ValueError("incorrect route window or simulation budget")
        for i,cell in enumerate(w["cells"]):
            key = f"{row['variant']}/{i//2}/{i%2}"
            for forks in cell:
                validate_forks(forks,w["moves"][0]==w["moves"][1])
                update(forks,row["seed"],counts[key],team[key])
    return dict(coverage={k:dict(v) for k,v in coverage.items()},
                panels={k:dict(c) | {"team":{d:scores_summary(v) for d,v in team[k].items()}}
                        for k,c in counts.items()})


def frozen(parent):
    return dict(engine=sha(RUN/"junqi.exe"), rules=sha(RUN/"ruleset.json"), parent=parent)


def replay(parent,key):
    return collect(RUN/"junqi.exe",RUN/"ruleset.json",parent,*key,mode="partner-route-followthrough")


def run():
    if RUN.exists() or REPORT.exists():
        raise FileExistsError("partner route test exists; use audit")
    parent = model_roster()["candidate"]
    RUN.mkdir()
    shutil.copyfile(EXE,RUN/"junqi.exe")
    shutil.copyfile(RULES,RUN/"ruleset.json")
    started = time.monotonic()
    rows = [replay(parent,key) for key in GRID]
    data = dict(frozen=frozen(parent),rows=rows,summary=summarize(rows),seconds=time.monotonic()-started)
    payload = (json.dumps(data,separators=(",",":"),allow_nan=False)+"\n").encode()
    if len(payload)>1024**2:
        raise ValueError("coordination report exceeded byte cap")
    with REPORT.open("xb") as output:
        output.write(payload)
    print(f"Measured four paired routes over {len(rows)} fixed source games")


def audit():
    data = json.loads(REPORT.read_text(encoding="utf8"))
    parent = model_roster()["candidate"]
    rows = data["rows"]
    if data["frozen"]!=frozen(parent) or data["summary"]!=summarize(rows):
        raise ValueError("factorial test inputs or results changed")
    for variant in (0,2):
        selected = next((r for r in rows if r["variant"]==variant and r["window"]),rows[variant//2])
        if replay(parent,tuple(selected[k] for k in ("seed","seat","variant")))!=selected:
            raise ValueError("fixed partner replays were not identical")
    print("PASS matched partner turns, actual acceptances, score interaction and replay")


if __name__=="__main__":
    if not sys.argv[1:]:
        run()
    elif sys.argv[1:]==["audit"]:
        audit()
    else:
        raise SystemExit("usage: partner_route_followthrough.py [audit]")
