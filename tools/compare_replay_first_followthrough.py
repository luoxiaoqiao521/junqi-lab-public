"""Compare a changed first action and a changed first decision after the reply."""
from collections import Counter, defaultdict
import json
import os
from statistics import mean
import subprocess
import sys

from audit_replay_curriculum_roots import CONFIG, ROOT, RULES, compile_native, move_counts, parse, prepare, require
from compare_archive_cycle_exit_values import interval


PROBE = ROOT/"build/replay-curriculum/audit-replay-curriculum-root-contract-5c48c6679f379d39697c/summary.json"
FIELDS = ("first", "reply_seat", "reply", "returned", "normal", "second", "before_return", "trace",
          "plies", "outcome", "ending", "contact", "enemy_loss", "enemy_exit", "friendly_exit")
ARMS = ((0,0),(1,0),(0,1),(1,1))
BASE, FIRST, NEXT, BOTH = ARMS


def cells(output, feed, tables=(0,1)):
    launch, follow = [],{}
    for line in output.splitlines():
        if not line.startswith("L "):
            launch.append(line)
            continue
        _,family,*payload = line.split()
        ply,table,rep,first,second,*values = map(int,payload)
        key = (family,ply,table,rep)
        arm = (first,second)
        arms = follow.setdefault(key,{})
        require(arm in ARMS and arm not in arms,"unexpected or duplicate arm")
        arms[arm] = dict(zip(FIELDS,values,strict=True))
    _,roots,branches,excluded = parse("\n".join(launch),move_counts(feed))
    saved = [r for r in json.loads(PROBE.read_bytes())["records"] if r["status"]=="branchable"]
    require(not excluded and roots=={(r["family"],r["ply"]):r["root"] for r in saved},
            "recovered source frame changed")
    baseline = {(r["family"],r["ply"],b["table"],b["arm"]):
                {k:v for k,v in b.items() if k not in ("table","arm")} for r in saved for b in r["branches"]}
    require(branches==baseline,"opening cycle differs from verified recovery")
    require(follow.keys()=={key+(table,rep) for key in roots for table in tables for rep in (0,1)},
            "missing factorial cell")
    for key,arms in follow.items():
        check(arms,roots[key[:2]]["recorded_action"])
    return follow


def check(arms, human):
    require(arms.keys()==set(ARMS),"missing arm")
    for no,yes in ((BASE,NEXT),(FIRST,BOTH)):
        left,right = arms[no],arms[yes]
        before = ("first","reply_seat","reply","returned","normal","before_return")
        require(all(left[k]==right[k] for k in before),"divergence before continuation intervention")
        require(no==BASE or left["first"]==human,"recorded intervention differs")
        if left["second"]==right["second"]:
            require(left==right,"same second action produced different trajectories")
    require(all(1<=row["plies"]<=960 and row["outcome"] in (-1,0,1,2) for row in arms.values()),
            "invalid natural or cutoff result")
    if arms[BASE]["first"]==human:
        require(arms[BASE]==arms[FIRST] and arms[NEXT]==arms[BOTH],"same first action diverged")


def outcome(row):
    score = row["outcome"]/2
    return (0,1) if score<0 else (score,score)


def contrasts(rows, signs):
    units = defaultdict(list)
    for key,arms in rows:
        lo,hi = 0.,0.
        for arm, sign in signs.items():
            a,b = outcome(arms[arm])
            lo += a if sign>0 else -b
            hi += b if sign>0 else -a
        units[key[0]].append((lo,hi))
    lo,hi = ([mean(pair[bound] for pair in group) for group in units.values()] for bound in (0,1))
    return {"bounds":[mean(lo),mean(hi)],"layout_interval":[interval(lo)[0],interval(hi)[1]]}


def report(follow, tables=(0,1)):
    panels = {}
    for table in tables:
        rows = [(k,a) for k,a in follow.items() if k[2]==table]
        stats = {}
        for arm in ARMS:
            choices = [a[arm] for _,a in rows]
            stats[str(arm)] = dict(outcomes=dict(Counter(str(v["outcome"]) for v in choices)),
                next_opportunity=sum(v["returned"]>0 for v in choices),
                next_change=sum(v["normal"]!=v["second"] for v in choices),
                first_change=sum(a[BASE]["first"]!=a[arm]["first"] for _,a in rows),
                win_within_80=sum(v["outcome"]==2 and v["plies"]<=80 for v in choices),
                loss_within_80=sum(v["outcome"]==0 and v["plies"]<=80 for v in choices))
        stats["first_only"]=contrasts(rows,{FIRST:1,BASE:-1})
        stats["return_only"]=contrasts(rows,{NEXT:1,BASE:-1})
        stats["return_after_original"]=contrasts(rows,{BOTH:1,FIRST:-1})
        stats["interaction"]=contrasts(rows,{BOTH:1,BASE:1,NEXT:-1,FIRST:-1})
        panels[str(table)]=stats
    return panels


def evidence(directory, mode, tables):
    feed,provenance=prepare()
    process=subprocess.run([directory/"takeover.exe",CONFIG,RULES,"960",mode],
                           input=feed,encoding="utf8",capture_output=True,timeout=170)
    require(process.returncode==0,f"native factorial probe failed: {process.stderr.strip()}")
    return cells(process.stdout,feed,tables),provenance


def games(results):
    return [[list(k),[arms[a] for a in ARMS]] for k,arms in results.items()]


def write_result(directory, data, audit):
    wire=(json.dumps(data,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n").encode()
    path=directory/"summary.json"
    if audit:
        require(path.read_bytes()==wire,"factorial rollouts did not reproduce")
    else:
        pending=path.with_suffix(".tmp")
        pending.write_bytes(wire)
        pending.replace(path)


def main(audit):
    directory=ROOT/"build/replay-curriculum"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        compile_native(directory)
    results,provenance=evidence(directory,"curriculum-followthrough",(0,1))
    panels=report(results)
    write_result(directory,dict(provenance=provenance,panels=panels,games=games(results)),audit)
    print(json.dumps(panels,sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(sys.argv[1:]==["audit"])
