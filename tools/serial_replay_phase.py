"""Matched replay phases, with two disjoint sets of hypothetical worlds."""
from collections import Counter
from hashlib import sha256
import json
from math import comb
import os
import sys

from measure_deadline_opening import ROOT, build, model_roster, require
from measure_human_winner_takeover import MODEL
from serial_worker import write_receipt
from train_early_deadline_student import model_arguments, native

INTAKE = ROOT/"build/replay-intake/intake-new-replay-families-729f8aa0fa7e1069e0e6"
FORMATS = (".jgs", ".jqh")
SEATS_PER_FORMAT = 8
TAGS = ("mid", "post")


def select(probe):
    result = {}
    families = set()
    for suffix in FORMATS:
        pool = [r for r in probe["rows"] if r["format"] == suffix and r["mid"] and r["post"]
                and not r["prior_layout"] and not r["prior_trajectory"]
                and (suffix != ".jgs" or not r["stopped"] and not r["notice"])]
        ordered = sorted(pool,key=lambda r:sha256(("phase-pair-v1:"+r["family"]).encode()).digest())
        found = 0
        for row in ordered:
            if row["family"] in families:
                continue
            families.add(row["family"])
            result[row["key"]] = row
            found += 1
            if found == SEATS_PER_FORMAT:
                break
        require(found == SEATS_PER_FORMAT, f"insufficient independent matched {suffix} cases")
    return result


def payload(cases):
    roster = model_roster()
    weights = [w for name in ("candidate","historical") for w in roster[name]["weights"]]
    student = json.loads(MODEL.read_bytes())
    prefix = " ".join(map(str,(1,1,*weights)))+" "+model_arguments(student,"replay_control")
    wires = {}
    for line in (INTAKE/"games.txt").read_text(encoding="utf8").splitlines():
        kind,key,rest = line.split(maxsplit=2)
        if key in cases:
            r = cases[key]
            if kind == "H":
                data = rest.split()
                last = max(r["mid"],r["post"])
                rest = " ".join([str(last), *data[1:1+100*3+4*last]])
            wires[key] = f'{kind} {key} {r["learner"]} {r["direction"]} {r["mid"]} {r["post"]} {rest}'
    require(set(wires) == set(cases),"missing registered replay wire")
    return prefix+"\n"+"\n".join(wires[key] for key in cases)+"\n"


def bound(first, other, equal):
    if equal:
        return (0,0)
    return ((0 if other<0 else other)-(2 if first<0 else first),
            (2 if other<0 else other)-(0 if first<0 else first))


def summarize_root(root):
    same = root["student"] == root["human"]
    panel = [[] for _ in range(4)]
    roots = root["runs"]
    for world in range(root["worlds"]):
        for desk in range(2):
            a, b = (roots[world,desk,k] for k in (0,1))
            require(not same or a == b,"same-action placebo diverged")
            panel[world//3*2+desk].append(bound(a[0],b[0],same))
    intervals = [(sum(lo for lo,_ in runs),sum(hi for _,hi in runs)) for runs in panel]
    if same:
        status = "same_action"
    elif root["worlds"] < 6:
        status = "limited_worlds"
    elif all(lo > 0 for lo,_ in intervals):
        status = "replicated_gain"
    elif all(hi < 0 for _,hi in intervals):
        status = "replicated_loss"
    elif any(lo > 0 for lo,_ in intervals) and any(hi < 0 for _,hi in intervals):
        status = "direction_flip"
    elif all(lo == hi == 0 for lo,hi in intervals):
        status = "tied"
    else:
        status = "unresolved"
    def reversed_order(a,b):
        return (a[0]>0 and b[1]<0) or (b[0]>0 and a[1]<0)

    return {"status":status, "panel_score_bounds_x2":intervals,
            "desk_flip":any(reversed_order(intervals[i],intervals[i+1]) for i in (0,2)),
            "sample_flip":any(reversed_order(intervals[i],intervals[i+2]) for i in (0,1)),
            "draw":sum(outcome == 1 for outcome,_ in roots.values()),
            "unresolved":sum(outcome == -1 for outcome,_ in roots.values()),
            "games":len(roots), "plies":sum(plies for _,plies in roots.values())}


def unpack(output,cases):
    results = {}
    for line in output.splitlines():
        tag,key,phase,*data = line.split()
        idx = (key,int(phase))
        require(key in cases and int(phase) in (0,1),"unexpected source or phase")
        row = list(map(int,data))
        if tag == "R":
            require(idx not in results,"duplicate root")
            results[idx] = dict(zip(("ply","legal","offered","student","human","worlds"),row,strict=True),runs={})
        else:
            world,desk,arm,outcome,plies = row
            root = results[idx]
            trial = world,desk,arm
            require(tag == "F" and world in range(root["worlds"]) and desk in (0,1) and arm in (0,1)
                    and outcome in (-1,0,1,2) and 1 <= plies <= 140 and trial not in root["runs"],
                    "unexpected branch")
            root["runs"][trial] = (outcome,plies)
    require(set(results)=={(key,i) for key,r in cases.items() for i,phase in enumerate(TAGS) if r[phase]}, "missing phase")
    rows = []
    for (key,phase),root in results.items():
        phase = TAGS[phase]
        meta = cases[key]
        require(root["ply"] == meta[phase] and len(root["runs"]) == root["worlds"]*4,"replay grid differed")
        evidence = summarize_root(root)
        rows.append(dict(key=key,family=meta["family"],format=meta["format"],phase=phase,
                         **{k:v for k,v in root.items() if k != "runs"},**evidence))
    return rows


def compare(rows):
    groups = {}
    for suffix in FORMATS:
        subset = [r for r in rows if r["format"]==suffix]
        pairs = {r["key"]:{v["phase"]:v["status"]=="replicated_gain" for v in subset if v["key"]==r["key"]}
                 for r in subset}
        mid = sum(pair["mid"] and not pair["post"] for pair in pairs.values())
        post = sum(pair["post"] and not pair["mid"] for pair in pairs.values())
        discordant = mid+post
        p_value = min(1., 2*sum(comb(discordant,i) for i in range(min(mid,post)+1))/2**discordant)
        groups[suffix] = dict(mid_only=mid,post_only=post,paired_exact_p=p_value)
        for phase in TAGS:
            window = [r for r in subset if r["phase"]==phase]
            groups[suffix][phase] = {
                "status":dict(Counter(r["status"] for r in window)),
                **{k:sum(r[k] for r in window) for k in ("offered","draw","unresolved","games","plies","desk_flip","sample_flip")},
            }
    primary = groups[".jgs"]
    conclusion = "inconclusive"
    if primary["paired_exact_p"] <= .05:
        conclusion = "supported" if primary["mid_only"] > primary["post_only"] else "refuted"
    return groups,conclusion


def main(audit):
    out = ROOT/"build/replay-phase"/os.environ["JUNQI_SERIAL_KEY"]
    out.mkdir(parents=True,exist_ok=True)
    cases = select(json.loads((INTAKE/"probe.json").read_bytes()))
    wire = payload(cases)
    if not audit:
        build(out)
    raw = native(out,"replay-phase-split",wire)
    rows = unpack(raw,cases)
    groups,conclusion = compare(rows)
    report = dict(input_sha256=sha256(wire.encode()).hexdigest(),groups=groups,outcome=conclusion,rows=rows)
    if audit:
        require((out/"branches.txt").read_text(encoding="utf8") == raw and
                json.loads((out/"summary.json").read_bytes()) == json.loads(json.dumps(report)),
                "phase experiment did not reproduce")
        print("JUNQI_RESEARCH "+conclusion)
    else:
        (out/"branches.txt").write_text(raw,encoding="utf8")
        write_receipt(out/"summary.json",report)
    print(json.dumps(groups))
    print("JUNQI_VERDICT accepted")


if __name__ == "__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected audit or no argument")
    main(bool(sys.argv[1:]))
