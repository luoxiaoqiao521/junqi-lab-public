"""Verify archived starting positions and natural alternate continuations."""
from collections import Counter
import json
import os
import subprocess
import sys

from probe_replay_midgame_takeover import CONFIG, ROOT, RULES, compile_native, prepare


LANDMARKS = (48,96,144)
ROOT_FIELDS = ("seat", "clock", "legal", "recorded_action", "moved_dark", "restricted", "archived_end")
BRANCH_FIELDS = ("first_action", "moves", "outcome", "next_seat", "retired_mask", "trace")
TABLES = (0,1)
ARMS = (0,1)


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def move_counts(feed):
    counts = {}
    for line in feed.splitlines()[1:]:
        family, content = line.split()
        tape = bytes.fromhex(content)
        count = int.from_bytes(tape[26:28],"little")
        rows = tape[412:412+10*count:10]
        counts[family] = rows.count(95)
    return counts


def parse(output, proposed):
    detail, states, branches, exclusions = {},{},{},set()
    for line in output.splitlines():
        tag,family,*payload = line.split()
        require(family in proposed, "unregistered source")
        data = list(map(int,payload))
        if tag=="D":
            require(family not in detail and len(data)==20, "invalid replay detail")
            detail[family] = data
        else:
            ply,*data = data
            require(ply in LANDMARKS, "unexpected landmark")
            root = (family,ply)
            if tag=="K":
                require(root not in states, "duplicate restored position")
                states[root] = dict(zip(ROOT_FIELDS,data,strict=True))
            elif tag=="B":
                table,arm,*data = data
                branch = root+(table,arm)
                require(branch not in branches, "duplicate continuation")
                branches[branch] = dict(zip(BRANCH_FIELDS,data,strict=True))
            else:
                require(tag=="X" and not data and root not in exclusions,"invalid rejected position")
                exclusions.add(root)
    require(detail.keys()==proposed.keys() and not(states.keys() & exclusions),"incomplete source ledger")
    require(set(branches)=={key+(t,a) for key in states for t in TABLES for a in ARMS},
            "incomplete continuation pairs")
    return detail,states,branches,exclusions


def summarize(proposed, detail, states, branches, exclusions):
    records, counts = [], {}
    for ply in LANDMARKS:
        tally = Counter()
        for family, moves in proposed.items():
            key = family,ply
            root = states.get(key)
            if root:
                status = "branchable"
                require(root["legal"]>0 and 0<=root["seat"]<4 and ply<moves,"invalid root facts")
                tally["known_archived_end"] += root["archived_end"]>=0
                tally["nontrivial_historical_constraints"] += root["restricted"]>0
                for table in TABLES:
                    first,forced = (branches[key+(table,a)] for a in ARMS)
                    require(forced["first_action"]==root["recorded_action"],"forced move differs from archive")
                    require(all(1<=r["moves"]<=4 and -1<=r["outcome"]<=2
                                and -1<=r["next_seat"]<4 for r in (first,forced)), "invalid first cycle")
                    if first["first_action"]==forced["first_action"]:
                        require(first==forced, "identical first actions produced different continuations")
                    else:
                        tally["distinct_branches"] += 1
                records.append(dict(family=family,ply=ply,status=status,root=root,
                    branches=[{"table":table,"arm":arm,**branches[key+(table,arm)]}
                              for table in TABLES for arm in ARMS]))
            else:
                status = "branch_rejected" if key in exclusions else (
                    "short_record" if moves<=ply else "prefix_rejected")
                records.append(dict(family=family,ply=ply,status=status,verified_plies=detail[family][0]))
            tally[status]+=1
        counts[str(ply)] = dict(tally)
    require(len(records)==len(proposed)*len(LANDMARKS), "missing planned starts")
    return records,counts


def main(audit):
    directory = ROOT/"build/replay-curriculum"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        compile_native(directory)
    feed, source = prepare()
    proposed = move_counts(feed)
    process = subprocess.run([directory/"takeover.exe",CONFIG,RULES,"240","curriculum-roots"],
                             input=feed,capture_output=True,encoding="utf8",timeout=130)
    require(process.returncode==0, f"native curriculum probe failed: {process.stderr.strip()}")
    records, counts = summarize(proposed,*parse(process.stdout,proposed))
    require(any(r["status"]=="branchable" for r in records),"no replay roots could be recovered")
    data = dict(provenance=source,counts=counts,records=records)
    wire = (json.dumps(data,sort_keys=True,ensure_ascii=False,separators=(",",":"))+"\n").encode()
    path = directory/"summary.json"
    if audit:
        require(path.read_bytes()==wire,"archive roots and continuations did not reproduce")
    else:
        pending = path.with_suffix(".tmp")
        pending.write_bytes(wire)
        pending.replace(path)
    print(json.dumps({"sources":len(proposed),"landmarks":counts},sort_keys=True))
    for diagnostic in process.stderr.splitlines():
        print(diagnostic)
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(sys.argv[1:]==["audit"])
