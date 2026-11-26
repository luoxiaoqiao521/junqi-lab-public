"""Check exact team outcomes from public positions and audit the normal simulator entry."""
from collections import Counter
import json
import os
import subprocess
import sys
from time import perf_counter

from measure_deadline_opening import ROOT, RULES, build, require
from serial_worker import read, write_receipt


def controls(directory, seed):
    result = subprocess.run([directory/"deadline_opening.exe", RULES, "visible-terminal"],
                            input=str(seed), text=True, capture_output=True, timeout=30, check=True)
    rows = [line.split() for line in result.stdout.splitlines()]
    require([row[0] for row in rows] == ["C"]*5+["B","G"], "incomplete settlement sample")
    return {rows[i][0]+str(i):list(map(int,rows[i][1:])) for i in range(len(rows))}


def simulate(engine, entry, policy, seed, path):
    begin = perf_counter()
    subprocess.run([engine,entry,str(seed),path,RULES,ROOT/"config/observation.json",policy],
                   check=True,capture_output=True,timeout=15)
    return [json.loads(line) for line in path.read_bytes().splitlines()], perf_counter()-begin


def compare_entry(directory):
    engine = ROOT/"build/junqi.exe"
    result = []
    for seed in range(2312000,2312016):
        logs=[]
        elapsed=[]
        for version in (2,3):
            path=directory/f"full-{seed}-v{version}.jsonl"
            rows, seconds=simulate(engine,"simulate-full",f"heuristic-{version}-south",seed,path)
            subprocess.run([engine,"replay",path],check=True,capture_output=True,timeout=15)
            logs.append(rows)
            elapsed.append(round(seconds,4))
        old, new=logs
        require(old[0]["layouts"]==new[0]["layouts"] and old[1]==new[1],"paired root drift")
        first=next(((x["seq"],x["event"],y["event"]) for x,y in zip(old[2:],new[2:]) if x["event"]!=y["event"]),None)
        require(first or len(old)==len(new),"same actions unexpectedly ended at different steps")
        if first:
            require(first[1][0]==first[2][0]==0 and old[first[0]]["hash"]==new[first[0]]["hash"],
                    "first change was not at the controlled seat's common root")
        exact=Counter(row["result"] for row in new[2:] if abs(row["decision"]["objective"])==20000)
        require(set(exact)<={1,2},"exact win/loss priority triggered without settled team result")
        result.append(dict(seed=seed,first_change=first,plies=[old[-1]["seq"],new[-1]["seq"]],
                           results=[old[-1]["result"],new[-1]["result"]],exact=exact,seconds=elapsed))

    baseline=directory/"old-engine-v2.jsonl"
    local=directory/"current-engine-v2.jsonl"
    simulate(ROOT/"build/visible-terminal-baseline/junqi.exe","simulate-full","heuristic-2-south",2312100,baseline)
    simulate(engine,"simulate-full","heuristic-2-south",2312100,local)
    require(baseline.read_bytes()==local.read_bytes(),"previous v2 engine output drifted")
    subprocess.run([engine,"replay",baseline],check=True,capture_output=True,timeout=15)

    dark_a=directory/"dark-v2.jsonl"
    dark_b=directory/"dark-v3.jsonl"
    a,_=simulate(engine,"simulate","heuristic-2-ns",2312200,dark_a)
    b,_=simulate(engine,"simulate","heuristic-3-ns",2312200,dark_b)
    a[0]["policy"]=b[0]["policy"]
    for row in a[2:]:
        if row["decision"]["policy"]=="heuristic-2":
            row["decision"]["policy"]="heuristic-3"
    require(a==b,"masked simulator changed gameplay in the negative control")
    return dict(games=result,old_v2_identical=True,dark_same_trace=True,
                changed=sum(bool(row["first_change"]) for row in result),
                exact_choices=sum(sum(row["exact"].values()) for row in result),
                total_seconds=[round(sum(row["seconds"][index] for row in result),3) for index in range(2)])


def main(audit):
    out=ROOT/"build/visible-terminal"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        out.mkdir(parents=True,exist_ok=True)
        build(out)
        report=controls(out,2311380)
        write_receipt(out/"controls.json",report)
    else:
        require(controls(out,2311380)==read(out/"controls.json"),"registered controls failed to reproduce")
        report=dict(confirmation=controls(out,2311480),entry=compare_entry(out))
        write_receipt(out/"confirmation.json",report)
        print("JUNQI_RESEARCH supported")
    print(json.dumps(report))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected audit or no argument")
    main(bool(sys.argv[1:]))
