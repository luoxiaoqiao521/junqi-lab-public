"""Verify the replay-fitted opening on generated and independent real layouts."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from archive_action_screen import INPUT, ROOT, EXE, OUT, RESULT, assemble, compare, prefixes, initial_board, tape

TARGET=OUT/"archive-opening-gate.json"
LABELS=("frozen","fit_existing","fit_position")
COHORTS=("generated","public_holdout","local_holdout")


def original():
    payload=RESULT.read_bytes()
    report=json.loads(payload)
    text="\n".join(" ".join(map(str,report["weights"][n])) for n in LABELS)+"\n"
    return text,hashlib.sha256(payload).hexdigest()


def real_inputs():
    manifest={k:Path(v) for k,v in json.loads(INPUT.read_text(encoding="utf-8-sig")).items()}
    items, counts=prefixes(manifest)
    grouped=[[],[]]
    for split,(_,layouts,seats,events) in items:
        if split==0: continue
        board=initial_board(layouts)
        moves,_=tape(layouts,seats,events)
        x,y,_,_,_=next(m for m in moves if m[0]>=0)
        grouped[split-1].append(str((board[(x,y)][0]+1)%4))
        grouped[split-1].extend(f"{x} {y} {rank}" for (x,y),(_,rank) in board.items())
    return [f"{count}\n"+"\n".join(body)+"\n" for count,body in zip(counts[1:],grouped,strict=True)], counts[1:]


def run_games(mode,weights,extra):
    output=subprocess.run([EXE,mode],input=weights+extra,text=True,check=True,
                          capture_output=True,timeout=70).stdout
    return [list(map(int,line.split())) for line in output.splitlines()]


def summarize(simulations,counts):
    output={}
    for name,count in zip(COHORTS,counts,strict=True):
        full,opening=(simulations[name][mode] for mode in ("full","opening"))
        if any(a[:3]!=b[:3] for a,b in zip(full,opening,strict=True)):
            raise ValueError("paired conditions or frozen reference drifted")
        output[name]={mode:compare(rows,count) for mode,rows in simulations[name].items()}
    return output


def measure():
    weights,digest=original()
    assemble()
    payloads, real_counts=real_inputs()
    counts=[256,*real_counts]
    simulations={}
    for name, extra in zip(COHORTS,["",*payloads],strict=True):
        prefix="fresh" if name=="generated" else "real"
        simulations[name]={mode:run_games(prefix+"-"+mode,weights,extra) for mode in ("full","opening")}
    stats=summarize(simulations,counts)
    temp=TARGET.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(frozen=digest,counts=counts,results=stats,simulations=simulations),
                               indent=2)+"\n",encoding="utf8")
    temp.replace(TARGET)
    print("opening-only paired result",json.dumps(stats,sort_keys=True))


def audit():
    report=json.loads(TARGET.read_text(encoding="utf8"))
    if report["frozen"]!=original()[1] or report["counts"]!=[256,64,41]:
        raise ValueError("fitted weights or cohort size changed")
    summary=summarize(report["simulations"],report["counts"])
    if json.loads(json.dumps(summary))!=report["results"]:
        raise ValueError("evaluation or score bounds changed")
    print("PASS paired opening gate, real cohorts and same frozen model after handoff")


if __name__=="__main__":
    audit() if sys.argv[1:]==["audit"] else measure()
