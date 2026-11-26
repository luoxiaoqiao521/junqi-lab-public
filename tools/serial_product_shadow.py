"""Rebuild, launch and shadow the real portable product without changing its decision path."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
from statistics import median
import subprocess
import sys
from time import perf_counter

from compare_qq_rules import VC
from full_mode_runtime import FullModeRun, sha
from serial_dynamic_reply import ROOT, PRIOR, build, require
from serial_quiet_options import FROZEN
from serial_reply_window import model_wire
from serial_worker import read, write_receipt

DIST=ROOT/"dist/junqi-mvp"
MODES=("four-full","four-dark","four-dual","single-dark")
EXPECTED={"four-full":100,"four-dark":25,"four-dual":50,"single-dark":25}
COMMAND={"four-full":"simulate-full","four-dark":"simulate","four-dual":"simulate-dual","single-dark":"simulate-single"}


def run(args,where,timeout=95):
    start=perf_counter()
    result=subprocess.run(args,cwd=where,capture_output=True,encoding="utf8",errors="replace",timeout=timeout)
    require(result.returncode==0,f"{Path(args[0]).name} failed: {result.stderr.strip()}")
    return result.stdout,perf_counter()-start


def prepare(folder):
    stock=folder/"portable-old"
    current=folder/"portable-current"
    for directory in (stock,current):
        shutil.copytree(DIST,directory)
    shutil.copyfile(ROOT/"start-mvp.ps1",current/"start-mvp.ps1")
    sources=("core/board.cpp","core/layout.cpp","core/rules.cpp","core/state.cpp","sim/main.cpp")
    files=" ".join(f'"{ROOT.as_posix()}/{s}"' for s in sources)
    command=(f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /W4 /WX /O2 /MT /utf-8 '
             f'/I"{ROOT.as_posix()}/core" {files} /Fe:junqi.exe')
    temp=folder/"engine-build"
    temp.mkdir()
    subprocess.run(f'cmd.exe /d /s /c "{command}"',cwd=temp,check=True,timeout=100)
    shutil.copyfile(temp/"junqi.exe",current/"junqi.exe")
    check=folder/"shadow-engine"
    check.mkdir()
    build(check)
    return dict(old=sha(stock/"junqi.exe"),current=sha(current/"junqi.exe"),
                launcher_sha=sha(current/"start-mvp.ps1"),
                stock_bytes=sum(p.stat().st_size for p in stock.rglob("*") if p.is_file()),
                current_bytes=sum(p.stat().st_size for p in current.rglob("*") if p.is_file()))


def launch(folder,mode,seed):
    directory=folder/"portable-current"
    output=folder/"launches"/f"{mode}-{seed}"
    args=["powershell.exe","-NoProfile","-ExecutionPolicy","Bypass","-File",str(directory/"start-mvp.ps1"),
          "-NoOpen","-RunDirectory",str(output),"-Mode",mode,"-Seed",str(seed)]
    text,seconds=run(args,directory,45)
    logs=list(output.glob("*/game.jsonl"))
    require(len(logs)==1 and "Replay: " in text and "Private log: " in text,"launcher did not emit a replay")
    path,=logs
    rows=[json.loads(line) for line in path.read_text(encoding="utf8").splitlines()]
    header,initial,*frames=rows
    expected="flag-route"
    count=sum(rank>=0 for rank in initial["views"][0])
    require(count==EXPECTED[mode] and header["policy"]==expected and frames[-1]["result"]!=0 and
            (path.parent/"index.html").exists(),"launched mode or chosen policy differs")
    if mode=="single-dark":
        require(header["match"]=="single-dark" and all(r["turn"] in (0,2) for r in (initial,*frames)),
                "two-seat product used a nonparticipant")
    if expected=="flag-route":
        require(all(r["decision"]["policy"]=="flag-route-1" for r in frames),"practice did not use its current policy")
    demo=(path.parent/"demo.js").read_text(encoding="utf8")
    require(demo.startswith("window.JUNQI_LOG=") and demo.strip().endswith(";"),"portable page received no live replay")
    page=[json.loads(row) for row in json.loads(demo.strip().removeprefix("window.JUNQI_LOG=").removesuffix(";")).splitlines()]
    require(page[-1]["hash"]==frames[-1]["hash"] and page[1]["views"]==initial["views"],"page's replay diverged")
    return path,dict(hash=sha(path),plies=len(frames),result=frames[-1]["result"],last=frames[-1]["hash"],
                     rules=header["rules_hash"],vis=count,policy=expected,launch_sec=seconds,
                     actual_engine=sha(directory/"junqi.exe"),page_sha=sha(path.parent/"demo.js"))


def compare_compatible_inputs(folder,path,mode,seed):
    # Compare policies with the same rules and observation. Only the serialization changes.
    inputs=[]
    for name in ("ruleset.json","observation.json"):
        original=path.parent/name
        converted=path.parent/("literal-"+name)
        converted.write_text(json.dumps(read(original),ensure_ascii=False),encoding="utf8")
        inputs.append(converted)
    times={}
    for side in ("old","current"):
        binary=folder/f"portable-{side}"/"junqi.exe"
        log=path.parent/(side+"-normalized.jsonl")
        _,elapsed=run([binary,COMMAND[mode],str(seed),log,*inputs,"flag-route"],path.parent)
        require(log.read_bytes()==path.read_bytes(),"same release scenario diverges at product decision")
        times[side]=dict(hash=sha(log),engine=sha(binary),native_sec=elapsed)
    return times


def percentiles(vals):
    data=sorted(vals)
    return {f"p{t}":data[(len(data)*t+99)//100-1] for t in (50,95,99)}


def shadow(folder,path,actual):
    exe=folder/"shadow-engine"/"deadline_opening.exe"
    text,wall=run([exe,path,"product-shadow-replay"],folder)
    performance={i:[[],[],[]] for i in range(4)}
    steps=0
    negative=[]
    collisions=Counter()
    for line in text.splitlines():
        tag,*fields=line.split()
        if tag=="H":
            require(fields[0]==actual["rules"],"same replay loaded different rules")
        elif tag=="P":
            require(fields[0]!=fields[1],"negative control failed")
            negative.append(fields)
        elif tag=="M":
            ply,phase,legal,gen,choice,proc,state,collision=map(int,fields)
            steps+=1
            require(ply==steps and legal>0,"no legal choices in ongoing game")
            collisions[collision]+=1
            for values,current in zip(performance[phase],(gen,choice,proc),strict=True):
                values.append(current)
        elif tag=="E":
            require(fields[:3]==[str(actual["plies"]),str(actual["result"]),actual["last"]] and
                    int(fields[-1])==len(negative),"shadow outcome differs")
    require(steps==actual["plies"] and (len(negative)==1)==(actual["vis"]==100),"lost probe or final state")
    groups={i:dict(n=len(r[0]),gen=percentiles(r[0]),select=percentiles(r[1]),step=percentiles(r[2]))
            for i,r in performance.items() if r[0]}
    return dict(actions=steps,collisions=collisions,negative_control=negative,shadow_sec=wall,phases_ns=groups)


def compare_launches(folder,audit):
    seeds=(19057,19060) if audit else (42,44)
    rows=[]
    for mode in MODES:
        for seed in seeds:
            path,actual=launch(folder,mode,seed)
            rows.append(dict(mode=mode,seed=seed,product=actual,compatibility=compare_compatible_inputs(folder,path,mode,seed),
                             shadow=shadow(folder,path,actual)))
    return rows


def evaluation_trace(scope,parent,child):
    payload="2859600 4 "+model_wire(parent,"window")+" "+model_wire(child,"window")
    timings={False:[],True:[]}
    reference=None
    game_costs={False:[],True:[]}
    for strict in (False,True,True,False,True,False,False,True):
        begin=perf_counter()
        raw=scope.run("full-student-evolution-games",payload,strict=strict)
        timings[strict].append(perf_counter()-begin)
        projected=[]
        inner=0
        for tag,*fields in map(str.split,raw.splitlines()):
            require(tag=="G","unexpected comparable research output")
            inner+=int(fields[9])
            fields[9]="timing"
            projected.append(fields)
        if reference is None:
            reference=projected
        else:
            require(reference==projected,"same frozen game changed under current guard")
        game_costs[strict].append(inner)
    return dict(games_per_run=len(reference),measured_runs=8,unguarded_sec=percentiles(timings[False]),
                active_sec=percentiles(timings[True]),unguarded_median=median(timings[False]),
                active_median=median(timings[True]),native_update_micros={str(k):v for k,v in game_costs.items()})


def main(audit):
    folder=ROOT/"build/product-shadow"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        folder.mkdir(parents=True,exist_ok=True)
        write_receipt(folder/"binaries.json",prepare(folder))
    rows=compare_launches(folder,audit)
    result=dict(rows=rows,compared_plies=sum(r["shadow"]["actions"] for r in rows),binaries=read(folder/"binaries.json"))
    if audit:
        original=read(folder/"development.json")
        require({r["seed"] for r in rows}.isdisjoint(r["seed"] for r in original["rows"]),"launch holdout reused a seed")
        scope=FullModeRun(folder/"runtime-probe")
        scope.engine.mkdir(parents=True)
        shutil.copyfile(folder/"shadow-engine"/"deadline_opening.exe",scope.binary)
        shutil.copyfile(ROOT/"config/ruleset-selfplay.json",scope.rules)
        parent=read(PRIOR/"students.json")["models"]["window"]
        child=read(FROZEN/"student.json")["models"]["window"]
        result["matched_research_runtime"]=evaluation_trace(scope,parent,child)
        text,elapsed=run(["node",ROOT/"tests/web_logic.cjs"],ROOT,30)
        result["web_checks"]=dict(seconds=elapsed,stdout=text.strip())
        print("JUNQI_RESEARCH supported")
    write_receipt(folder/("confirmation.json" if audit else "development.json"),result)
    print(json.dumps(dict(compared_plies=result["compared_plies"],games=len(rows),
                          runtime=result.get("matched_research_runtime"),binaries=result["binaries"])))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(bool(sys.argv[1:]))
