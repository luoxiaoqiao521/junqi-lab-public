"""Evaluate deadline-aware public opening plans under matched full-game rules."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from statistics import mean, median
import subprocess
import sys

from archive_pressure_training import RULES
from compare_archive_cycle_exit_values import interval
from compare_qq_rules import VC
from partner_crossplay import model_roster
from pilot_relay_candidate_budget import file_sha

ROOT = Path(__file__).resolve().parents[1]
START, COUNT = 1089700, 24
SCORE_BUDGET = 0.025
EARLY_LOSS_BUDGET = 0.025
HORIZONS = (100,140,280)
COLUMNS = ("seed seat first table arm plies outcome ending changed changed100 changed140 "
           "contact enemy_retired friendly_retired pre_exit_trace").split()
SCORE = {-1:None,0:0,1:.5,2:1}
LABEL = {-1:"unresolved",0:"loss",1:"draw",2:"win"}
TABLES = ("random","historical")
ARMS = ("frozen","single_seat","allied_pair","full_table")


def require(ok,message):
    if not ok:
        raise ValueError(message)


def build(directory,root=ROOT):
    sources=("core/board.cpp","core/layout.cpp","core/rules.cpp","core/state.cpp","tests/deadline_opening.cpp")
    files=" ".join(f'"{root.as_posix()}/{s}"' for s in sources)
    args=(f'call "{VC}" >nul && cl /nologo /std:c++17 /EHsc /W4 /WX /O2 /MT /utf-8 '
          f'/I"{root.as_posix()}/core" {files} /Fe:deadline_opening.exe')
    subprocess.run(f'cmd.exe /d /s /c "{args}"',cwd=directory,check=True,timeout=100)


def collect(directory,conversion=False):
    roster=model_roster()
    configs=[roster[k] for k in ("candidate","historical")]
    weights=[n for config in configs for n in config["weights"]]
    first_seed=START+100 if conversion else START
    stdin=" ".join(map(str,(first_seed,COUNT,*weights)))
    exe=directory/"deadline_opening.exe"
    arguments=[exe,RULES]+(["conversion"] if conversion else [])
    run=subprocess.run(arguments,input=stdin,encoding="utf8",errors="replace",
                       capture_output=True,text=True,timeout=200,check=False)
    require(run.returncode==0 and run.stderr.strip()=="PASS early capture, phase risk, and hidden-identity isolation",
            f"deadline native execution did not pass: {run.stderr.strip()}")
    rows=[dict(zip(COLUMNS,map(int,line.split()),strict=True)) for line in run.stdout.splitlines()]
    arms=(2,4) if conversion else range(4)
    expected=[(t,a,s,j) for t in range(2) for a in arms
              for s in range(first_seed,first_seed+COUNT) for j in range(4)]
    require([(r["table"],r["arm"],r["seed"],r["seat"]) for r in rows]==expected,"incomplete sample grid")
    require(all(0<r["plies"]<=3000 and r["outcome"] in SCORE and
                0<=r["changed100"]<=r["changed140"]<=r["changed"]<=r["plies"] for r in rows),
            "invalid native row")
    fingerprint=dict(engine_sha256=file_sha(exe), rules_sha256=file_sha(RULES),seeds=[first_seed,COUNT],
                     frozen_weights=[item["artifact_sha256"] for item in configs],
                     score_budget=SCORE_BUDGET,early_loss_budget=EARLY_LOSS_BUDGET)
    return rows,fingerprint


def measure(rows):
    outcomes=Counter(LABEL[r["outcome"]] for r in rows)
    known=[r["plies"] for r in rows if r["outcome"]>=0]
    totals=[r["plies"] for r in rows]
    groups={str(h):dict(Counter(LABEL[r["outcome"]] if r["plies"]<=h else "unresolved" for r in rows))
            for h in HORIZONS}
    return dict(outcomes=dict(outcomes),mean_observed=mean(totals),median_observed=median(totals),
                p90_observed=sorted(totals)[(9*len(rows)+9)//10-1],
                mean_complete=mean(known) if len(known)==len(rows) else None,
                before=groups,changed100=sum(r["changed100"] for r in rows),
                changed140=sum(r["changed140"] for r in rows),
                first_enemy_exit={str(h):sum(0<r["enemy_retired"]<=h for r in rows) for h in HORIZONS},
                first_friendly_exit={str(h):sum(0<r["friendly_retired"]<=h for r in rows) for h in HORIZONS})


def paired(a,b,old,new):
    blocks=defaultdict(list)
    for x,y in zip(a,b,strict=True):
        require(tuple(x[k] for k in ("seed","seat","first"))==tuple(y[k] for k in ("seed","seat","first")),
                "paired deployment/first seat moved")
        sx,sy=SCORE[x["outcome"]],SCORE[y["outcome"]]
        blocks[x["seed"]].append(((sy if sy is not None else 0)-(sx if sx is not None else 1),
                                   (sy if sy is not None else 1)-(sx if sx is not None else 0),
                                   y["plies"]-x["plies"]))
    low,high,tempo=map(list,zip(*(tuple(map(mean,zip(*rows))) for rows in blocks.values())))
    gains={str(h):new["before"][str(h)].get("win",0)-old["before"][str(h)].get("win",0) for h in HORIZONS}
    early_losses=new["before"]["140"].get("loss",0)-old["before"]["140"].get("loss",0)
    bounds=[mean(low),mean(high)]
    evidence_interval=[interval(low)[0],interval(high)[1]]
    within_exploration=bounds[0]>=-SCORE_BUDGET and early_losses/len(b)<=EARLY_LOSS_BUDGET
    final_wins=new["outcomes"].get("win",0)>=old["outcomes"].get("win",0)
    minimum=(within_exploration and final_wins and new["mean_complete"] is not None
             and new["mean_complete"]<=140 and gains["140"]>0)
    return dict(score_bounds=bounds,score_outer_interval=evidence_interval,paired_turns=mean(tempo),
                new_wins=gains,extra_losses_by_140=early_losses,
                exploratory_budget_ok=within_exploration,minimum_140=minimum,
                stretch_100=minimum and new["mean_complete"]<=100 and gains["100"]>0)


def summarize(rows):
    answer={}
    for table,label in enumerate(TABLES):
        partitions=[[r for r in rows if r["table"]==table and r["arm"]==arm] for arm in range(4)]
        summaries=list(map(measure,partitions))
        answer[label]=dict(zip(ARMS,summaries,strict=True))
        # Candidate versus frozen opponent conditions is distinct from training-table pace.
        for index in (1,2):
            answer[label][ARMS[index]]["vs_frozen"]=paired(partitions[0],partitions[index],summaries[0],summaries[index])
    return answer


def conversion_summary(rows):
    report={}
    for table,label in enumerate(TABLES):
        continuing=[row for row in rows if row["table"]==table and row["arm"]==2]
        stopping=[row for row in rows if row["table"]==table and row["arm"]==4]
        for a,b in zip(continuing,stopping,strict=True):
            require(all(a[k]==b[k] for k in ("enemy_retired","contact","pre_exit_trace")),
                    "first enemy retirement or contact diverged before switch")
            if not a["enemy_retired"]:
                require(all(a[k]==b[k] for k in COLUMNS if k!="arm"),"unswitched trials diverged")
        panels={"continue_pressure":measure(continuing),"return_to_frozen":measure(stopping)}
        panels["paired_continuation"]=paired(stopping,continuing,panels["return_to_frozen"],panels["continue_pressure"])
        for name,block in (("continue_pressure",continuing),("return_to_frozen",stopping)):
            won=[r for r in block if r["outcome"]==2 and r["enemy_retired"]]
            panels[name]["median_gap_in_wins"]=median(r["plies"]-r["enemy_retired"] for r in won) if won else None
        report[label]=panels
    return report


def main(audit,conversion):
    key=os.environ.get("JUNQI_SERIAL_KEY","manual")
    require(key and all(c.isalnum() or c=="-" for c in key),"invalid serial key")
    out=ROOT/"build/deadline-opening"/key
    if not audit:
        out.mkdir(parents=True,exist_ok=True)
        build(out)
    rows,inputs=collect(out,conversion)
    report=conversion_summary(rows) if conversion else summarize(rows)
    result=dict(inputs=inputs,report=report,rows=rows)
    path=out/"summary.json"
    if audit:
        require(json.loads(path.read_text(encoding="utf8"))==result,"matched trials changed on audit")
    else:
        temp=path.with_suffix(".tmp")
        temp.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf8")
        temp.replace(path)
    print(json.dumps(result["report"],sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    if sys.argv[1:] not in ([],["audit"]):
        raise SystemExit("expected no arguments or audit")
    main(audit=sys.argv[1:] == ["audit"], conversion=False)
