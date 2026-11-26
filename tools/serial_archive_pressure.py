"""Check full-information pressure on distinct, strictly restorable replay families."""
from collections import Counter
import hashlib
import json
import os
import subprocess
import sys

from measure_deadline_opening import ROOT, RULES, build, require
from serial_full_dual_pressure import pressure_controls, run_native
from serial_capture_history import SOURCE
from serial_worker import write_receipt


def archives():
    rows = json.loads((SOURCE/"probe.json").read_bytes())["rows"]
    eligible = (r for r in rows if r["verified_plies"] >= 100 and
                not any(r[k] for k in ("prior_layout", "prior_trajectory", "stopped", "notice")))
    unique = {}
    for row in sorted(eligible, key=lambda r: (r["format"] != ".jqh", -r["verified_plies"], r["key"])):
        unique.setdefault(row["family"], row)
    assigned = {}
    for suffix in (".jqh", ".jgs"):
        group = sorted((r for r in unique.values() if r["format"] == suffix),
                       key=lambda r: hashlib.sha256(("archive-pressure:"+r["family"]).encode()).digest())
        for i,row in enumerate(group):
            assigned[row["key"]] = dict(family=row["family"], format=suffix, audit=bool(i%2),
                                        ccw=row["direction"], cap=min(140,row["verified_plies"]))
    wires = {}
    with (SOURCE/"games.txt").open(encoding="utf8") as source:
        for line in source:
            tag,key,tape = line.split(maxsplit=2)
            if key in assigned:
                row = assigned[key]
                require(tag == {".jqh":"H", ".jgs":"G"}[row["format"]],"replay format mismatch")
                wires[key] = f'{tag} {key} {row["ccw"]} {row["cap"]} {tape.rstrip()}\n'
    require(wires.keys()==assigned.keys() and len(assigned)>100,"pressure archive does not cover declared replay families")
    return assigned,wires


def collect(directory, assigned, wires, audit):
    selected = {key:row for key,row in assigned.items() if row["audit"] == audit}
    run = subprocess.run([directory/"deadline_opening.exe",RULES,"archive-pressure"],
                         input="".join(wires[key] for key in selected),encoding="utf8",capture_output=True,timeout=60)
    require(run.returncode == 0, f"archive pressure could not execute: {run.stderr.strip()}")
    coverage, roots = {}, []
    columns = ("ply", "threats", "unassisted", "actions", "safe", "lost", "unknown", "chosen_lower", "chosen_upper",
               "human_lower", "human_upper", "chosen_terminal", "flag_nodes", "chosen", "human", "alternative",
               "chosen_team_lower", "chosen_team_upper", "safer_team_lower", "safer_team_upper")
    for line in run.stdout.splitlines():
        tag,key,*fields = line.split()
        require(key in selected,"unknown local replay key")
        if tag=="A":
            agreed,diff,reason = fields
            require(key not in coverage and 0<=int(agreed)<=selected[key]["cap"] and
                    (int(diff)==int(agreed)+1 if reason!="matched" else diff=="0"),"invalid replay boundary")
            coverage[key] = dict(agreed=int(agreed), first_difference=int(diff), reason=reason)
        else:
            require(tag=="P","unrecognized pressure evidence")
            row = dict(zip(columns,map(int,fields),strict=True))
            require(row["actions"]==sum(row[k] for k in ("safe","lost","unknown")) and row["threats"] in (1,2),
                    "unaccounted legal actions or bad threat count")
            roots.append(dict(key=key, **row))
    require(selected.keys()==coverage.keys() and all(r["ply"]<=coverage[r["key"]]["agreed"] for r in roots),
            "replay omitted a family or used a decision past first divergence")
    return dict(coverage=coverage,roots=roots)


def summarize(result):
    rows = result["roots"]
    missed = [r for r in rows if r["alternative"]>=0]
    return dict(families=len(result["coverage"]),prefix_plies=sum(r["agreed"] for r in result["coverage"].values()),
                rule_boundary=dict(Counter(r["reason"] for r in result["coverage"].values())),
                roots=len(rows), double=sum(r["threats"]==2 for r in rows),
                unassisted=sum(r["unassisted"] for r in rows),
                proven_no_flag_save=sum(r["lost"]==r["actions"] for r in rows),
                any_flag_save=sum(r["safe"]>0 for r in rows),
                picked_flag_exposure_with_alternative=len(missed),
                safer_team_proven_better=sum(r["safer_team_lower"]>r["chosen_team_upper"] for r in missed),
                safer_team_proven_worse=sum(r["safer_team_upper"]<r["chosen_team_lower"] for r in missed),
                direct_team_win=sum(r["chosen_terminal"]==2 for r in rows))


def main(audit):
    directory = ROOT/"build/archive-pressure"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        build(directory)
    assigned,wires = archives()
    results = collect(directory,assigned,wires,audit)
    summary = summarize(results)
    if audit:
        pressure_controls(directory)
        require("E1_min_first_child" in run_native(directory,"prove-evaluator-controls"),"solver regression did not run")
        first = json.loads((directory/"development.json").read_bytes())
        require(set(first["result"]["coverage"]).isdisjoint(results["coverage"]) and
                collect(directory,assigned,wires,False)==first["result"],"development replay or independent split changed")
        repeated = first["summary"]["picked_flag_exposure_with_alternative"] and summary["picked_flag_exposure_with_alternative"]
        print("JUNQI_RESEARCH "+("supported" if repeated else "inconclusive"))
    selected = {key:assigned[key] for key in results["coverage"]}
    write_receipt(directory/("confirmation.json" if audit else "development.json"),
                  dict(roots_source=selected,summary=summary,result=results))
    print(json.dumps(summary))
    print("JUNQI_VERDICT accepted")


if __name__ == "__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected audit or no argument")
    main(bool(sys.argv[1:]))
