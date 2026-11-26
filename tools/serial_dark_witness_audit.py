"""Reproduce first replay failures for archived and generated dark worlds."""
import json
import os
import shutil
import sys

from full_mode_runtime import FullModeRun, isolated, sha
from serial_dark_opening_reply import RULES, ROOT
from serial_dynamic_reply import build, require
from serial_worker import read, write_receipt


ARCHIVED=ROOT/"build/dark-reply-leaf/audit-dark-response-versus-leaf-value-rescreen-e8b9bc9e8f66f9515c29/replay-1.txt"
WINDOWS=ROOT/"build/dark-opening-windows/compare-dark-reply-opening-windows-f81ab1f4c7510ebba971"
BASELINE=ROOT/"build/dark-witness-audit/audit-dark-midgame-strict-witness-loss-6def43b8deda8a1b7a32"


def wire(audit):
    lines=[]
    for fold in ("development","confirmation"):
        for row in read(WINDOWS/(fold+".json"))["middle"]:
            if row["replay_failed"] and (audit != (not row["other"] and not row["bomb"])):
                seed,seat,table,cut=(row[k] for k in ("seed","learner","table","ply"))
                lines.append(f"G {seed}-{table} {seed} {seat} {cut} {table}")
    if not audit:
        line,=tuple(t for t in ARCHIVED.read_text(encoding="ascii").splitlines() if t.startswith("881bd1a4768b"))
        key,viewer,cut,direction,seed,_,_,tape=line.split()
        lines.append(f"J {key} {seed} {viewer} {cut} {direction} {tape}")
    return "\n".join(lines)+"\n",{line.split()[1]:int(line.split()[4]) for line in lines}


def run(audit):
    scope=FullModeRun(ROOT/"build/dark-witness-audit"/os.environ["JUNQI_SERIAL_KEY"])
    folder=scope.folder
    if not audit:
        scope.engine.mkdir(parents=True,exist_ok=True)
        build(scope.engine)
        shutil.copyfile(RULES,scope.rules)
        write_receipt(folder/"fingerprints.json",{p:sha(folder/p) for p in ("engine/deadline_opening.exe","engine/rules.json")})
    require(all(sha(folder/p)==s for p,s in read(folder/"fingerprints.json").items()),"witness audit inputs changed")
    payload,expected=wire(audit)
    with isolated(scope,ROOT/"build"):
        raw=scope.run("dark-witness-audit",payload)
    roots={}
    for tag,key,*fields in map(str.split,raw.splitlines()):
        if tag=="C":
            same,alternate,invalid=map(int,fields[:3])
            require(same==alternate==expected[key] and 0<invalid<=same and fields[-1]!="ok","strict replay controls failed")
            roots[key]=dict(cut=same,negative=dict(ply=invalid,reason=fields[-1]),attempts=[])
        else:
            require(tag=="F","unexpected strict witness output")
            result=dict(zip(("side","trial","status","accepted","ply"),map(int,fields[:5]),strict=True),reason=fields[-1])
            roots[key]["attempts"].append(result)
    require(roots.keys()==expected.keys() and all(len(r["attempts"])==6 for r in roots.values()),"missing replay audit attempts")
    write_receipt(folder/("confirmation.json" if audit else "development.json"),roots)
    if audit:
        baseline=read(BASELINE/"development.json")|read(BASELINE/"confirmation.json")
        combined=read(folder/"development.json")|roots
        require(combined.keys()==baseline.keys(),"witness comparison scope changed")
        key,=tuple(k for k in combined if k.startswith("881bd1a4768b"))
        require(all(combined[k]==baseline[k] for k in combined if k!=key),"archive flag constraint affected generated controls")
        before=[r for r in baseline[key]["attempts"] if r["accepted"]]
        after=[r for r in combined[key]["attempts"] if r["accepted"]]
        confirmed=not before and {r["side"] for r in after}=={0,1}
        print(json.dumps(dict(archived_witnesses_before=len(before),archived_witnesses_after=len(after),simulated_controls=6)))
        print("JUNQI_RESEARCH "+("supported" if confirmed else "inconclusive"))
    print(json.dumps(roots))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no argument or audit")
    run(bool(sys.argv[1:]))
