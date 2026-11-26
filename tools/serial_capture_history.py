"""Verify dynamic capture evidence on untouched anonymized JGS source families."""
from collections import Counter
from hashlib import sha256
import json
import os
import subprocess
import sys

from serial_worker import ROOT, write_receipt
from verify_hypothesis_archive import build

SOURCE = ROOT/"build/replay-intake/experience-census-20261122"
FIELDS = ("plies","roots","record_reads","tightened_masks","new_unique_losses","stock_checks","controls",
          "changed_contact_outcomes","sampled_worlds","compatible_worlds","contradicted_worlds",
          "unresolved_worlds","replay_controls_unresolved")
COURSE_FIELDS = ("ply","quiet_actions","contact_actions","old_outcomes","new_outcomes","contacts_changed",
                 "sampled_worlds","compatible_worlds","contradicted_worlds","unresolved_worlds",
                 "replay_control_unresolved","proven_outcomes","unresolved_outcomes")
COURSE_KEY = "43e317e81cc04f7c14362294d62d6f19a74ea4eaab62bf78a2178d6aede51b7e"


def select(holdout=False):
    probe = json.loads((SOURCE/"probe.json").read_bytes())
    blocked = {family for _,family in select()[0].values()} if holdout else set()
    salt = "joint-holdout-v1:" if holdout else "capture-evidence-v1:"
    pool = sorted((r for r in probe["rows"] if r["format"]==".jgs" and not r["stopped"] and not r["prior_layout"]
                   and r["result"] in (1,2,3) and r["family"] not in blocked),
                  key=lambda r:sha256((salt+r["family"]).encode()).digest())
    cases, families = {},set()
    for r in pool:
        if r["family"] in families:
            continue
        cases[r["key"]]=r["direction"],r["family"]
        families.add(r["family"])
        if len(cases)==24:
            break
    if len(cases)!=24:
        raise ValueError("missing archive families")
    tapes = {}
    for line in (SOURCE/"games.txt").read_text(encoding="utf8").splitlines():
        tag,key,tape = line.split(maxsplit=2)
        if tag=="G" and key in cases:
            tapes[key]=tape
    return cases,"".join(f"{direction} {tapes[key]}\n" for key,(direction,_) in cases.items())


def native(exe, mode, wire):
    raw = subprocess.run([exe,mode],cwd=ROOT,input=wire,encoding="ascii",capture_output=True,timeout=65)
    if raw.returncode:
        raise ValueError(f"native evidence verifier returned {raw.returncode}: {raw.stderr.strip()}")
    return raw.stdout.splitlines()


def course(exe):
    for line in (SOURCE/"games.txt").read_text(encoding="utf8").splitlines():
        kind,key,tape=line.split(maxsplit=2)
        if kind=="H" and key==COURSE_KEY:
            prefix=tape.split()[1:1+300+4*172]
            (row,)=native(exe,"--capture-course","172 "+" ".join(prefix)+"\n")
            result=dict(zip(COURSE_FIELDS,map(int,row.split()),strict=True))
            if (result["proven_outcomes"] | result["unresolved_outcomes"]) & 4 == 0:
                raise ValueError("proof discarded the course's real defensive win")
            return result
    raise ValueError("course prefix unavailable")


def archive(exe, holdout):
    cases, wire=select(holdout)
    lines=native(exe,"--capture-ledger",wire)
    if len(lines)!=len(cases):
        raise ValueError("missing archive verification rows")
    totals=Counter()
    report={}
    for (key,(_,family)),line in zip(cases.items(),lines,strict=True):
        counts=dict(zip(FIELDS,map(int,line.split()),strict=True))
        report[key]=dict(family=family,**counts)
        totals.update(counts)
    for required in ("roots","record_reads","controls","stock_checks"):
        if not totals[required]:
            raise ValueError(f"coverage missing: {required}")
    if (not totals["sampled_worlds"] or totals["sampled_worlds"]!=sum(totals[k] for k in
        ("compatible_worlds","contradicted_worlds","unresolved_worlds"))):
        raise ValueError("hypothesis outcomes do not cover the sampled worlds")
    return {"families": report, "totals": dict(families=len(report),**totals)}


def main(mode):
    out = ROOT/"build/capture-inventory"/os.environ["JUNQI_SERIAL_KEY"]
    out.mkdir(parents=True,exist_ok=True)
    exe=out/"hypothesis_archive.exe"
    if mode=="initial":
        build(out)
    summaries=[]
    for holdout in (False,True) if mode=="audit" else (False,):
        result=archive(exe,holdout)
        if holdout:
            result["course"]=course(exe)
        file=out/("joint-holdout.json" if holdout else "joint-initial.json")
        if holdout:
            repeated=archive(exe,True)
            repeated["course"]=course(exe)
            if repeated!=result:
                raise ValueError("independent cohort changed on repeat")
            write_receipt(file,result)
        elif mode=="audit":
            if json.loads(file.read_bytes())!=result:
                raise ValueError("capture evidence changed on repeat")
        else:
            write_receipt(file,result)
        summaries.append(result["totals"]["contradicted_worlds"])
        print(json.dumps(dict(holdout=holdout,**result["totals"],course=result.get("course"))))
    if mode=="audit":
        print("JUNQI_RESEARCH "+("supported" if all(summaries) else "inconclusive"))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    if sys.argv[1:] not in ([],["audit"]):
        raise ValueError("expected audit or no argument")
    main(sys.argv[1] if sys.argv[1:] else "initial")
