"""Audit capture knowledge and compare replies against the frozen student's actual action."""
from collections import defaultdict
from hashlib import sha256
import json
import os
import sys

from evidence_model import load, sha
from learn_short import RULES as SHORT_TRAINING_RULES
from measure_deadline_opening import ROOT, RULES, build, require
from partner_crossplay import ARCHIVE, FILES
from serial_capture_history import COURSE_KEY, SOURCE, select
from serial_capture_replies import ROOT_FIELDS, choose, comparisons, sources
from serial_worker import write_receipt
from train_early_deadline_student import EXCHANGE, model_arguments, native

MODEL = ROOT/"build/deadline-opening/train-public-exchange-student-9bc43e1e537e5e447099/collision.json"
DISCOVERY_SEED = (2026112311, 2026112341)
CONFIRM_SEED = (2026112312, 2026112342)
FRESH_SEED = (2026112313, 2026112343)


def independent_sources(previous):
    census = json.loads((SOURCE/"probe.json").read_bytes())["rows"]
    excluded = {family for holdout in (False, True) for _, family in select(holdout)[0].values()}
    excluded.update(row["family"] for row in census if row["key"] in previous)
    pool = (row for row in census if row["format"]==".jgs" and row["mid"] and row["result"] in (1,2,3)
            and not any(row[field] for field in ("prior_layout","prior_trajectory","stopped","notice")))
    keys = {}
    for row in sorted(pool,key=lambda item:sha256(("student-capture-v1:"+item["family"]).encode()).digest()):
        if row["family"] not in excluded:
            keys[row["key"]] = row
            excluded.add(row["family"])
        if len(keys)==8:
            break
    result = {}
    for line in (SOURCE/"games.txt").read_text(encoding="utf8").splitlines():
        tag,key,tape = line.split(maxsplit=2)
        if key in keys:
            row = keys[key]
            require(tag=="G","unexpected independent source format")
            result[key] = (tag,row["mid"],row["direction"],tape)
    require(len(result)==8 and not set(result)&set(previous),"missing independent source roots")
    return result


def student_wire():
    rules, engine = ARCHIVE/"ruleset.json", ARCHIVE/"junqi.exe"
    require(sha(rules)==sha(SHORT_TRAINING_RULES),"frozen short policies use a different training ruleset")
    parent, peer = (load(ARCHIVE/FILES[role],engine,rules,"short-v1")["weights"] for role in ("candidate","historical"))
    artifact = json.loads(MODEL.read_bytes())
    return " ".join(map(str,peer+parent))+" "+model_arguments(artifact,"collision",EXCHANGE)


def read_panel(text, selected):
    roots, traces = {}, defaultdict(dict)
    for line in text.splitlines():
        tag,key,*values = line.split()
        require(key in selected,"unexpected source key")
        row = list(map(int,values))
        if tag=="S":
            actor,archived,actual,revised,pressure,features,collisions,*shortlist = row
            require(not traces[key] and actor in range(4) and 0<=collisions<=features
                    and len(shortlist)<=6 and set((actual,revised))<=set([pressure,*shortlist]),"invalid student trace")
            traces[key] = dict(actor=actor,archived=archived,actual=actual,revised=revised,pressure=pressure,
                               features_changed=features,collision_features_changed=collisions,shortlist=shortlist,
                               captures=[],loss_ranges=[])
        elif tag=="E":
            ply,attacker,defender,a,b,outcome,amask,dmask = row
            require(0<ply<selected[key][1] and a!=b and amask and dmask,"invalid prior capture evidence")
            traces[key]["captures"].append(row)
        elif tag=="R":
            require(key not in roots,"duplicate replay root")
            roots[key] = dict(zip(ROOT_FIELDS,row,strict=True),actions={})
        elif tag=="A":
            move,world,desk,outcome,plies = row
            trials = roots[key]["actions"].setdefault(move,{})
            require(0<=world<roots[key]["worlds"] and desk in (0,1) and (world,desk) not in trials
                    and outcome in(-1,0,1,2) and 1<=plies<=140,"invalid paired continuation")
            trials[world,desk] = (outcome,plies)
        elif tag=="L":
            other,victim,rank,total,lower,upper = row
            require(other!=victim and 0<=lower<=upper<=total,"invalid casualty range")
            traces[key]["loss_ranges"].append(row)
        else:
            raise ValueError(f"unexpected capture trace tag {tag}")
    require(roots.keys()==traces.keys()==selected.keys(),"missing student-anchored roots")
    for key,root in roots.items():
        require(root["original"]==traces[key]["actual"],"paired branch failed to use actual student action")
        root["trace"] = traces[key]
    return roots


def collect(directory, cases, seeds, locked=None):
    selected = cases if locked is None else {key:cases[key] for key in locked}
    wire = [student_wire()]
    for key,(kind,ply,ccw,tape) in selected.items():
        target = -1 if locked is None else locked[key]
        wire.append(f"{kind} {key} {ply} {target} {seeds[0]} {seeds[1]} {ccw} {tape}")
    roots = read_panel(native(directory,"student-capture-replies","\n".join(wire)+"\n"),selected)
    for key,root in roots.items():
        actions = root["actions"]
        expected = (root["legal"] if locked is None else 2) if root["worlds"] else 0
        require(root["ply"]==selected[key][1] and len(actions)==expected and
                all(len(trials)==2*root["worlds"] for trials in actions.values()),"incomplete student panel")
        if actions:
            require(root["original"] in actions and (locked is None or locked[key] in actions),"missing locked action")
    return roots


def report_row(root, alternative):
    bounds = comparisons(root,alternative) if alternative in root["actions"] else None
    trace = root["trace"]
    revised = trace["revised"]
    return {**{field:root[field] for field in ROOT_FIELDS},"trace":trace,"alternative":alternative,
            "alternative_in_shortlist":alternative is not None and alternative in [trace["pressure"],*trace["shortlist"]],
            "difference_x2_by_desk":bounds,
            "ledger_only_x2_by_desk":comparisons(root,revised) if revised!=trace["actual"] and revised in root["actions"] else None,
            "rollouts":sum(len(trials) for trials in root["actions"].values())}


def supported(row):
    bounds = row["difference_x2_by_desk"]
    return row["worlds"]==4 and bounds is not None and min(lo for lo,_ in bounds)>0


def print_summary(rows):
    print(json.dumps(dict(roots=len(rows),branches=sum(row["rollouts"] for row in rows.values()),
          feature_roots=sum(bool(row["trace"]["features_changed"]) for row in rows.values()),
          collision_roots=sum(bool(row["trace"]["collision_features_changed"]) for row in rows.values()),
          changed=sum(row["trace"]["actual"]!=row["trace"]["revised"] for row in rows.values()),
          locked=sum(row["alternative"] is not None for row in rows.values()),
          positive=sum(supported(row) for row in rows.values()))))


def main(audit):
    directory = ROOT/"build/visible-student-capture"/os.environ["JUNQI_SERIAL_KEY"]
    cases = sources()
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        build(directory)
        roots = collect(directory,cases,DISCOVERY_SEED)
        rows = {key:report_row(root,choose(root)) for key,root in roots.items()}
        write_receipt(directory/"development.json",rows)
        print_summary(rows)
    else:
        development = json.loads((directory/"development.json").read_bytes())
        lock = {key:r["alternative"] for key,r in development.items() if r["alternative"] is not None}
        confirmation = collect(directory,cases,CONFIRM_SEED,lock) if lock else {}
        require(all(root["trace"]==development[key]["trace"] for key,root in confirmation.items()),
                "changing world or reply seed changed data available before the move")
        new_sources = independent_sources(cases)
        heldout = collect(directory,new_sources,FRESH_SEED)
        confirmed = {key:report_row(root,lock[key]) for key,root in confirmation.items()}
        fresh = {key:report_row(root,choose(root)) for key,root in heldout.items()}
        fresh_lock = {key:r["alternative"] for key,r in fresh.items() if r["alternative"] is not None}
        fresh_confirmation = collect(directory,new_sources,CONFIRM_SEED,fresh_lock) if fresh_lock else {}
        require(all(root["trace"]==fresh[key]["trace"] for key,root in fresh_confirmation.items()),
                "changing confirmation streams changed the held-out student choice")
        reproduced = {key:report_row(root,fresh_lock[key]) for key,root in fresh_confirmation.items()}
        if lock:
            require(collect(directory,cases,CONFIRM_SEED,lock)==confirmation,"locked student continuations did not reproduce")
        ledger_cases = {key:row["trace"]["revised"] for key,row in (development|fresh).items()
                        if row["trace"]["revised"]!=row["trace"]["actual"]}
        ledger_panel = collect(directory,cases|new_sources,CONFIRM_SEED,ledger_cases) if ledger_cases else {}
        ledger_rows = {key:report_row(root,ledger_cases[key]) for key,root in ledger_panel.items()}
        usable = [key for key,row in ledger_rows.items() if supported(row)]
        report = dict(model=sha(MODEL),engine=sha(directory/"deadline_opening.exe"),encoding=EXCHANGE,
                      rules_source=sha(RULES),mode="student-capture-replies", course=COURSE_KEY,
                      confirmed=confirmed,fresh=fresh,reproduced=reproduced,
                      ledger_only=ledger_rows, confirmed_ledger_alternatives=usable,
                      capture_columns=["ply","attacker_initial_node","defender_initial_node","attacker_seat",
                                       "defender_seat","outcome","attacker_rank_mask","defender_rank_mask"],
                      loss_columns=["counterpart","victim","rank","collision_losses","minimum","maximum"])
        write_receipt(directory/"confirmation.json",report)
        for rows in (confirmed,fresh,reproduced):
            print_summary(rows)
        # Even confirmed local improvements need a visible positive/negative condition before training.
        evidence = "supported" if len(usable)>=2 and any(key in new_sources for key in usable) else "inconclusive"
        print("JUNQI_RESEARCH "+evidence)
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected audit or no argument")
    main(bool(sys.argv[1:]))
