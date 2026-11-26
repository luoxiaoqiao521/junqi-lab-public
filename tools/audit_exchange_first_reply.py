"""Audit a single decision after a natural exchange, then test its full-game effect."""
from collections import Counter, defaultdict
from itertools import product
import json
import math
import os
import statistics
import subprocess
import sys

from audit_human_timing import sanitized_tape
from audit_qq_archive import decode
from cross_partner_deadline_student import encode, persist
from measure_deadline_opening import ROOT, RULES, SCORE, build, model_roster, paired, require
from pilot_relay_candidate_budget import ROOTS
from train_early_deadline_student import COLUMNS, model_arguments, native

ARCHIVE = ROOT/"build/replay-rule-variants/audit-replay-rule-variants-6a659a373631ad6cc804"/"summary.json"
PARENT = ROOT/"build/deadline-opening/train-student-from-own-visits-53bd170e2ea17edfa6ab"/"own_visits.json"
START, COUNT, FAMILY_COUNT, REPS = 1096100, 32, 32, 8
MIN_GAIN, MAX_EXTRA_LOSS = .01, .025
PIVOT = ("contact_ply contact_outcome next_ply choices base alternate method samples prefix replies "
         "survives first_reply ally_contact enemy_contacts").split()
END = ("plies outcome ending trace decisions legal_contact selected_contact max_quiet last_collision reverse "
       "last_reverse friendly_exit enemy_exit friendly_reason enemy_reason").split()
RESULT = {-1:"cutoff",0:"loss",1:"draw",2:"win"}


def archives():
    ledger = json.loads(ARCHIVE.read_bytes())
    candidates = {}
    for key,info in sorted(ledger["sources"].items()):
        family = info["family"]
        state = ledger["games"][key]
        if int(family[:8],16)%4==0 and state["valid"] and state["end"]["ccw"]==0:
            candidates.setdefault(family,key)
    selected = set(list(candidates.values())[:FAMILY_COUNT])
    require(len(selected)==FAMILY_COUNT,"insufficient prior sealed layout families")
    tapes={}
    for folder in ROOTS:
        for path in sorted(folder.rglob("*.jgs")):
            try:
                raw=path.read_bytes()
                key,_,_,events=decode(raw)
            except (ValueError,OSError):
                continue
            if key in selected:
                tapes.setdefault(key,sanitized_tape(raw,events))
    require(tapes.keys()==selected,"missing archived inputs")
    return "".join(f"{key} {tapes[key]}\n" for key in sorted(tapes)), sorted(tapes)


def parse(output, expected, archive):
    rows={}
    for line in output.splitlines():
        tag,key,*payload=line.split()
        values=list(map(int,payload))
        require(key in expected,"unregistered deployment")
        if tag=="N":
            require(key not in rows and len(values)==1 and 0<=values[0]<4,"duplicated game")
            rows[key]=dict(first=values[0],runs={})
        elif tag=="P":
            d=rows[key]
            require("pivot" not in d and "exit" not in d,"duplicate decision")
            d["pivot"]=dict(zip(PIVOT,values,strict=True))
        elif tag=="X":
            d=rows[key]
            require("pivot" not in d and "exit" not in d,"duplicate no-return")
            d["exit"]=values
        else:
            rep,arm,*stats=values
            runs=rows[key]["runs"]
            require(tag=="F" and arm in (0,1) and f"{rep}/{arm}" not in runs,"duplicate rollout")
            runs[f"{rep}/{arm}"]=dict(zip(END,stats,strict=True))
    require(rows.keys()==set(expected),"incomplete game list")
    for data in rows.values():
        validate(data,archive)
    return rows


def validate(data,archive):
    p=data.get("pivot")
    runs=data["runs"]
    require(("exit" in data)!=(p is not None),"not exactly one trigger outcome")
    reps=REPS if archive and p else 0 if archive else 1
    require(set(runs)=={f"{r}/{a}" for r,a in product(range(reps),(0,1))},"missing paired rollout")
    if p:
        require(24<p["contact_ply"]<p["next_ply"] and p["choices"]>=1 and
                0<=p["base"]<129**2 and 0<=p["alternate"]<129**2 and
                (p["choices"]>1 or p["base"]==p["alternate"]),"invalid prefix or candidate")
    for r in range(reps):
        a,b=(runs[f"{r}/{i}"] for i in (0,1))
        if not p or p["base"]==p["alternate"]:
            require(a==b,"identical decision yielded different path")
        for row in (a,b):
            require(row["plies"]>0 and row["outcome"] in SCORE and 0<row["ending"]<=5 and
                    0<=row["selected_contact"]<=row["legal_contact"]<=row["decisions"]
                    and 0<=row["last_collision"]<=row["plies"],"invalid referee ledger")


def compare_archive(rows):
    facts=Counter()
    for d in rows.values():
        p=d.get("pivot")
        if not p:
            contact,out,_=d["exit"]
            facts["no_exchange" if not contact else "retired" if out else "ended_before_return"]+=1
            continue
        if p["base"]==p["alternate"]:
            facts["only_legal" if p["choices"]==1 else "no_teacher_samples" if p["method"]==4 else "same_move"]+=1
            continue
        runs=d["runs"]
        scores=[(SCORE[runs[f"{rep}/0"]["outcome"]],SCORE[runs[f"{rep}/1"]["outcome"]]) for rep in range(REPS)]
        if any(None in pair for pair in scores):
            facts["cutoff_in_changed_root"]+=1
            continue
        a=statistics.mean(b-a for a,b in scores[:4])
        b=statistics.mean(b-a for a,b in scores[4:])
        d["score_batches"]=[a,b]
        facts["repeated_help" if min(a,b)>0 else "repeated_harm" if max(a,b)<0 else
              "reversal" if a*b<0 else "inconclusive"]+=1
    return dict(facts)


def horizons(rows, horizon):
    keys=("win","draw","loss","still_running","cutoff")
    tally=Counter(RESULT[r["outcome"]] if r["plies"]<=horizon else "still_running" for r in rows)
    return {key:tally[key] for key in keys}


def plain(rows):
    done=[r["plies"] for r in rows if r["outcome"]>=0]
    return dict(n=len(rows),results=dict(Counter(RESULT[r["outcome"]] for r in rows)),
                end_reasons=dict(Counter(r["ending"] for r in rows)),at_100=horizons(rows,100),at_140=horizons(rows,140),
                median_done=statistics.median(done) if done else None,
                p90_done=sorted(done)[math.ceil(.9*len(done))-1] if done else None,
                mean_done=statistics.mean(done) if done else None)


def inputs_for_pair(rows):
    return dict(before={str(n):horizons(rows,n) for n in (100,140,280)},
                outcomes=Counter(RESULT[r["outcome"]] for r in rows),
                mean_complete=statistics.mean(r["plies"] for r in rows) if all(r["outcome"]>=0 for r in rows) else None)


def panel(rows):
    old,new=[],[]
    reasons=Counter()
    flags=Counter()
    for key,data in rows.items():
        seed,seat,*_=map(int,key.split("/"))
        p=data.get("pivot")
        for arm,group in enumerate((old,new)):
            group.append(dict(seed=seed,seat=seat,first=data["first"],**data["runs"][f"0/{arm}"]))
        if p:
            changed=p["base"]!=p["alternate"]
            status="changed" if changed else "only_legal" if p["choices"]==1 else (
                   "no_teacher_samples" if p["method"]==4 else "same_move")
            flags.update(reached=1, changed_by_140=changed and p["next_ply"]<=140,
                         changed_after_140=changed and p["next_ply"]>140,
                         no_surviving_attacker=not p["survives"], ally_contact=p["ally_contact"])
        else:
            contact,out,_=data["exit"]
            status="no_exchange" if not contact else "retired" if out else "ended_before_return"
        reasons[status]+=1
    comparison=paired(old,new,inputs_for_pair(old),inputs_for_pair(new))
    delta=[]
    for a,b in zip(old,new,strict=True):
        if SCORE[a["outcome"]] is not None and SCORE[b["outcome"]] is not None:
            delta.append((a["seed"],SCORE[b["outcome"]]-SCORE[a["outcome"]]))
    blocks=defaultdict(list)
    for seed,value in delta:
        blocks[seed].append(value)
    effect_layouts=[statistics.mean(x) for x in blocks.values()]
    projected=math.ceil((1.96*statistics.stdev(effect_layouts)/MIN_GAIN)**2) if len(effect_layouts)>1 else None
    changed=[(a,b) for a,b in zip(old,new,strict=True) if a["trace"]!=b["trace"]]
    n=len(rows)
    require(reasons["changed"]==len(changed),"changed action did not change trace")
    return dict(
        baseline=plain(old), alternate=plain(new), comparison=comparison, triggers=dict(reasons), timing=dict(flags),
        upper_limit=reasons["changed"]/n, contact_per_100_decisions=100*sum(a["legal_contact"] for a in old)/sum(
            a["decisions"] for a in old), chosen_given_legal=sum(a["selected_contact"] for a in old)/max(
            1,sum(a["legal_contact"] for a in old)),
        transitions=dict(Counter(f'{RESULT[a["outcome"]]}->{RESULT[b["outcome"]]}' for a,b in changed)),
        projected_layouts_to_halfwidth_1pp=projected)


def long_baseline(natural):
    cells=[d["runs"]["0/0"] for d in natural.values() if d["runs"]["0/0"]["plies"]>140]
    if not cells:
        return {"count":0}
    return dict(count=len(cells), end_reasons=dict(Counter(r["ending"] for r in cells)),
                outcomes=dict(Counter(RESULT[r["outcome"]] for r in cells)),
                median_longest_quiet=statistics.median(r["max_quiet"] for r in cells),
                median_last_collision=statistics.median(r["last_collision"] for r in cells),
                same_piece_reversals=sum(r["reverse"] for r in cells),
                no_enemy_retirement_by_140=sum(r["enemy_exit"]==0 or r["enemy_exit"]>140 for r in cells),
                latest_reverse=max(r["last_reverse"] for r in cells))


def verify_parent(directory,natural,payload):
    genuine=native(directory,"student-apply",payload)
    expected={}
    for line in genuine.splitlines():
        x=dict(zip(COLUMNS,map(int,line.split()),strict=True))
        key="/".join(str(x[n]) for n in ("seed","seat","partner","table"))
        expected[key]=x
    require(expected.keys()==natural.keys(),"parent roster mismatch")
    for key,d in natural.items():
        a=d["runs"]["0/0"]
        x=expected[key]
        require(d["first"]==x["first"] and all(a[k]==x[k] for k in ("trace","outcome","plies")),
                "frozen parent did not reproduce normal path")


def main(audit):
    d=ROOT/"build/exchange-first-reply"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        d.mkdir(parents=True,exist_ok=True)
        build(d)
    roster=model_roster()
    weights=[w for name in ("candidate","historical") for w in roster[name]["weights"]]
    params=" ".join(map(str,(START,COUNT,*weights)))
    archive_payload,ids=archives()
    archive=parse(native(d,"exchange-archive",params+"\n"+archive_payload),ids,True)
    archive_summary=compare_archive(archive)
    parent=json.loads(PARENT.read_bytes())
    payload=params+" "+model_arguments(parent,"own_visits")
    keys={"/".join(map(str,b)) for b in product(range(START,START+COUNT),range(4),(1,2),(0,1))}
    natural=parse(native(d,"exchange-natural",payload),keys,False)
    verify_parent(d,natural,payload)
    cells={f"{p}/{t}":panel({k:v for k,v in natural.items() if k.endswith(f"/{p}/{t}")})
           for p,t in product((1,2),(0,1))}
    full=panel(natural)
    c=full["comparison"]
    margin_ok=all(x["comparison"]["extra_losses_by_140"]<=MAX_EXTRA_LOSS*x["baseline"]["n"] for x in cells.values())
    if c["score_outer_interval"][0]>=MIN_GAIN and margin_ok:
        route="validate_on_unseen_partner_before_training"
    elif full["upper_limit"]<MIN_GAIN:
        route="audit_opportunity_coverage"
    else:
        route="diagnose_teacher_or_context_before_training"
    summary=dict(archive=archive_summary, natural=full, blocks=cells,long=long_baseline(natural),
                 decision=route, early_loss_budget_ok=margin_ok)
    persist(d/"summary.json",encode(dict(summary=summary,archives=archive,openings=natural)),audit)
    print(json.dumps(summary,sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(sys.argv[1:]==["audit"])
