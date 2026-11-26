"""Paired single-seat takeover benchmark with strict archived prefixes."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import mean, median
import sys
import tarfile

from archive_action_screen import layout_group
from audit_human_timing import sanitized_tape
from audit_qq_archive import INPUT, LIMIT, archived_move, decode, initial_board
from compare_archive_cycle_exit_values import interval
from local_archive_inputs import archive_value
from cross_partner_deadline_student import encode, persist
from measure_deadline_opening import ROOT, SCORE, build, model_roster, require
from train_early_deadline_student import model_arguments, native

MODEL = (ROOT/"build/deadline-opening/train-public-exchange-student-9bc43e1e537e5e447099"/"replay_control.json")
LOCAL_AUDIT = ROOT/"build/replay-rule-variants/audit-replay-rule-variants-6a659a373631ad6cc804/summary.json"
CUTS = ("before_first","after_first","late","comeback")
ARMS = ("frozen","dispatcher","pressure","human_first")
ROOT_FIELDS = ("ply human_left seat human legal offered rank known_flag_takes visible_finish threat forced "
               "enemies quiet prefix").split()
END_FIELDS = ("plies outcome ending trace first first_enemy last_enemy corrections guards misses "
              "known_win earliest_visible_win max_quiet friendly_exit reverse").split()


def controls(events):
    if events[-1][0]!=245 or events[-1][1]!=5 or events[-1][3]!=4:
        return None
    flags,outs,ply,exits=[],set(),0,[]
    for i,event in enumerate(events):
        if event[0]==95:
            ply+=1
            continue
        if event[0]!=245:
            return None
        kind,seat,cause=event[1:4]
        if kind==3 and cause in (1,2) and seat not in outs:
            outs.add(seat)
            exits.append((seat,cause,ply))
            if cause==1:
                flags.append((seat,ply))
        elif not (kind==4 and seat in outs or kind==5 and i==len(events)-1):
            return None
    lost={s%2 for s in outs if (s+2)%4 in outs}
    if not flags or len(lost)!=1:
        return None
    loser=next(iter(lost))
    if flags[0][0]%2!=loser and exits[0][1]==1:
        return "comeback",ply,flags[0][1]
    enemies=[at for s,at in flags if s%2==loser]
    if len(enemies)==2 and enemies[-1]==ply:
        return "winning",ply,enemies[0]
    return None


def closed_prefix(layouts,seats,events,mark):
    board=initial_board(layouts)
    ply=0
    for e in events:
        if e[0]==95:
            actor,_=archived_move(board,e)
            if ply==mark:
                return actor
            ply+=1
        elif e[1]==3 and e[2] in seats:
            rank=seats.index(e[2])
            board={pos:piece for pos,piece in board.items() if piece[0]!=rank}
    return None


def source_frames():
    pilot_selection=archive_value("takeover_pilots")
    source=json.loads(INPUT.read_text(encoding="utf-8-sig"))
    local=json.loads(LOCAL_AUDIT.read_bytes())["sources"]
    local_families={row["family"] for row in local.values()}
    ledger,seen,pilots={},set(),set()
    stats=Counter()
    with tarfile.open(source["corpus"],"r:gz") as archive:
        for item in archive:
            name=item.name.rsplit("/",1)[-1]
            if not item.isfile() or not 412<=item.size<=LIMIT or name.startswith("._"):
                continue
            raw=archive.extractfile(item).read()
            try:
                key,layouts,seats,events=decode(raw)
            except ValueError:
                continue
            if key in seen:
                continue
            seen.add(key)
            info=controls(events)
            if info is None:
                continue
            group,total,first=info
            stats[group]+=1
            pilot=name in pilot_selection
            if group=="winning":
                try:
                    actor=closed_prefix(layouts,seats,events,first)
                except ValueError:
                    continue
                if actor is None:
                    continue
                # Some winners had lost a flag earlier. Determine the side from the
                # two eliminated seats in the terminal record instead of that event.
                out={e[2] for e in events if e[0]==245 and e[1]==3 and e[3] in (1,2)}
                loser=next(s%2 for s in out if (s+2)%4 in out)
                if seats[actor]%2==loser:
                    continue
                stats["immediate_winner_followup"]+=1
            family=layout_group(layouts).hex()
            if pilot:
                require(group=="winning" and total-first==pilot_selection[name],"pilot human timing differs")
                pilots.add(name)
            if not pilot and (key in local or family in local_families):
                stats["overlap_excluded"]+=1
                continue
            ledger[key]=dict(group=group,total=total,first=first,family=family,pilot=name if pilot else None,
                             tape=sanitized_tape(raw,events))
    stats["decoded_unique"]=len(seen)
    require(pilots==pilot_selection.keys(),"missing requested pilot")
    return ledger,stats


def choose(ledger):
    found={k:row for k,row in ledger.items() if row["pilot"]}
    families={r["family"] for r in found.values()}
    for group in ("winning","comeback"):
        candidates=[(k,row) for k,row in sorted(ledger.items()) if row["group"]==group and not row["pilot"]]
        limit=24 if group=="winning" else 12
        added=0
        for k,row in candidates:
            if row["family"] in families:
                continue
            families.add(row["family"])
            found[k]=row
            added+=1
            if added==limit:
                break
        require(added==limit,"missing independent layout cohort")
    return found


def analyze_native(stdout, selected, arms=(0,1,2,3)):
    all_cases={}
    rejected={}
    for line in stdout.splitlines():
        tag,key,*values=line.split()
        v=list(map(int,values))
        require(key in selected,"unregistered replay")
        if tag=="X":
            require(key not in rejected,"duplicate rejection")
            rejected[key]=v
        elif tag=="R":
            cut,*fields=v
            row=dict(zip(ROOT_FIELDS,fields,strict=True))
            row["runs"]={}
            require((key,cut) not in all_cases,"duplicate cut")
            all_cases[key,cut]=row
        else:
            cut,rep,arm,*fields=v
            require(tag=="F" and f"{rep}/{arm}" not in all_cases[key,cut]["runs"],"duplicate branch")
            all_cases[key,cut]["runs"][f"{rep}/{arm}"]=dict(zip(END_FIELDS,fields,strict=True))
    for key,meta in selected.items():
        expected={3} if meta["group"]=="comeback" else {0,1,2}
        returned={c for s,c in all_cases if s==key}
        require(returned==(set() if key in rejected else expected),"incomplete selected source")
        if key in rejected:
            require(not meta["pilot"],"required pilot failed to restore")
            continue
        for cut in returned:
            r=all_cases[key,cut]
            require(r["ply"]+r["human_left"]==meta["total"] and r["human_left"]>0 and
                    (cut!=1 or r["ply"]==meta["first"]) and r["legal"]>0,"wrong archived snapshot or time")
            reps=2 if meta["pilot"] else 4
            require(set(r["runs"])=={f"{rep}/{arm}" for rep in range(reps) for arm in arms},
                    "incomplete continuation roster")
            for rep in range(reps):
                branches=[r["runs"][f"{rep}/{arm}"] for arm in arms]
                a=branches[0]
                h=r["runs"].get(f"{rep}/3")
                require(not h or h["first"]==r["human"],"forced arm did not use the recorded first move")
                if h and a["first"]==h["first"]:
                    require(a["trace"]==h["trace"] and a["outcome"]==h["outcome"] and a["plies"]==h["plies"],
                            "identical first move changed the branch")
                require(all(z["outcome"] in SCORE and z["plies"]>0 and 1<=z["ending"]<=5 for z in branches),
                        "invalid settlement ledger")
    return all_cases,rejected


def speed(rows, arm):
    tally=Counter()
    deltas=[]
    first,remaining=[],[]
    for _,r in rows:
        for k,z in r["runs"].items():
            if int(k[-1])!=arm:
                continue
            if z["outcome"]==2:
                gap=z["plies"]-r["human_left"]
                deltas.append(gap)
                tally["faster" if gap<0 else "slower" if gap>0 else "tied"]+=1
                if z["first_enemy"]:
                    first.append(z["first_enemy"])
                    remaining.append(z["plies"]-z["first_enemy"])
            else:
                tally[{-1:"cutoff",0:"loss",1:"draw"}[z["outcome"]]]+=1
    keys=("faster","tied","slower","draw","loss","cutoff")
    return dict(tally={k:tally[k] for k in keys},gap_median=median(deltas) if deltas else None,
                faster_median=median([-x for x in deltas if x<0]) if any(x<0 for x in deltas) else None,
                slower_median=median([x for x in deltas if x>0]) if any(x>0 for x in deltas) else None,
                max_delay=max(deltas) if deltas else None,
                first_retirement_median=median(first) if first else None,
                remainder_after_next_retirement_median=median(remaining) if remaining else None)


def comparison(rows, challenger):
    adjusted=[]
    counts=Counter()
    deltas=[]
    for key,row in rows:
        block=[]
        for rp in sorted(int(k.split("/")[0]) for k in row["runs"] if k.endswith("/0")):
            a,b=(row["runs"][f"{rp}/{i}"] for i in (0,challenger))
            x,y=(SCORE[z["outcome"]] for z in (a,b))
            block.append(((0 if y is None else y)-(1 if x is None else x),
                          (1 if y is None else y)-(0 if x is None else x)))
            counts["win_to_loss"]+=a["outcome"]==2 and b["outcome"]==0
            counts["loss_to_win"]+=a["outcome"]==0 and b["outcome"]==2
            counts["own_risk"]+=b["friendly_exit"]>0
            counts["corrections"]+=b["corrections"]
            counts["guards"]+=b["guards"]
            counts["missed_visible_win"]+=a["misses"]
            if a["outcome"]==b["outcome"]==2:
                deltas.append(b["plies"]-a["plies"])
        adjusted.append(tuple(map(mean,zip(*block))))
    return dict(bounds=tuple(map(mean,zip(*adjusted))) if adjusted else None,
                interval=[interval([x[0] for x in adjusted])[0],interval([x[1] for x in adjusted])[1]]
                if adjusted else None,counts=dict(counts),paired_win_median=median(deltas) if deltas else None)


def summarize(cases, selected):
    metrics={}
    for cohort in ("pilot","confirm","comeback"):
        staged=[(k,c,r) for (k,c),r in cases.items() if
                cohort==("pilot" if selected[k]["pilot"] else "comeback" if c==3 else "confirm")]
        for cut in range(len(CUTS)):
            rows=[(k,r) for k,c,r in staged if c==cut]
            if not rows:
                continue
            contexts={}
            for name,test in (("visible_flag_access",lambda r:r["known_flag_takes"]>0),
                              ("no_visible_flag_access",lambda r:r["known_flag_takes"]==0),
                              ("visible_defensive_threat",lambda r:r["threat"]>0)):
                group=[(k,r) for k,r in rows if test(r)]
                contexts[name]=dict(n=len(group),pressure=comparison(group,2))
            metrics[cohort+"/"+CUTS[cut]]=dict(
                roots=len(rows),human_median=median(r["human_left"] for _,r in rows),
                listed=sum(r["offered"] for _,r in rows),
                profiles={name:speed(rows,arm) for arm,name in enumerate(ARMS)},
                paired={name:comparison(rows,arm) for arm,name in enumerate(ARMS) if arm},contexts=contexts)
    return metrics


def run(directory, selected, params, pilot):
    rows={key:r for key,r in selected.items() if bool(r["pilot"])==pilot}
    payload=[params]
    for key,r in sorted(rows.items()):
        payload.append(f'{key} {int(r["group"]=="comeback")} {2 if pilot else 4} {r["tape"]}')
    return analyze_native(native(directory,"winner-takeover","\n".join(payload)),rows)


def main(audit):
    directory=ROOT/"build/human-winner-takeover"/os.environ["JUNQI_SERIAL_KEY"]
    if not audit:
        directory.mkdir(parents=True,exist_ok=True)
        build(directory)
    ledger,index=source_frames()
    selected=choose(ledger)
    roster=model_roster()
    weights=[w for name in ("candidate","historical") for w in roster[name]["weights"]]
    model=json.loads(MODEL.read_bytes())
    params=" ".join(map(str,(1,1,*weights)))+" "+model_arguments(model,"replay_control")
    pilot,missing=run(directory,selected,params,True)
    require(len(pilot)==9 and not missing,"pilot did not verify all cuts")
    other,rejected=run(directory,selected,params,False)
    cases=pilot|other
    panels=summarize(cases,selected)
    manifest={key: {k:v for k,v in row.items() if k!="tape"} | {"tape_sha256":sha256(bytes.fromhex(row["tape"])).hexdigest()}
              for key,row in selected.items()}
    result=dict(panels=panels, manifest=manifest, inventory=dict(index), rejected=rejected,
                cases=[dict(source=k,cut=CUTS[c],**r) for (k,c),r in cases.items()])
    persist(directory/"summary.json",encode(result),audit)
    print(json.dumps(dict(inventory=index,accepted=len(cases),rejected=rejected,panels=panels),sort_keys=True))
    print("JUNQI_VERDICT accepted")


if __name__=="__main__":
    require(sys.argv[1:] in ([],["audit"]),"expected no arguments or audit")
    main(sys.argv[1:]==["audit"])
